# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	E. Barszcz
#	P. Frederickson
#	A. Woo
#	M. Yarrow
# 
# ------------------------------------------------------------------------------
# 
# The serial C++ version is a translation of the original NPB 3.4.1
# Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
# 
# Authors of the C++ code: 
# 	Dalvan Griebler <dalvangriebler@gmail.com>
# 	Gabriell Araujo <hexenoften@gmail.com>
# 	Júnior Löff <loffjh@gmail.com>
#
# ------------------------------------------------------------------------------
# 
# The CUDA version is a parallel implementation of the serial C++ version
# CUDA version: https://github.com/GMAP/NPB-GPU/tree/master/CUDA
# 
# Authors of the C++ code: 
# 	Gabriell Araujo <hexenoften@gmail.com>
#
# ------------------------------------------------------------------------------
#
# The CUDA Python version is a translation of the NPB CUDA version
# CUDA Python version: https://github.com/PYTHON
# 
# Authors of the CUDA Python code:
#	LUPS (Laboratory of Ubiquitous and Parallel Systems)
#	UFPEL (Federal University of Pelotas)
#	Pelotas, Rio Grande do Sul, Brazil
#
# ------------------------------------------------------------------------------


import argparse
import sys
import os
import numpy
import math
from math import ceil as m_ceil
import numba
from numba import njit
from numba import cuda
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
from c_randdp import randlc
from c_randdp import vranlc
import c_timers
import c_print_results

sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')))
import gpu_config


# Global variables
NM = 0 # actual dimension including ghost cells for communications
NV = 0 # size of rhs array
NR = 0 # size of residual array
MAXLEVEL = 0 # maximum number of levels
M = 0 # set at m=1024, can handle cases up to 1024^3 case
MM = 10
A = pow(5.0, 13.0)
X = 314159265.0
PROFILING_TOTAL_TIME = 0
PROFILING_COMM3 = 1
PROFILING_INTERP = 2
PROFILING_NORM2U3 =3
PROFILING_PSINV = 4
PROFILING_RESID = 5
PROFILING_RPRJ3 = 6
PROFILING_ZERO3 = 7

nx = None
ny = None
nz = None
m1 = None
m2 = None
m3 = None
ir = None
debug_vec = numpy.repeat(npbparams.DEBUG_DEFAULT, 8)
u = None
v = None
r = None

is1 = 0
is2 = 0
is3 = 0
ie1 = 0
ie2 = 0
ie3 = 0
lt = 0
lb = 0

# GPU variables
a_device = None
c_device = None
u_device = None
v_device = None
r_device = None

threads_per_block_on_comm3 = 0
threads_per_block_on_interp = 0
threads_per_block_on_norm2u3 = 0
threads_per_block_on_psinv = 0
threads_per_block_on_resid = 0
threads_per_block_on_rprj3 = 0
threads_per_block_on_zero3 = 0

stream = 0
size_shared_data_on_interp = 0
size_shared_data_on_norm2u3 = 0
size_shared_data_on_psinv = 0
size_shared_data_on_resid = 0
size_shared_data_on_rprj3 = 0

gpu_device_id = 0
total_devices = 0
device_prop = None

def set_global_variables():
	global NM, NV, NR, MAXLEVEL, M
	global nx, ny, nz, m1, m2, m3, ir
	global u, v, r
	
	NM = 2 + (1 << npbparams.LM)
	NV = npbparams.ONE * (2 + (1 << npbparams.NDIM1)) * (2 + (1 << npbparams.NDIM2)) * (2 + (1 << npbparams.NDIM3))
	NR = int( (NV + NM*NM + 5*NM + 7*npbparams.LM + 6) / 7 ) * 8
	MAXLEVEL = npbparams.LT_DEFAULT + 1 
	M = NM + 1 
	
	nx = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	ny = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	nz = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	m1 = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	m2 = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	m3 = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	ir = numpy.empty(MAXLEVEL+1, dtype=numpy.int32)
	
	u = numpy.empty(NR, dtype=numpy.float64)
	v = numpy.empty(NV, dtype=numpy.float64)
	r = numpy.empty(NR, dtype=numpy.float64)
#END set_global_variables()


def setup_gpu(a, c):
	global gpu_device_id, total_devices
	global device_prop
	
	global threads_per_block_on_comm3, threads_per_block_on_interp, threads_per_block_on_norm2u3
	global threads_per_block_on_psinv, threads_per_block_on_resid, threads_per_block_on_rprj3
	global threads_per_block_on_zero3
	
	global size_shared_data_on_interp, size_shared_data_on_norm2u3, size_shared_data_on_psinv
	global size_shared_data_on_resid, size_shared_data_on_rprj3
	
	global a_device, c_device, u_device, v_device, r_device
	
	# amount of available devices 
	devices = cuda.gpus
	total_devices = len(devices)

	# define gpu_device
	if total_devices == 0:
		print("\n\n\nNo Nvidia GPU found!\n")
		sys.exit()
	elif gpu_config.GPU_DEVICE >= 0 and gpu_config.GPU_DEVICE < total_devices:
		gpu_device_id = gpu_config.GPU_DEVICE
	else: 
		gpu_device_id = 0

	cuda.select_device(gpu_device_id)
	device_prop = cuda.get_current_device()
	
	# define threads_per_block
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_COMM3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_comm3 = aux_threads_per_block
	else: 
		threads_per_block_on_comm3 = device_prop.WARP_SIZE
	
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_INTERP
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_interp = aux_threads_per_block
	else: 
		threads_per_block_on_interp = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_NORM2U3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_norm2u3 = aux_threads_per_block
	else: 
		threads_per_block_on_norm2u3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_PSINV
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_psinv = aux_threads_per_block
	else: 
		threads_per_block_on_psinv = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_RESID
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_resid = aux_threads_per_block
	else: 
		threads_per_block_on_resid = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_RPRJ3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_rprj3 = aux_threads_per_block
	else: 
		threads_per_block_on_rprj3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.MG_THREADS_PER_BLOCK_ON_ZERO3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_zero3 = aux_threads_per_block
	else: 
		threads_per_block_on_zero3 = device_prop.WARP_SIZE
		
	a_device = cuda.to_device(a)
	c_device = cuda.to_device(c)
	u_device = cuda.to_device(u)
	v_device = cuda.to_device(v)
	r_device = cuda.to_device(r)
	
	size_shared_data_on_interp = 3 * M * a_device.dtype.itemsize
	size_shared_data_on_norm2u3 = 2 * threads_per_block_on_norm2u3 * a_device.dtype.itemsize
	size_shared_data_on_psinv = 2 * M * a_device.dtype.itemsize
	size_shared_data_on_resid = 2 * M * a_device.dtype.itemsize
	size_shared_data_on_rprj3 = 2 * M * a_device.dtype.itemsize
#END setup_gpu()


#*****************************************************************
#************************* GPU FUNCTIONS *************************
#*****************************************************************
@cuda.jit('void(float64[:], float64[:], int32, int32, int32, int32, int32, int32, int32, int32)')
def interp_gpu_kernel(z_device,
					u_device,
					mm1, 
					mm2, 
					mm3,
					n1, 
					n2, 
					n3,
					amount_of_work,
					M_aux):
	check = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if check >= amount_of_work:
		return

	#double* z1 = (double*)(extern_share_data);
	#double* z2 = (double*)(z1+M);
	#double* z3 = (double*)(z2+M);
	z1 = cuda.shared.array(shape=0, dtype=numba.float64)
	z2 = z1[M_aux:]
	z3 = z2[M_aux:]

	i3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	i2 = cuda.blockIdx.x
	i1 = cuda.threadIdx.x

	z1[i1] = z_device[i3*mm2*mm1+(i2+1)*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1]
	z2[i1] = z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z_device[i3*mm2*mm1+i2*mm1+i1]
	z3[i1] = z_device[(i3+1)*mm2*mm1+(i2+1)*mm1+i1]+z_device[(i3+1)*mm2*mm1+i2*mm1+i1]+z1[i1]

	cuda.syncthreads()
	if i1 < mm1-1:
		z321 = z_device[i3*mm2*mm1+i2*mm1+i1]
		u_device[2*i3*n2*n1+2*i2*n1+2*i1] += z321
		u_device[2*i3*n2*n1+2*i2*n1+2*i1+1] += 0.5*(z_device[i3*mm2*mm1+i2*mm1+i1+1]+z321)
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1] += 0.5*z1[i1]
		u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1] += 0.25*(z1[i1]+z1[i1+1])
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1] += 0.5*z2[i1]
		u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1] += 0.25*(z2[i1]+z2[i1+1])
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1] += 0.25*z3[i1]
		u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1] += 0.125*(z3[i1]+z3[i1+1])
#END interp_gpu_kernel()


#static void interp_gpu(double* z_device, int mm1, int mm2, int mm3, double* u_device, int n1, int n2, int n3, int k)
def interp_gpu(z_device, 
			mm1, 
			mm2, 
			mm3, 
			u_device, 
			n1, 
			n2, 
			n3, 
			k):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_INTERP)

	if n1 != 3 and n2 != 3 and n3 != 3:
		amount_of_work_x = (mm2 - 1) * mm1
		amount_of_work_y = mm3 - 1
		threads_per_block_x = threads_per_block_on_interp
		if threads_per_block_on_interp != mm1:
			threads_per_block_x = mm1
		threads_per_block_y = 1
		threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
		blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
						m_ceil(amount_of_work_y / threadsPerBlock[1]),
						1)
		interp_gpu_kernel[blocksPerGrid, 
			threadsPerBlock,
			stream,
			size_shared_data_on_interp](z_device,
										u_device,
										mm1,
										mm2,
										mm3,
										n1,
										n2,
										n3,
										amount_of_work_x,
										M)
	#END if n1 != 3 and n2 != 3 and n3 != 3:

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_INTERP)
#END interp_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], int32, int32, int32, int32, int32)')
def psinv_gpu_kernel(r,
					u,
					c,
					n1,
					n2,
					n3,
					amount_of_work,
					M_aux):
	check = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if check >= amount_of_work:
		return

	#double* r1 = (double*)(extern_share_data);
	#double* r2 = (double*)(r1+M);
	r1 = cuda.shared.array(shape=0, dtype=numba.float64)
	r2 = r1[M_aux:]

	i3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	i2 = cuda.blockIdx.x + 1
	lid = cuda.threadIdx.x

	for i1 in range(lid, n1, cuda.blockDim.x):
		r1[i1] = ( r[i3*n2*n1+(i2-1)*n2+i1]
			+r[i3*n2*n1+(i2+1)*n1+i1]
			+r[(i3-1)*n2*n1+i2*n1+i1]
			+r[(i3+1)*n2*n1+i2*n1+i1] )
		r2[i1] = ( r[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+r[(i3+1)*n2*n1+(i2+1)*n1+i1] )
	
	cuda.syncthreads()
	
	for i1 in range(lid+1, n1-1, cuda.blockDim.x):
		u[i3*n2*n1+i2*n1+i1] = ( u[i3*n2*n1+i2*n1+i1]
			+c[0]*r[i3*n2*n1+i2*n1+i1]
			+c[1]*(r[i3*n2*n1+i2*n1+i1-1]
					+r[i3*n2*n1+i2*n1+i1+1]
					+r1[i1])
			+c[2]*(r2[i1]+r1[i1-1]+r1[i1+1]) )
#END psinv_gpu_kernel()


#static void psinv_gpu(double* r_device, double* u_device, int n1, int n2, int n3, double* c_device, int k)
def psinv_gpu(r_device, 
			u_device, 
			n1, 
			n2, 
			n3, 
			c_device, 
			k):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_PSINV)

	threads_per_block = threads_per_block_on_psinv if (n1 > threads_per_block_on_psinv) else n1
	amount_of_work_x = (n2 - 2) * threads_per_block
	amount_of_work_y = n3 - 2
	threads_per_block_x = threads_per_block
	threads_per_block_y = 1
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1)

	psinv_gpu_kernel[blocksPerGrid, 
		threadsPerBlock,
		stream,
		size_shared_data_on_psinv](r_device,
								u_device,
								c_device,
								n1,
								n2,
								n3,
								amount_of_work_x,
								M)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_PSINV)

	# --------------------------------------------------------------------
	# exchange boundary points
	# --------------------------------------------------------------------
	comm3_gpu(u_device, n1, n2, n3, k)
#END psinv_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, int32)')
def zero3_gpu_kernel(z, 
					n1, 
					n2, 
					n3, 
					amount_of_work):
	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if thread_id >= (n1*n2*n3): 
		return
	z[thread_id] = 0.0
#END zero3_gpu_kernel()


#static void zero3_gpu(double* z_device, int n1, int n2, int n3)
def zero3_gpu(z_device, 
			n1, 
			n2, 
			n3):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ZERO3)

	threads_per_block = threads_per_block_on_zero3
	amount_of_work = n1 * n2 * n3
	blocks_per_grid = m_ceil(amount_of_work / threads_per_block)

	zero3_gpu_kernel[blocks_per_grid, threads_per_block](z_device,
														n1,
														n2,
														n3,
														amount_of_work)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ZERO3)
#END zero3_gpu()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, int32, int32, int32, int32, int32, int32, int32, int32)')
def rprj3_gpu_kernel(r_device,
					s_device,
					m1k,
					m2k,
					m3k,
					m1j,
					m2j,
					m3j,
					d1, 
					d2, 
					d3,
					amount_of_work,
					M_aux):
	check = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if check >= amount_of_work:
		return

	#double* x1 = (double*)(extern_share_data);
	#double* y1 = (double*)(x1+M);
	x1 = cuda.shared.array(shape=0, dtype=numba.float64)
	y1 = x1[M_aux:]

	j3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	j2 = cuda.blockIdx.x + 1
	j1 = cuda.threadIdx.x + 1

	i3 = 2 * j3 - d3
	i2 = 2 * j2 - d2
	i1 = 2 * j1 - d1
	x1[i1] = (r_device[(i3+1)*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+1)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1] )
	y1[i1] = (r_device[i3*m2k*m1k+i2*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+i2*m1k+i1]
		+r_device[i3*m2k*m1k+(i2+2)*m1k+i1]
		+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1] )
	cuda.syncthreads()
	if j1 < m1j-1:
		i1 = 2 * j1 - d1
		y2 = ( r_device[i3*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+i2*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+2)*m1k+i1+1] )
		x2 = ( r_device[(i3+1)*m2k*m1k+i2*m1k+i1+1]
			+r_device[(i3+1)*m2k*m1k+(i2+2)*m1k+i1+1]
			+r_device[i3*m2k*m1k+(i2+1)*m1k+i1+1]
			+r_device[(i3+2)*m2k*m1k+(i2+1)*m1k+i1+1] )
		s_device[j3*m2j*m1j+j2*m1j+j1] = (
			0.5*r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+1]
			+0.25*(r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1]
					+r_device[(i3+1)*m2k*m1k+(i2+1)*m1k+i1+2]+x2)
			+0.125*(x1[i1]+x1[i1+2]+y2)
			+0.0625*(y1[i1]+y1[i1+2]) )
#END rprj3_gpu_kernel()


#static void rprj3_gpu(double* r_device, int m1k, int m2k, int m3k, double* s_device, int m1j, int m2j, int m3j, int k)
def rprj3_gpu(r_device, 
			m1k, 
			m2k, 
			m3k, 
			s_device, 
			m1j, 
			m2j, 
			m3j, 
			k):
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RPRJ3)
	
	d1, d2, d3 = 0, 0, 0
	if m1k == 3:
		d1 = 2
	else:
		d1 = 1

	if m2k == 3:
		d2 = 2
	else:
		d2 = 1

	if m3k == 3:
		d3 = 2
	else:
		d3 = 1

	amount_of_work_x = (m2j - 2) * (m1j - 1)
	amount_of_work_y = m3j - 2
	
	threads_per_block_x = threads_per_block_on_rprj3
	if threads_per_block_on_rprj3 != m1j-1:
		threads_per_block_x = m1j-1
	threads_per_block_y = 1
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1 )

	rprj3_gpu_kernel[blocksPerGrid, 
		threadsPerBlock,
		stream,
		size_shared_data_on_rprj3](r_device,
								s_device,
								m1k,
								m2k,
								m3k,
								m1j,
								m2j,
								m3j,
								d1,
								d2,
								d3,
								amount_of_work_x,
								M)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RPRJ3)
		
	j = k-1
	comm3_gpu(s_device, m1j, m2j, m3j, j)
#END rprj3_gpu()


#static void mg3P_gpu(double* u_device, double* v_device, double* r_device, double* a_device, double* c_device, 
#int n1, int n2, int n3, int k)
def mg3P_gpu(u_device, 
			v_device, 
			r_device, 
			a_device, 
			c_device, 
			n1, 
			n2, 
			n3, 
			k):
	# --------------------------------------------------------------------
	# down cycle.
	# restrict the residual from the find grid to the coarse
	# -------------------------------------------------------------------
	for k in range(lt, lb+1-1, -1):
		j = k - 1
		rprj3_gpu(r_device[ir[k]:], m1[k], m2[k], m3[k], r_device[ir[j]:], m1[j], m2[j], m3[j], k)

	k = lb
	
	# --------------------------------------------------------------------
	# compute an approximate solution on the coarsest grid
	# --------------------------------------------------------------------
	zero3_gpu(u_device[ir[k]:], m1[k], m2[k], m3[k])
	psinv_gpu(r_device[ir[k]:], u_device[ir[k]:], m1[k], m2[k], m3[k], c_device, k)
	
	for k in range(lb+1, lt-1+1):
		j = k - 1
		# --------------------------------------------------------------------
		# prolongate from level k-1  to k
		# -------------------------------------------------------------------
		zero3_gpu(u_device[ir[k]:], m1[k], m2[k], m3[k])
		interp_gpu(u_device[ir[j]:], m1[j], m2[j], m3[j], u_device[ir[k]:], m1[k], m2[k], m3[k], k)
		# --------------------------------------------------------------------
		# compute residual for level k
		# --------------------------------------------------------------------
		resid_gpu(u_device[ir[k]:], r_device[ir[k]:], r_device[ir[k]:], m1[k], m2[k], m3[k], a_device, k)
		# --------------------------------------------------------------------
		# apply smoother
		# --------------------------------------------------------------------
		psinv_gpu(r_device[ir[k]:], u_device[ir[k]:], m1[k], m2[k], m3[k], c_device, k)

	j = lt - 1
	k = lt
	
	interp_gpu(u_device[ir[j]:], m1[j], m2[j], m3[j], u_device, n1, n2, n3, k)
	resid_gpu(u_device, v_device, r_device, n1, n2, n3, a_device, k)
	psinv_gpu(r_device, u_device, n1, n2, n3, c_device, k)
#END mg3P_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:], float64[:], int32, int32)')
def norm2u3_gpu_kernel(r,
					n1, 
					n2, 
					n3,
					res_sum,
					res_max,
					number_of_blocks_on_x_axis,
					amount_of_work):
	check = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if check>=amount_of_work:
		return

	#double* scratch_sum = (double*)(extern_share_data);
	#double* scratch_max = (double*)(scratch_sum+blockDim.x);
	scratch_sum = cuda.shared.array(shape=0, dtype=numba.float64)
	scratch_max = scratch_sum[cuda.blockDim.x:]

	i3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	i2 = cuda.blockIdx.x + 1
	i1 = cuda.threadIdx.x + 1

	s = 0.0
	my_rnmu = 0.0

	while i1 < (n1-1):
		r321 = r[i3*n2*n1+i2*n1+i1]
		s = s + r321 * r321
		a = abs(r321)
		my_rnmu = a if (a > my_rnmu) else my_rnmu
		i1 += cuda.blockDim.x
	
	lid = cuda.threadIdx.x
	scratch_sum[lid] = s
	scratch_max[lid] = my_rnmu

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if lid < i:
			scratch_sum[lid] += scratch_sum[lid+i]
			scratch_max[lid] = scratch_max[lid] if (scratch_max[lid] > scratch_max[lid+i]) else scratch_max[lid+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if lid == 0:
		idx = cuda.blockIdx.y * number_of_blocks_on_x_axis + cuda.blockIdx.x
		res_sum[idx] = scratch_sum[0]
		res_max[idx] = scratch_max[0]
#END norm2u3_gpu_kernel()


#static void norm2u3_gpu(r_device, int n1, int n2, int n3, int nx, int ny, int nz)
def norm2u3_gpu(r_device, 
				n1, 
				n2, 
				n3, 
				nx, 
				ny, 
				nz):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_NORM2U3)

	dn = 1.0 * nx * ny * nz
	s = 0.0
	max_rnmu = 0.0

	threads_per_block_x = threads_per_block_on_norm2u3
	threads_per_block_y = 1
	amount_of_work_x = (n2 - 2) * threads_per_block_x
	amount_of_work_y = n3 - 2
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1)

	temp_size = int((amount_of_work_x * amount_of_work_y) / (threads_per_block_x * threads_per_block_y))
	data_device = cuda.device_array(2*temp_size, numpy.float64)
	sum_device = data_device
	max_device = data_device[temp_size:]

	norm2u3_gpu_kernel[blocksPerGrid, 
		threadsPerBlock,
		stream,
		size_shared_data_on_norm2u3](r_device,
									 n1,
									 n2,
									 n3,
									 sum_device,
									 max_device,
									 blocksPerGrid[0],
									 amount_of_work_x)

	data_host = data_device.copy_to_host()
	sum_host = data_host
	max_host = data_host[temp_size:]

	for j in range(temp_size):
		s = s + sum_host[j]
		if max_rnmu < max_host[j]:
			max_rnmu = max_host[j]

	rnmu = max_rnmu
	rnm2 = math.sqrt(s / dn)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_NORM2U3)
	
	return rnm2, rnmu
#END norm2u3_gpu()

@cuda.jit('void(float64[:], int32, int32, int32, int32)')
def comm3_gpu_kernel_3(u, 
					n1, 
					n2, 
					n3, 
					amount_of_work):
	i2 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	i1 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if i1 >= n1:
		return

	u[0*n2*n1+i2*n1+i1] = u[(n3-2)*n2*n1+i2*n1+i1]
	u[(n3-1)*n2*n1+i2*n1+i1] = u[1*n2*n1+i2*n1+i1]
#END comm3_gpu_kernel_3()


@cuda.jit('void(float64[:], int32, int32, int32, int32)')
def comm3_gpu_kernel_2(u, 
					n1, 
					n2, 
					n3, 
					amount_of_work):
	i3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	i1 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if i1 >= n1:
		return

	u[i3*n2*n1+0*n1+i1] = u[i3*n2*n1+(n2-2)*n1+i1]
	u[i3*n2*n1+(n2-1)*n1+i1] = u[i3*n2*n1+1*n1+i1]
#END comm3_gpu_kernel_2()


@cuda.jit('void(float64[:], int32, int32, int32, int32)')
def comm3_gpu_kernel_1(u, 
					n1, 
					n2, 
					n3, 
					amount_of_work):
	i3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	i2 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1

	if i2 >= n2-1:
		return

	u[i3*n2*n1+i2*n1+0] = u[i3*n2*n1+i2*n1+n1-2]
	u[i3*n2*n1+i2*n1+n1-1] = u[i3*n2*n1+i2*n1+1]
#END comm3_gpu_kernel_1()


#static void comm3_gpu(double* u_device, int n1, int n2, int n3, int kk)
def comm3_gpu(u_device, 
			n1, 
			n2, 
			n3, 
			kk):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_COMM3)

	threads_per_block_x = threads_per_block_on_comm3
	threads_per_block_y = 1

	# axis = 1
	amount_of_work_x = m_ceil((n2-2) / threads_per_block_x) * threads_per_block_x
	amount_of_work_y = n3 - 2
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1)
	comm3_gpu_kernel_1[blocksPerGrid,
		threadsPerBlock](u_device,
						n1,
						n2,
						n3,
						amount_of_work_x)

	# axis = 2
	amount_of_work_x = m_ceil(n1 / threads_per_block_x) * threads_per_block_x
	amount_of_work_y = n3-2
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1)
	comm3_gpu_kernel_2[blocksPerGrid,
		threadsPerBlock](u_device,
						n1,
						n2,
						n3,
						amount_of_work_x)

	# axis = 3
	amount_of_work_x = m_ceil(n1 / threads_per_block_x) * threads_per_block_x
	amount_of_work_y = n2
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1)
	comm3_gpu_kernel_3[blocksPerGrid,
		threadsPerBlock](u_device,
						n1,
						n2,
						n3,
						amount_of_work_x)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_COMM3)
#endif
#END comm3_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32, int32, int32, int32, int32)')
def resid_gpu_kernel(u,
					v,
					r,
					a,
					n1,
					n2,
					n3,
					amount_of_work,
					M_aux):
	check = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if check >= amount_of_work:
		return

	#double* u1=(double*)(extern_share_data)
	#double* u2=(double*)(u1+M)
	u1 = cuda.shared.array(shape=0, dtype=numba.float64)
	u2 = u1[M_aux:]

	i3 = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	i2 = cuda.blockIdx.x + 1
	lid = cuda.threadIdx.x

	for i1 in range(lid, n1, cuda.blockDim.x):
		u1[i1] = ( u[i3*n2*n1+(i2-1)*n1+i1]
			+u[i3*n2*n1+(i2+1)*n1+i1]
			+u[(i3-1)*n2*n1+i2*n1+i1]
			+u[(i3+1)*n2*n1+i2*n1+i1] )
		u2[i1] = ( u[(i3-1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3-1)*n2*n1+(i2+1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2-1)*n1+i1]
			+u[(i3+1)*n2*n1+(i2+1)*n1+i1] )
	
	cuda.syncthreads()
	
	for i1 in range(lid+1, n1-1, cuda.blockDim.x):
		r[i3*n2*n1+i2*n1+i1] = ( v[i3*n2*n1+i2*n1+i1]
			-a[0]*u[i3*n2*n1+i2*n1+i1]
			-a[2]*(u2[i1]+u1[i1-1]+u1[i1+1])
			-a[3]*(u2[i1-1]+u2[i1+1] ) )
#END resid_gpu_kernel()


#static void resid_gpu(double* u_device, double* v_device, double* r_device, int n1, int n2, int n3, double* a_device, int k)
def resid_gpu(u_device,
			v_device,
			r_device,
			n1,
			n2,
			n3,
			a_device,
			k):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RESID)

	threads_per_block = threads_per_block_on_resid if n1 > threads_per_block_on_resid else n1
	amount_of_work_x = (n2 - 2) * threads_per_block
	amount_of_work_y = n3 - 2
	threads_per_block_x = threads_per_block
	threads_per_block_y = 1
	
	threadsPerBlock = (threads_per_block_x, threads_per_block_y, 1)
	blocksPerGrid = ( m_ceil(amount_of_work_x / threadsPerBlock[0]),
					m_ceil(amount_of_work_y / threadsPerBlock[1]),
					1 )

	resid_gpu_kernel[blocksPerGrid, 
		threadsPerBlock,
		stream,
		size_shared_data_on_resid](u_device,
								v_device,
								r_device,
								a_device,
								n1,
								n2,
								n3,
								amount_of_work_x,
								M)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RESID)

	# --------------------------------------------------------------------
	# exchange boundary data
	# --------------------------------------------------------------------
	comm3_gpu(r_device, n1, n2, n3, k) 
#END resid_gpu()


#*****************************************************************
#************************* CPU FUNCTIONS *************************
#*****************************************************************
# --------------------------------------------------------------------
# interp adds the trilinear interpolation of the correction
# from the coarser grid to the current approximation: u = u + Qu'
#     
# observe that this  implementation costs  16A + 4M, where
# A and M denote the costs of addition and multiplication.  
# note that this vectorizes, and is also fine for cache 
# based machines. vector machines may get slightly better 
# performance however, with 8 separate "do i1" loops, rather than 4.
# --------------------------------------------------------------------
#static void interp(void* pointer_z, int mm1, int mm2, int mm3, void* pointer_u, int n1, int n2, int n3, int k)
@njit
def interp(pointer_z, mm1, mm2, mm3, pointer_u, n1, n2, n3, k):
	#double (*z)[mm2][mm1] = (double (*)[mm2][mm1])pointer_z;
	#double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
	z = pointer_z #(i3*mm2+i2)*mm1 + i1
	u = pointer_u #(i3*n2+i2)*n1 + i1

	# --------------------------------------------------------------------
	# note that m = 1037 in globals.h but for this only need to be
	# 535 to handle up to 1024^3
	# integer m
	# parameter( m=535 )
	# --------------------------------------------------------------------
	z1 = numpy.empty(shape=M, dtype=numpy.float64)
	z2 = numpy.empty(shape=M, dtype=numpy.float64)
	z3 = numpy.empty(shape=M, dtype=numpy.float64)

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_INTERP)
	if n1 != 3 and n2 != 3 and n3 != 3:
		for i3 in range(mm3-1):
			for i2 in range(mm2-1):
				for i1 in range(mm1):
					z1[i1] = z[(i3*mm2+(i2+1))*mm1 + i1] + z[(i3*mm2+i2)*mm1 + i1] #z[i3][i2+1][i1] + z[i3][i2][i1]
					z2[i1] = z[((i3+1)*mm2+i2)*mm1 + i1] + z[(i3*mm2+i2)*mm1 + i1] #z[i3+1][i2][i1] + z[i3][i2][i1]
					z3[i1] = z[((i3+1)*mm2+(i2+1))*mm1 + i1] + z[((i3+1)*mm2+i2)*mm1 + i1] + z1[i1] #z[i3+1][i2+1][i1] + z[i3+1][i2][i1] + z1[i1]

				for i1 in range(mm1-1):
					#u[2*i3][2*i2][2*i1] = u[2*i3][2*i2][2*i1] + z[i3][i2][i1]
					u[((2*i3)*n2+(2*i2))*n1 + (2*i1)] = u[((2*i3)*n2+(2*i2))*n1 + (2*i1)] + z[(i3*mm2+i2)*mm1 + i1]
					#u[2*i3][2*i2][2*i1+1] = u[2*i3][2*i2][2*i1+1] +0.5*(z[i3][i2][i1+1]+z[i3][i2][i1])
					u[((2*i3)*n2+(2*i2))*n1 + (2*i1+1)] = u[((2*i3)*n2+(2*i2))*n1 + (2*i1+1)] + 0.5*(z[(i3*mm2+i2)*mm1 + (i1+1)]+z[(i3*mm2+i2)*mm1 + i1])

				for i1 in range(mm1-1):
					#u[2*i3][2*i2+1][2*i1] = u[2*i3][2*i2+1][2*i1] +0.5 * z1[i1]
					u[((2*i3)*n2+(2*i2+1))*n1 + (2*i1)] = u[((2*i3)*n2+(2*i2+1))*n1 + (2*i1)] +0.5 * z1[i1]
					#u[2*i3][2*i2+1][2*i1+1] = u[2*i3][2*i2+1][2*i1+1] +0.25*( z1[i1] + z1[i1+1] )
					u[((2*i3)*n2+(2*i2+1))*n1 + (2*i1+1)] = u[((2*i3)*n2+(2*i2+1))*n1 + (2*i1+1)] +0.25*( z1[i1] + z1[i1+1] )

				for i1 in range(mm1-1):
					#u[2*i3+1][2*i2][2*i1] = u[2*i3+1][2*i2][2*i1] +0.5 * z2[i1]
					u[((2*i3+1)*n2+(2*i2))*n1 + (2*i1)] = u[((2*i3+1)*n2+(2*i2))*n1 + (2*i1)] +0.5 * z2[i1]
					#u[2*i3+1][2*i2][2*i1+1] = u[2*i3+1][2*i2][2*i1+1] +0.25*( z2[i1] + z2[i1+1] )
					u[((2*i3+1)*n2+(2*i2))*n1 + (2*i1+1)] = u[((2*i3+1)*n2+(2*i2))*n1 + (2*i1+1)] +0.25*( z2[i1] + z2[i1+1] )

				for i1 in range(mm1-1):
					#u[2*i3+1][2*i2+1][2*i1] = u[2*i3+1][2*i2+1][2*i1] +0.25* z3[i1]
					u[((2*i3+1)*n2+(2*i2+1))*n1 + (2*i1)] = u[((2*i3+1)*n2+(2*i2+1))*n1 + (2*i1)] +0.25* z3[i1]
					#u[2*i3+1][2*i2+1][2*i1+1] = u[2*i3+1][2*i2+1][2*i1+1] +0.125*( z3[i1] + z3[i1+1] )
					u[((2*i3+1)*n2+(2*i2+1))*n1 + (2*i1+1)] = u[((2*i3+1)*n2+(2*i2+1))*n1 + (2*i1+1)] +0.125*( z3[i1] + z3[i1+1] )
			#END for i2 in range(mm2-1):
		#END for i3 in range(mm3-1):
	else:
		if n1 == 3:
			d1 = 2
			t1 = 1
		else:
			d1 = 1
			t1 = 0

		if n2 == 3:
			d2 = 2
			t2 = 1
		else:
			d2 = 1
			t2 = 0

		if n3 == 3:
			d3 = 2
			t3 = 1
		else:
			d3 = 1
			t3 = 0

		for i3 in range(d3, mm3-1+1):
			for i2 in range(d2, mm2-1+1):
				for i1 in range(d1, mm1-1+1):
					#u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] = u[2*i3-d3-1][2*i2-d2-1][2*i1-d1-1] +z[i3-1][i2-1][i1-1]
					u[((2*i3-d3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-d1-1)] = u[((2*i3-d3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-d1-1)] +z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]

				for i1 in range(1, mm1-1+1):
					#u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] = u[2*i3-d3-1][2*i2-d2-1][2*i1-t1-1] +0.5*(z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-d3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-t1-1)] = ( u[((2*i3-d3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-t1-1)] +
										   0.5*(z[((i3-1)*mm2+(i2-1))*mm1 + i1]+z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]) )

			for i2 in range(1, mm2-1+1):
				for i1 in range(d1, mm1-1+1):
					#u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] = u[2*i3-d3-1][2*i2-t2-1][2*i1-d1-1] +0.5*(z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-d3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-d1-1)] = ( u[((2*i3-d3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-d1-1)] + 
										   0.5*(z[((i3-1)*mm2+i2)*mm1 + (i1-1)]+z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]) )

				for i1 in range(1, mm1-1+1):
					#u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] = u[2*i3-d3-1][2*i2-t2-1][2*i1-t1-1] +
							#0.25*(z[i3-1][i2][i1]+z[i3-1][i2-1][i1] +z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-d3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-t1-1)] = ( u[((2*i3-d3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-t1-1)] +
							0.25*(z[((i3-1)*mm2+i2)*mm1 + i1]+z[((i3-1)*mm2+(i2-1))*mm1 + i1] +z[((i3-1)*mm2+i2)*mm1 + (i1-1)]+z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]) )
		#END for i3 in range(d3, mm3-1+1):
		
		for i3 in range(1, mm3-1+1):
			for i2 in range(d2, mm2-1+1):
				for i1 in range(d1, mm1-1+1):
					#u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] = u[2*i3-t3-1][2*i2-d2-1][2*i1-d1-1] + 0.5*(z[i3][i2-1][i1-1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-t3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-d1-1)] = ( u[((2*i3-t3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-d1-1)] + 
							0.5*(z[(i3*mm2+(i2-1))*mm1 + (i1-1)]+z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]) )

				for i1 in range(1, mm1-1+1):
					#u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] = u[2*i3-t3-1][2*i2-d2-1][2*i1-t1-1] +
					#		0.25*(z[i3][i2-1][i1]+z[i3][i2-1][i1-1] +z[i3-1][i2-1][i1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-t3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-t1-1)] = ( u[((2*i3-t3-1)*n2+(2*i2-d2-1))*n1 + (2*i1-t1-1)] +
							0.25*(z[(i3*mm2+(i2-1))*mm1 + i1]+z[(i3*mm2+(i2-1))*mm1 + (i1-1)] +z[((i3-1)*mm2+(i2-1))*mm1 + i1]+z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]) )

			for i2 in range(1, mm2-1+1):
				for i1 in range(d1, mm1-1+1):
					#u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] = u[2*i3-t3-1][2*i2-t2-1][2*i1-d1-1] +
					#		0.25*(z[i3][i2][i1-1]+z[i3][i2-1][i1-1] +z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-t3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-d1-1)] = ( u[((2*i3-t3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-d1-1)] +
							0.25*(z[(i3*mm2+i2)*mm1 + (i1-1)]+z[(i3*mm2+(i2-1))*mm1 + (i1-1)] +z[((i3-1)*mm2+i2)*mm1 + (i1-1)]+z[((i3-1)*mm2+(i2-1))*mm1 + (i1-1)]) )

				for i1 in range(1, mm1-1+1):
					#u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] = u[2*i3-t3-1][2*i2-t2-1][2*i1-t1-1] +
					#		0.125*(z[i3][i2][i1]+z[i3][i2-1][i1]
					#			+z[i3][i2][i1-1]+z[i3][i2-1][i1-1]
					#			+z[i3-1][i2][i1]+z[i3-1][i2-1][i1]
					#			+z[i3-1][i2][i1-1]+z[i3-1][i2-1][i1-1])
					u[((2*i3-t3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-t1-1)] = ( u[((2*i3-t3-1)*n2+(2*i2-t2-1))*n1 + (2*i1-t1-1)] +
							0.125*(z[(i3*n2+i2)*n1 + i1]+z[(i3*n2+(i2-1))*n1 + i1]
								+z[(i3*n2+i2)*n1 + (i1-1)]+z[(i3*n2+(i2-1))*n1 + (i1-1)]
								+z[((i3-1)*n2+i2)*n1 + i1]+z[((i3-1)*n2+(i2-1))*n1 + i1]
								+z[((i3-1)*n2+i2)*n1 + (i1-1)]+z[((i3-1)*n2+(i2-1))*n1 + (i1-1)]) )
		#END for i3 in range(1, mm3-1+1):
	#END if n1 != 3 and n2 != 3 and n3 != 3:
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_INTERP)

	if debug_vec[0] >= 1:
		rep_nrm(z, mm1, mm2, mm3, "z: inter", k-1)
		rep_nrm(u, n1, n2, n3, "u: inter", k)

	if debug_vec[5] >= k:
		showall(z, mm1, mm2, mm3)
		showall(u, n1, n2, n3)
#END interp()


# --------------------------------------------------------------------
# psinv applies an approximate inverse as smoother: u = u + Cr
# 
# this  implementation costs  15A + 4M per result, where
# A and M denote the costs of Addition and Multiplication.  
# presuming coefficient c(3) is zero (the NPB assumes this,
# but it is thus not a general case), 2A + 1M may be eliminated,
# resulting in 13A + 3M.
# note that this vectorizes, and is also fine for cache 
# based machines.  
# --------------------------------------------------------------------
#static void psinv(void* pointer_r, void* pointer_u, int n1, int n2, int n3, double c[4], int k)
@njit
def psinv(pointer_r, pointer_u, n1, n2, n3, c, k):
	#double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
	#double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
	r = pointer_r
	u = pointer_u
	
	r1 = numpy.empty(shape=M, dtype=numpy.float64)
	r2 = numpy.empty(shape=M, dtype=numpy.float64)

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_PSINV)
	for i3 in range(1, n3-1):
		for i2 in range(1, n2-1):
			for i1 in range(n1):
				#r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1] + r[i3-1][i2][i1] + r[i3+1][i2][i1]
				r1[i1] = r[(i3*n2+(i2-1))*n1 + i1] + r[(i3*n2+(i2+1))*n1 + i1] + r[((i3-1)*n2+i2)*n1 + i1] + r[((i3+1)*n2+i2)*n1 + i1]
				#r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1] + r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1]
				r2[i1] = r[((i3-1)*n2+(i2-1))*n1 + i1] + r[((i3-1)*n2+(i2+1))*n1 + i1] + r[((i3+1)*n2+(i2-1))*n1 + i1] + r[((i3+1)*n2+(i2+1))*n1 + i1]

			for i1 in range(1, n1-1):
				u[(i3*n2+i2)*n1 + i1] = ( u[(i3*n2+i2)*n1 + i1]
					+ c[0] * r[(i3*n2+i2)*n1 + i1]
					+ c[1] * ( r[(i3*n2+i2)*n1 + (i1-1)] + r[(i3*n2+i2)*n1 + (i1+1)] #( r[i3][i2][i1-1] + r[i3][i2][i1+1]
							+ r1[i1] )
					+ c[2] * ( r2[i1] + r1[i1-1] + r1[i1+1] ) )
				# --------------------------------------------------------------------
				# assume c(3) = 0    (enable line below if c(3) not= 0)
				# --------------------------------------------------------------------
				# > + c(3) * ( r2(i1-1) + r2(i1+1) )
				# --------------------------------------------------------------------
		#END
	#END
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_PSINV)

	# --------------------------------------------------------------------
	# exchange boundary points
	# --------------------------------------------------------------------
	comm3(u, n1, n2, n3, k)

	if debug_vec[0] >= 1:
		rep_nrm(u, n1, n2, n3, "   psinv", k)

	if debug_vec[3] >= k:
		showall(u, n1, n2, n3)
#END psinv()


# --------------------------------------------------------------------
# rprj3 projects onto the next coarser grid, 
# using a trilinear finite element projection: s = r' = P r
#     
# this  implementation costs 20A + 4M per result, where
# A and M denote the costs of addition and multiplication.  
# note that this vectorizes, and is also fine for cache 
# based machines.  
# --------------------------------------------------------------------
#static void rprj3(void* pointer_r, int m1k, int m2k, int m3k, void* pointer_s, int m1j, int m2j, int m3j, int k)
@njit
def rprj3(pointer_r, m1k, m2k, m3k, pointer_s, m1j, m2j, m3j, k):
	#double (*r)[m2k][m1k] = (double (*)[m2k][m1k])pointer_r; 
	#double (*s)[m2j][m1j] = (double (*)[m2j][m1j])pointer_s;
	r = pointer_r #(i3*m2k+i2)*m1k + i1
	s = pointer_s #(i3*m2j+i2)*m1j + i1

	x1 = numpy.empty(shape=M, dtype=numpy.float64)
	y1 = numpy.empty(shape=M, dtype=numpy.float64)

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RPRJ3)
	if m1k == 3:
		d1 = 2
	else:
		d1 = 1

	if m2k == 3:
		d2 = 2
	else:
		d2 = 1

	if m3k == 3:
		d3 = 2
	else:
		d3 = 1

	for j3 in range(1, m3j-1):
		i3 = 2*j3-d3
		for j2 in range(1, m2j-1):
			i2 = 2*j2-d2
			for j1 in range(1, m1j):
				i1 = 2*j1-d1
				#x1[i1] = r[i3+1][i2][i1] + r[i3+1][i2+2][i1] + r[i3][i2+1][i1] + r[i3+2][i2+1][i1]
				x1[i1] = r[((i3+1)*m2k+i2)*m1k + i1] + r[((i3+1)*m2k+(i2+2))*m1k + i1] + r[(i3*m2k+(i2+1))*m1k + i1] + r[((i3+2)*m2k+(i2+1))*m1k + i1]
				#y1[i1] = r[i3][i2][i1] + r[i3+2][i2][i1] + r[i3][i2+2][i1] + r[i3+2][i2+2][i1]
				y1[i1] = r[(i3*m2k+i2)*m1k + i1] + r[((i3+2)*m2k+i2)*m1k + i1] + r[(i3*m2k+(i2+2))*m1k + i1] + r[((i3+2)*m2k+(i2+2))*m1k + i1]

			for j1 in range(1, m1j-1):
				i1 = 2*j1-d1
				#y2 = r[i3][i2][i1+1] + r[i3+2][i2][i1+1] + r[i3][i2+2][i1+1] + r[i3+2][i2+2][i1+1]
				y2 = r[(i3*m2k+i2)*m1k + (i1+1)] + r[((i3+2)*m2k+i2)*m1k + (i1+1)] + r[(i3*m2k+(i2+2))*m1k + (i1+1)] + r[((i3+2)*m2k+(i2+2))*m1k + (i1+1)]
				#x2 = r[i3+1][i2][i1+1] + r[i3+1][i2+2][i1+1] + r[i3][i2+1][i1+1] + r[i3+2][i2+1][i1+1]
				x2 = r[((i3+1)*m2k+i2)*m1k + (i1+1)] + r[((i3+1)*m2k+(i2+2))*m1k + (i1+1)] + r[(i3*m2k+(i2+1))*m1k + (i1+1)] + r[((i3+2)*m2k+(i2+1))*m1k + (i1+1)]
				s[(j3*m2j+j2)*m1j + j1] = (
					0.5 * r[((i3+1)*m2k+(i2+1))*m1k + (i1+1)] #r[i3+1][i2+1][i1+1]
					+ 0.25 * ( r[((i3+1)*m2k+(i2+1))*m1k + i1] + r[((i3+1)*m2k+(i2+1))*m1k + (i1+2)] + x2) #( r[i3+1][i2+1][i1] + r[i3+1][i2+1][i1+2] + x2) 
					+ 0.125 * ( x1[i1] + x1[i1+2] + y2)
					+ 0.0625 * ( y1[i1] + y1[i1+2] ) )
		#END for j2 in range(1, m2j-1) 
	#END for j3 in range(1, m3j-1)
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RPRJ3)

	j = k - 1
	comm3(s, m1j, m2j, m3j, j)

	if debug_vec[0] >= 1:
		rep_nrm(s, m1j, m2j, m3j, "   rprj3", k-1)

	if debug_vec[4] >= k:
		showall(s, m1j, m2j, m3j)
#END rprj3()


# --------------------------------------------------------------------
# multigrid v-cycle routine
# --------------------------------------------------------------------
#static void mg3P(double u[], double v[], double r[], double a[4], double c[4], int n1, int n2, int n3, int k)
@njit
def mg3P(u, v, r, a, c, n1, n2, n3, k):
	# --------------------------------------------------------------------
	# down cycle.
	# restrict the residual from the find grid to the coarse
	# -------------------------------------------------------------------
	for k in range(lt, lb+1-1, -1):
		j = k - 1
		r_sub_k = r[ir[k]:]
		r_sub_j = r[ir[j]:]
		rprj3(r_sub_k, m1[k], m2[k], m3[k], r_sub_j, m1[j], m2[j], m3[j], k)

	k = lb
	# --------------------------------------------------------------------
	# compute an approximate solution on the coarsest grid
	# --------------------------------------------------------------------
	u_sub = u[ir[k]:]
	zero3(u_sub, m1[k], m2[k], m3[k])
	r_sub_p = r[ir[k]:]
	u_sub_p = u[ir[k]:]
	psinv(r_sub_p, u_sub_p, m1[k], m2[k], m3[k], c, k)

	for k in range(lb+1, lt-1+1):
		j = k-1
		# --------------------------------------------------------------------
		# prolongate from level k-1  to k
		# -------------------------------------------------------------------
		u_sub1 = u[ir[k]:]
		zero3(u_sub1, m1[k], m2[k], m3[k])
		u_sub2_j = u[ir[j]:]
		u_sub2_k = u[ir[k]:]
		interp(u_sub2_j, m1[j], m2[j], m3[j], u_sub2_k, m1[k], m2[k], m3[k], k)
		# --------------------------------------------------------------------
		# compute residual for level k
		# --------------------------------------------------------------------
		u_sub3 = u[ir[k]:]
		r_sub3 = r[ir[k]:]
		resid(u_sub3, r_sub3, r_sub3, m1[k], m2[k], m3[k], a, k)
		# --------------------------------------------------------------------
		# apply smoother
		# --------------------------------------------------------------------
		r_sub4 = r[ir[k]:]
		u_sub4 = u[ir[k]:]
		psinv(r_sub4, u_sub4, m1[k], m2[k], m3[k], c, k)

	j = lt - 1
	k = lt
	u_sub_5 = u[ir[j]:]
	interp(u_sub_5, m1[j], m2[j], m3[j], u, n1, n2, n3, k)
	resid(u, v, r, n1, n2, n3, a, k)
	psinv(r, u, n1, n2, n3, c, k)
#END mg3P()


#static void showall(void* pointer_z, int n1, int n2, int n3){
@njit
def showall(pointer_z, n1, n2, n3):
	#double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
	z = pointer_z

	m1 = min(n1, 18)
	m2 = min(n2, 14)
	m3 = min(n3, 18)

	print()
	for i3 in range(m3):
		for i2 in range(m2):
			for i1 in range(m1):
				print(z[(i3*n2+i2)*n1 + i1]) #Added break line to the original code
			print()
		print(" - - - - - - - ")
	print()
#END showall


# ---------------------------------------------------------------------
# report on norm
# ---------------------------------------------------------------------
#static void rep_nrm(void* pointer_u, int n1, int n2, int n3, char* title, int kk)
@njit
def rep_nrm(pointer_u, n1, n2, n3, title, kk):
	rnm2, rnmu = norm2u3(pointer_u, n1, n2, n3, nx[kk], ny[kk], nz[kk])
	print(" Level ", kk, " in ", title, ": norms =", rnm2, rnmu)
#END rep_nrm()


# --------------------------------------------------------------------
# resid computes the residual: r = v - Au
#
# this  implementation costs  15A + 4M per result, where
# A and M denote the costs of addition (or subtraction) and 
# multiplication, respectively. 
# presuming coefficient a(1) is zero (the NPB assumes this,
# but it is thus not a general case), 3A + 1M may be eliminated,
# resulting in 12A + 3M.
# note that this vectorizes, and is also fine for cache 
# based machines.  
# --------------------------------------------------------------------
#static void resid(void* pointer_u, void* pointer_v, void* pointer_r, int n1, int n2, int n3, double a[4], int k){
@njit
def resid(pointer_u, pointer_v, pointer_r, n1, n2, n3, a, k):
	#double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
	#double (*v)[n2][n1] = (double (*)[n2][n1])pointer_v;
	#double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
	u = pointer_u
	v = pointer_v
	r = pointer_r

	u1 = numpy.empty(shape=M, dtype=numpy.float64)
	u2 = numpy.empty(shape=M, dtype=numpy.float64)

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RESID)
	for i3 in range (1, n3-1):
		for i2 in range(1, n2-1):
			for i1 in range (n1):
				#u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1] + u[i3-1][i2][i1] + u[i3+1][i2][i1]
				u1[i1] = u[(i3*n2+(i2-1))*n1 + i1] + u[(i3*n2+(i2+1))*n1 + i1] + u[((i3-1)*n2+i2)*n1 + i1] + u[((i3+1)*n2+i2)*n1 + i1]
				#u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1] + u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1]
				u2[i1] = u[((i3-1)*n2+(i2-1))*n1 + i1] + u[((i3-1)*n2+(i2+1))*n1 + i1] + u[((i3+1)*n2+(i2-1))*n1 + i1] + u[((i3+1)*n2+(i2+1))*n1 + i1]
				
			for i1 in range(1, n1-1):
				r[(i3*n2+i2)*n1 + i1] = ( v[(i3*n2+i2)*n1 + i1] 
					- a[0] * u[(i3*n2+i2)*n1 + i1]
					 # ---------------------------------------------------------------------
					 # assume a(1) = 0 (enable 2 lines below if a(1) not= 0)
					 # ---------------------------------------------------------------------
					 # > - a(1) * ( u(i1-1,i2,i3) + u(i1+1,i2,i3)
					 # > + u1(i1) )
					 # ---------------------------------------------------------------------
					- a[2] * ( u2[i1] + u1[i1-1] + u1[i1+1] )
					- a[3] * ( u2[i1-1] + u2[i1+1] ) )
		#END for i2 in range(1, n2-1):
	#END for i3 in range (1, n3-1):
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RESID)

	# --------------------------------------------------------------------
	# exchange boundary data
	# --------------------------------------------------------------------
	comm3(r, n1, n2, n3, k)

	if debug_vec[0] >= 1:
		rep_nrm(r, n1, n2, n3, "   resid", k)

	if debug_vec[2] >= k:
		showall(r,n1,n2,n3)
#END resid()


# ---------------------------------------------------------------------
# norm2u3 evaluates approximations to the l2 norm and the
# uniform (or l-infinity or chebyshev) norm, under the
# assumption that the boundaries are periodic or zero. add the
# boundaries in with half weight (quarter weight on the edges
# and eighth weight at the corners) for inhomogeneous boundaries.
# ---------------------------------------------------------------------
#static void norm2u3(void* pointer_r, int n1, int n2, int n3, double* rnm2, double* rnmu, int nx, int ny, int nz)
@njit
def norm2u3(pointer_r, n1, n2, n3, nx, ny, nz):
	#double (*r)[n2][n1] = (double (*)[n2][n1])pointer_r;
	r = pointer_r

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_NORM2)
	dn = 1.0 * nx * ny * nz

	s = 0.0
	rnmu = 0.0
	for i3 in range(1, n3-1):
		for i2 in range(1, n2-1):
			for i1 in range(1, n1-1):
				s = s + r[(i3*n2+i2)*n1 + i1] * r[(i3*n2+i2)*n1 + i1] #s = s + r[i3][i2][i1] * r[i3][i2][i1]
				a = abs(r[(i3*n2+i2)*n1 + i1]) #a = abs(r[i3][i2][i1])
				if a > rnmu:
					rnmu = a

	rnm2 = math.sqrt(s / dn)
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_NORM2)
	
	return rnm2, rnmu
#END norm2u3()


# ---------------------------------------------------------------------
# comm3 organizes the communication on all borders 
# ---------------------------------------------------------------------
#static void comm3(void* pointer_u, int n1, int n2, int n3, int kk)
@njit
def comm3(pointer_u, n1, n2, n3, kk):
	#double (*u)[n2][n1] = (double (*)[n2][n1])pointer_u;
	u = pointer_u

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_COMM3)
		
	# axis = 1
	for i3 in range(1, n3-1):
		for i2 in range(1, n2-1):
			u[(i3*n2+i2)*n1 + 0] = u[(i3*n2+i2)*n1 + (n1-2)] #u[i3][i2][0] = u[i3][i2][n1-2]
			u[(i3*n2+i2)*n1 + (n1-1)] = u[(i3*n2+i2)*n1 + 1] #u[i3][i2][n1-1] = u[i3][i2][1]

	# axis = 2
	for i3 in range(1, n3-1):
		for i1 in range(n1):
			u[(i3*n2+0)*n1 + i1] = u[(i3*n2+(n2-2))*n1 + i1] #u[i3][0][i1] = u[i3][n2-2][i1]
			u[(i3*n2+(n2-1))*n1 + i1] = u[(i3*n2+1)*n1 + i1] #u[i3][n2-1][i1] = u[i3][1][i1]

	# axis = 3
	for i2 in range(n2):
		for i1 in range(n1):
			(0*n2+i2)*n1 + i1
			u[(0*n2+i2)*n1 + i1] = u[((n3-2)*n2+i2)*n1 + i1] #u[0][i2][i1] = u[n3-2][i2][i1]
			u[((n3-1)*n2+i2)*n1 + i1] = u[(1*n2+i2)*n1 + i1] #u[n3-1][i2][i1] = u[1][i2][i1]

	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_COMM3)
#END comm3()


# ---------------------------------------------------------------------
# bubble does a bubble sort in direction dir
# ---------------------------------------------------------------------
#static void bubble(double ten[][MM], int j1[][MM], int j2[][MM], int j3[][MM], int m, int ind)
@njit
def bubble(ten, j1, j2, j3, m, ind):
	if ind == 1:
		for i in range(m-1):
			if ten[ind][i] > ten[ind][i+1]:
				temp = ten[ind][i+1]
				ten[ind][i+1] = ten[ind][i]
				ten[ind][i] = temp

				j_temp = j1[ind][i+1]
				j1[ind][i+1] = j1[ind][i]
				j1[ind][i] = j_temp

				j_temp = j2[ind][i+1]
				j2[ind][i+1] = j2[ind][i]
				j2[ind][i] = j_temp

				j_temp = j3[ind][i+1]
				j3[ind][i+1] = j3[ind][i]
				j3[ind][i] = j_temp
			else: 
				return
		#END for i in range(m-1):
	else:
		for i in range(m-1):
			if ten[ind][i] < ten[ind][i+1]:
				temp = ten[ind][i+1]
				ten[ind][i+1] = ten[ind][i]
				ten[ind][i] = temp

				j_temp = j1[ind][i+1]
				j1[ind][i+1] = j1[ind][i]
				j1[ind][i] = j_temp

				j_temp = j2[ind][i+1]
				j2[ind][i+1] = j2[ind][i]
				j2[ind][i] = j_temp

				j_temp = j3[ind][i+1]
				j3[ind][i+1] = j3[ind][i]
				j3[ind][i] = j_temp
			else:
				return
		#END for i in range(m-1):
	#END if ind == 1:
#END bubble()


# ---------------------------------------------------------------------
# power raises an integer, disguised as a double
# precision real, to an integer power
# ---------------------------------------------------------------------
#static double power(double a, int n)
@njit
def power(a, n):
	power = 1.0
	nj = n
	aj = a

	while nj != 0:
		if (nj % 2) == 1:
			rdummy, power = randlc(power, aj)
		rdummy, aj = randlc(aj, aj)
		nj = int(nj / 2)

	return power
#END power()


#---------------------------------------------------------------------
# zran3 loads +1 at ten randomly chosen points,
# loads -1 at a different ten random points,
# and zero elsewhere.
# ---------------------------------------------------------------------
#static void zran3(void* pointer_z, int n1, int n2, int n3, int nx, int ny, int k)
@njit
def zran3(pointer_z, n1, n2, n3, nx, ny, k):
	#double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z;
	z = pointer_z

	ten = numpy.empty(shape=(2, MM), dtype=numpy.float64)
	j1 = numpy.empty(shape=(2, MM), dtype=numpy.int32)
	j2 = numpy.empty(shape=(2, MM), dtype=numpy.int32)
	j3 = numpy.empty(shape=(2, MM), dtype=numpy.int32)
	jg = numpy.empty(shape=(2, MM, 4), dtype=numpy.int32)
	
	a1 = power(A, nx)
	a2 = power(A, nx*ny)

	zero3(z, n1, n2, n3)

	i = int(is1 - 2 + nx*(is2 - 2 + ny*(is3 - 2)))

	ai = power(A, i)
	d1 = ie1 - is1 + 1
	e1 = ie1 - is1 + 2
	e2 = ie2 - is2 + 2
	e3 = ie3 - is3 + 2
	x0 = X
	aux, x0 = randlc(x0, ai)
	for i3 in range(1, e3):
		x1 = x0
		for i2 in range (1, e2):
			xx = x1
			
			idx = (i3*n2+i2)*n1 + 1
			z_sub = z[idx:]
			xx = vranlc(d1, xx, A, z_sub) #vranlc(d1, &xx, A, &(z[i3][i2][1]));
			
			aux, x1 = randlc(x1, a1)
		aux, x0 = randlc(x0, a2)

	# ---------------------------------------------------------------------
	# each processor looks for twenty candidates
	# ---------------------------------------------------------------------
	for i in range(MM):
		ten[1][i] = 0.0
		j1[1][i] = 0
		j2[1][i] = 0
		j3[1][i] = 0
		ten[0][i] = 1.0
		j1[0][i] = 0
		j2[0][i] = 0
		j3[0][i] = 0
	
	for i3 in range(1, n3-1):
		for i2 in range(1, n2-1):
			for i1 in range(1, n1-1):
				if z[(i3*n2+i2) * n1+i1] > ten[1][0]: #if(z[i3][i2][i1] > ten[1][0]){
					ten[1][0] = z[(i3*n2+i2)*n1 + i1]
					j1[1][0] = i1
					j2[1][0] = i2
					j3[1][0] = i3
					bubble(ten, j1, j2, j3, MM, 1)

				if z[(i3*n2+i2) * n1+i1] < ten[0][0]: #if(z[i3][i2][i1] < ten[0][0]){
					ten[0][0] = z[(i3*n2+i2)*n1 + i1]
					j1[0][0] = i1
					j2[0][0] = i2
					j3[0][0] = i3
					bubble(ten, j1, j2, j3, MM, 0)
	#END for i3 in range(1, n3-1):

	# ---------------------------------------------------------------------
	# now which of these are globally best?
	# ---------------------------------------------------------------------
	i1 = MM - 1
	i0 = MM - 1 
	for i in range(MM-1, 0-1, -1):
		best = 0.0
		if best < ten[1][i1]:
			jg[1][i][0] = 0
			jg[1][i][1] = is1 - 2 + j1[1][i1]
			jg[1][i][2] = is2 - 2 + j2[1][i1]
			jg[1][i][3] = is3 - 2 + j3[1][i1]
			i1 = i1-1
		else:
			jg[1][i][0] = 0
			jg[1][i][1] = 0
			jg[1][i][2] = 0
			jg[1][i][3] = 0
		
		best = 1.0
		if best > ten[0][i0]:
			jg[0][i][0] = 0
			jg[0][i][1] = is1 - 2 + j1[0][i0]
			jg[0][i][2] = is2 - 2 + j2[0][i0]
			jg[0][i][3] = is3 - 2 + j3[0][i0]
			i0 = i0 - 1
		else:
			jg[0][i][0] = 0
			jg[0][i][1] = 0
			jg[0][i][2] = 0
			jg[0][i][3] = 0
	#END for i in range(MM-1, 0-1, -1):
	m1 = 0
	m0 = 0

	for i3 in range(n3):
		for i2 in range(n2):
			for i1 in range(n1):
				z[(i3*n2+i2) * n1+i1] = 0.0 #z[i3][i2][i1] = 0.0

	for i in range(MM-1, m0-1, -1):
		#z[jg[0][i][3]][jg[0][i][2]][jg[0][i][1]] = -1.0
		i3, i2, i1 = jg[0][i][3], jg[0][i][2], jg[0][i][1] 
		z[(i3*n2+i2)*n1 + i1] = -1.0

	for i in range(MM-1, m1-1, -1):
		#z[jg[1][i][3]][jg[1][i][2]][jg[1][i][1]] = +1.0
		i3, i2, i1 = jg[1][i][3], jg[1][i][2], jg[1][i][1]
		z[(i3*n2+i2)*n1 + i1] = +1.0

	comm3(z, n1, n2, n3, k)
#END zran3()


@njit
def zero3(pointer_z, n1, n2, n3):
	#double (*z)[n2][n1] = (double (*)[n2][n1])pointer_z; 
	z = pointer_z
	
	for i3 in range(n3):
		for i2 in range(n2):
			for i1 in range(n1):
				z[(i3*n2+i2) * n1+i1] = 0.0
#END zero3()


@njit
def setup(k, nx, ny, nz, m1, m2, m3, ir):
	mi = numpy.empty(shape=(MAXLEVEL+1, 3), dtype=numpy.int32)
	ng = numpy.empty(shape=(MAXLEVEL+1, 3), dtype=numpy.int32)
	
	ng[lt][0] = nx[lt]
	ng[lt][1] = ny[lt]
	ng[lt][2] = nz[lt]
	for ax in range(3):
		for k in range(lt-1, 1-1, -1):
			ng[k][ax] = ng[k+1][ax] / 2

	for k in range(lt, 1-1, -1):
		nx[k] = ng[k][0]
		ny[k] = ng[k][1]
		nz[k] = ng[k][2]

	for k in range(lt, 1-1, -1):
		for ax in range(3):
			mi[k][ax] = 2 + ng[k][ax]

		m1[k] = mi[k][0]
		m2[k] = mi[k][1]
		m3[k] = mi[k][2]

	k = lt
	is1 = 2 + ng[k][0] - ng[lt][0]
	ie1 = 1 + ng[k][0]
	n1 = 3 + ie1 - is1
	is2 = 2 + ng[k][1] - ng[lt][1]
	ie2 = 1 + ng[k][1]
	n2 = 3 + ie2 - is2
	is3 = 2 + ng[k][2] - ng[lt][2]
	ie3 = 1 + ng[k][2]
	n3 = 3 + ie3 - is3

	ir[lt] = 0
	
	for j in range(lt-1, 1-1, -1):
		ir[j] = ir[j+1] + npbparams.ONE*m1[j+1]*m2[j+1]*m3[j+1]

	if debug_vec[1] >= 1:
		print(" in setup")
		print("   k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3")
		print("  ", k, " ",  lt, "",  ng[k][0], "",  ng[k][1], "",  ng[k][2], "",  n1, "",  n2, "",  n3, " ",
			is1, " ", is2, " ", is3, "", ie1, "", ie2, "", ie3)
		
	return n1, n2, n3, is1, is2, is3, ie1, ie2, ie3
#END setup()


def main():
	global nx, ny, nz, m1, m2, m3, ir
	#global u, v, r
	global debug_vec
	global is1, is2, is3, ie1, ie2, ie3, lb, lt
	
	if gpu_config.PROFILING:
		print(" PROFILING mode on")
		
	# ----------------------------------------------------------------------
	# read in and broadcast input data
	# ----------------------------------------------------------------------
	lt = npbparams.LT_DEFAULT
	nit = npbparams.NIT_DEFAULT
	nx[lt] = npbparams.NX_DEFAULT
	ny[lt] = npbparams.NY_DEFAULT
	nz[lt] = npbparams.NZ_DEFAULT
	#for i in range (7+1): #Already innitialized
	#	debug_vec[i] = npbparams.DEBUG_DEFAULT
	
	# ---------------------------------------------------------------------
	# use these for debug info:
	# ---------------------------------------------------------------------
	# debug_vec(0) = 1 !=> report all norms
	# debug_vec(1) = 1 !=> some setup information
	# debug_vec(1) = 2 !=> more setup information
	# debug_vec(2) = k => at level k or below, show result of resid
	# debug_vec(3) = k => at level k or below, show result of psinv
	# debug_vec(4) = k => at level k or below, show result of rprj
	# debug_vec(5) = k => at level k or below, show result of interp
	# debug_vec(6) = 1 => (unused)
	# debug_vec(7) = 1 => (unused)
	# ---------------------------------------------------------------------
	#debug_vec[0] = 1
	#debug_vec[1] = 1
	
	a = numpy.empty(4, dtype=numpy.float64)
	a[0] = -8.0 / 3.0
	a[1] =  0.0
	a[2] =  1.0 / 6.0
	a[3] =  1.0 / 12.0
	
	c = numpy.empty(4, dtype=numpy.float64)
	if npbparams.CLASS in ['A', 'S', 'W']:
		# coefficients for the s(a) smoother
		c[0] = -3.0 / 8.0
		c[1] = +1.0 / 32.0
		c[2] = -1.0 / 64.0
		c[3] = 0.0
	else:
		# coefficients for the s(b) smoother
		c[0] = -3.0 / 17.0
		c[1] = +1.0 / 33.0
		c[2] = -1.0 / 61.0
		c[3] = 0.0
	
	lb = 1
	k = lt
	
	n1, n2, n3, is1, is2, is3, ie1, ie2, ie3 = setup(k, nx, ny, nz, m1, m2, m3, ir)

	zero3(u, n1, n2, n3)
	zran3(v, n1, n2, n3, nx[lt], ny[lt], k)

	rnm2, rnmu = norm2u3(v, n1, n2, n3, nx[lt], ny[lt], nz[lt])
	
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - MG Benchmark\n")
	print(" Size: %3dx%3dx%3d (class_npb %1c)" % (nx[lt], ny[lt], nz[lt], npbparams.CLASS))
	print(" Iterations: %3d" % (nit))
	
	resid(u, v, r, n1, n2, n3, a, k)
	rnm2, rnmu = norm2u3(r, n1, n2, n3, nx[lt], ny[lt], nz[lt])

	# ---------------------------------------------------------------------
	# one iteration for startup
	# ---------------------------------------------------------------------
	mg3P(u, v, r, a, c, n1, n2, n3, k)
	resid(u, v, r, n1, n2, n3, a, k)

	n1, n2, n3, is1, is2, is3, ie1, ie2, ie3 = setup(k, nx, ny, nz, m1, m2, m3, ir)

	zero3(u, n1, n2, n3)
	zran3(v, n1, n2, n3, nx[lt], ny[lt], k)
	
	setup_gpu(a, c)

	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if(gpu_config.PROFILING):
		c_timers.timer_clear(PROFILING_COMM3)
		c_timers.timer_clear(PROFILING_INTERP)
		c_timers.timer_clear(PROFILING_NORM2U3)
		c_timers.timer_clear(PROFILING_PSINV)
		c_timers.timer_clear(PROFILING_RESID)
		c_timers.timer_clear(PROFILING_RPRJ3)
		c_timers.timer_clear(PROFILING_ZERO3)
	
	c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	resid_gpu(u_device, v_device, r_device, n1, n2, n3, a_device, k)
	rnm2, rnmu = norm2u3_gpu(r_device, n1, n2, n3, nx[lt], ny[lt], nz[lt])

	for it in range(1, nit+1):
		mg3P_gpu(u_device, v_device, r_device, a_device, c_device, n1, n2, n3, k)
		resid_gpu(u_device, v_device, r_device, n1, n2, n3, a_device, k)
	
	rnm2, rnmu = norm2u3_gpu(r_device, n1, n2, n3, nx[lt], ny[lt], nz[lt])

	c_timers.timer_stop(PROFILING_TOTAL_TIME)
	t = c_timers.timer_read(PROFILING_TOTAL_TIME)
	
	verify_value = 0.0
	
	print(" Benchmark completed")
	
	epsilon = 1.0e-8
	
	if npbparams.CLASS == 'S':
		verify_value = 0.5307707005734e-04
	elif npbparams.CLASS == 'W':
		verify_value = 0.6467329375339e-05
	elif npbparams.CLASS == 'A':
		verify_value = 0.2433365309069e-05
	elif npbparams.CLASS == 'B':
		verify_value = 0.1800564401355e-05
	elif npbparams.CLASS == 'C':
		verify_value = 0.5706732285740e-06
	elif npbparams.CLASS == 'D':
		verify_value = 0.1583275060440e-09
	elif npbparams.CLASS == 'E':
		verify_value = 0.8157592357404e-10
	
	verified = False
	err = abs(rnm2 - verify_value) / verify_value
	if err <= epsilon:
		verified = True
		print(" VERIFICATION SUCCESSFUL")
		print(" L2 Norm is %20.13e" % (rnm2))
		print(" Error is   %20.13e" % (err))
	else:
		print(" VERIFICATION FAILED")
		print(" L2 Norm is             %20.13e" % (rnm2))
		print(" The correct L2 Norm is %20.13e" % (err))
	
	nn = 1.0*nx[lt]*ny[lt]*nz[lt]

	mflops = 0.0
	if t != 0.0:
		mflops = 58.0 * nit * nn * 1.0e-6 / t
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_COMM3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" comm3", threads_per_block_on_comm3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_INTERP)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" interp", threads_per_block_on_interp, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_NORM2U3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" norm2u3", threads_per_block_on_norm2u3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_PSINV)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" psinv", threads_per_block_on_psinv, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RESID)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" resid", threads_per_block_on_resid, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RPRJ3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" rprj3", threads_per_block_on_rprj3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ZERO3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" zero3", threads_per_block_on_zero3, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" comm3", threads_per_block_on_comm3)
		gpu_config_string += "%29s\t%25d\n" % (" interp", threads_per_block_on_interp)
		gpu_config_string += "%29s\t%25d\n" % (" norm2u3", threads_per_block_on_norm2u3)
		gpu_config_string += "%29s\t%25d\n" % (" psinv", threads_per_block_on_psinv)
		gpu_config_string += "%29s\t%25d\n" % (" resid", threads_per_block_on_resid)
		gpu_config_string += "%29s\t%25d\n" % (" rprj3", threads_per_block_on_rprj3)
		gpu_config_string += "%29s\t%25d\n" % (" zero3", threads_per_block_on_zero3)
	
	c_print_results.c_print_results("MG",
			npbparams.CLASS,
			nx[lt],
			ny[lt],
			nz[lt],
			nit,
			t,
			mflops,
			"          floating point",
			verified,
			device_prop.name,
			gpu_config_string)
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA MG')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_mg_info(args.CLASS)
	set_global_variables()
	
	main()
