# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	M. Yarrow
#	C. Kuszmaul
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
# Authors of the CUDA code: 
# 	Gabriell Araujo <hexenoften@gmail.com>
#
# ------------------------------------------------------------------------------
#
# The CUDA Python version is a translation of the NPB CUDA version
# CUDA Python version: https://github.com/danidomenico/NPB-PYTHON/tree/master/NPB-CUDA
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
import numba
from numba import njit
from numba import cuda
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
from c_randdp import randlc
import c_timers
import c_print_results

sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')))
import gpu_config


# ---------------------------------------------------------------------
# note: please observe that in the routine conj_grad three 
# implementations of the sparse matrix-vector multiply have
# been supplied. the default matrix-vector multiply is not
# loop unrolled. the alternate implementations are unrolled
# to a depth of 2 and unrolled to a depth of 8. please
# experiment with these to find the fastest for your particular
# architecture. if reporting timing results, any of these three may
# be used without penalty.
# ---------------------------------------------------------------------
# class specific parameters: 
# it appears here for reference only.
# these are their values, however, this info is imported in the common/npbparams.py
# ---------------------------------------------------------------------


# Global variables
NZ = 0
NAZ = 0
PROFILING_TOTAL_TIME = 0
PROFILING_KERNEL_ONE = 1
PROFILING_KERNEL_TWO = 2
PROFILING_KERNEL_THREE = 3
PROFILING_KERNEL_FOUR = 4
PROFILING_KERNEL_FIVE = 5
PROFILING_KERNEL_SIX = 6
PROFILING_KERNEL_SEVEN = 7
PROFILING_KERNEL_EIGHT = 8
PROFILING_KERNEL_NINE = 9
PROFILING_KERNEL_TEN = 10
PROFILING_KERNEL_ELEVEN = 11

colidx = None
rowstr = None
iv = None
arow = None
acol = None
aelt = None
a = None
x = None
z = None
p = None
q = None
r = None

naa = 0
nzz = 0
firstrow = 0
lastrow = 0
firstcol = 0
lastcol = 0
tran = 0.0
amult = 0.0

# GPU variables
rowstr_device = None
colidx_device = None
a_device = None
p_device = None
q_device = None
r_device = None
x_device = None
z_device = None
rho_device = None
d_device = None
#alpha_device = None
#beta_device = None
sum_device = None
#norm_temp1_device = None
#norm_temp2_device = None

#global_data = None
#global_data_two = None
global_data_device = None
global_data_two_device = None

blocks_per_grid_on_kernel_one = 0
blocks_per_grid_on_kernel_two = 0
blocks_per_grid_on_kernel_three = 0
blocks_per_grid_on_kernel_four = 0
blocks_per_grid_on_kernel_five = 0
blocks_per_grid_on_kernel_six = 0
blocks_per_grid_on_kernel_seven = 0
blocks_per_grid_on_kernel_eight = 0
blocks_per_grid_on_kernel_nine = 0
blocks_per_grid_on_kernel_ten = 0
blocks_per_grid_on_kernel_eleven = 0

threads_per_block_on_kernel_one = 0
threads_per_block_on_kernel_two = 0
threads_per_block_on_kernel_three = 0
threads_per_block_on_kernel_four = 0
threads_per_block_on_kernel_five = 0
threads_per_block_on_kernel_six = 0
threads_per_block_on_kernel_seven = 0
threads_per_block_on_kernel_eight = 0
threads_per_block_on_kernel_nine = 0
threads_per_block_on_kernel_ten = 0
threads_per_block_on_kernel_eleven = 0

stream = 0
size_shared_data_on_kernel_two = 0
size_shared_data_on_kernel_three = 0
size_shared_data_on_kernel_four = 0
size_shared_data_on_kernel_six = 0
size_shared_data_on_kernel_eight = 0
size_shared_data_on_kernel_nine = 0
size_shared_data_on_kernel_ten = 0

gpu_device_id = 0
total_devices = 0
device_prop = None

def set_global_variables():
	global NZ, NAZ
	global colidx, rowstr, iv, arow, acol, aelt
	global a, x, z, p, q, r
	
	NZ = npbparams.NA * (npbparams.NONZER+1) * (npbparams.NONZER+1)
	NAZ = npbparams.NA * (npbparams.NONZER+1)
	
	colidx = numpy.repeat(0, NZ)
	rowstr = numpy.repeat(0, npbparams.NA+1)
	iv = numpy.repeat(0, npbparams.NA)
	arow = numpy.repeat(0, npbparams.NA)
	acol = numpy.repeat(0, NAZ)
	aelt = numpy.repeat(0.0, NAZ)
	a = numpy.repeat(0.0, NZ)
	x = numpy.repeat(1.0, npbparams.NA+2)
	z = numpy.repeat(0.0, npbparams.NA+2)
	p = numpy.repeat(0.0, npbparams.NA+2)
	q = numpy.repeat(0.0, npbparams.NA+2)
	r = numpy.repeat(0.0, npbparams.NA+2)
#END set_global_variables()


def create_zeta_verify_value():
	zeta_verify_value = 0.0
	if npbparams.CLASS == 'S':
		zeta_verify_value = 8.5971775078648
	elif npbparams.CLASS == 'W':
		zeta_verify_value = 10.362595087124
	elif npbparams.CLASS == 'A':
		zeta_verify_value = 17.130235054029
	elif npbparams.CLASS == 'B':
		zeta_verify_value = 22.712745482631
	elif npbparams.CLASS == 'C':
		zeta_verify_value = 28.973605592845
	elif npbparams.CLASS == 'D':
		zeta_verify_value = 52.514532105794
	elif npbparams.CLASS == 'E':
		zeta_verify_value = 77.522164599383
	
	return zeta_verify_value
#END create_zeta_verify_value


def setup_gpu():
	global gpu_device_id, total_devices
	global device_prop
	
	global threads_per_block_on_kernel_one, threads_per_block_on_kernel_two, threads_per_block_on_kernel_three
	global threads_per_block_on_kernel_four, threads_per_block_on_kernel_five, threads_per_block_on_kernel_six
	global threads_per_block_on_kernel_seven, threads_per_block_on_kernel_eight, threads_per_block_on_kernel_nine
	global threads_per_block_on_kernel_ten, threads_per_block_on_kernel_eleven
	
	global blocks_per_grid_on_kernel_one, blocks_per_grid_on_kernel_two, blocks_per_grid_on_kernel_three
	global blocks_per_grid_on_kernel_four, blocks_per_grid_on_kernel_five, blocks_per_grid_on_kernel_six
	global blocks_per_grid_on_kernel_seven, blocks_per_grid_on_kernel_eight, blocks_per_grid_on_kernel_nine
	global blocks_per_grid_on_kernel_ten, blocks_per_grid_on_kernel_eleven
	
	global size_shared_data_on_kernel_two, size_shared_data_on_kernel_three, size_shared_data_on_kernel_four
	global size_shared_data_on_kernel_six, size_shared_data_on_kernel_eight, size_shared_data_on_kernel_nine
	global size_shared_data_on_kernel_ten
	
	global rowstr_device, colidx_device
	global a_device, p_device, q_device, r_device, x_device, z_device
	global rho_device, d_device, sum_device
	#alpha_device, beta_device, norm_temp1_device, norm_temp2_device
	global global_data_device, global_data_two_device

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
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_ONE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_one = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_one = device_prop.WARP_SIZE
	
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_TWO
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_two = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_two = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_THREE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_three = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_three = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_FOUR
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_four = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_four = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_FIVE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_five = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_five = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_SIX
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_six = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_six = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_SEVEN
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_seven = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_seven = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_EIGHT
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_eight = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_eight = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_NINE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_nine = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_nine = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_TEN
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_ten = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_ten = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.CG_THREADS_PER_BLOCK_ON_KERNEL_ELEVEN
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_kernel_eleven = aux_threads_per_block
	else: 
		threads_per_block_on_kernel_eleven = device_prop.WARP_SIZE
		
	m_ceil = math.ceil
	blocks_per_grid_on_kernel_one = m_ceil(npbparams.NA / threads_per_block_on_kernel_one)
	blocks_per_grid_on_kernel_two = m_ceil(npbparams.NA / threads_per_block_on_kernel_two)   
	blocks_per_grid_on_kernel_three = npbparams.NA
	blocks_per_grid_on_kernel_four = m_ceil(npbparams.NA / threads_per_block_on_kernel_four)
	blocks_per_grid_on_kernel_five = m_ceil(npbparams.NA / threads_per_block_on_kernel_five)
	blocks_per_grid_on_kernel_six = m_ceil(npbparams.NA / threads_per_block_on_kernel_six)
	blocks_per_grid_on_kernel_seven = m_ceil(npbparams.NA/threads_per_block_on_kernel_seven)
	blocks_per_grid_on_kernel_eight = npbparams.NA
	blocks_per_grid_on_kernel_nine = m_ceil(npbparams.NA / threads_per_block_on_kernel_nine)
	blocks_per_grid_on_kernel_ten = m_ceil(npbparams.NA / threads_per_block_on_kernel_ten)
	blocks_per_grid_on_kernel_eleven = m_ceil(npbparams.NA / threads_per_block_on_kernel_eleven)
	
	n_float64 = numpy.float64
	rho_device = cuda.device_array(1, n_float64)
	d_device = cuda.device_array(1, n_float64)
	#alpha_device = cuda.device_array(1, n_float64)
	#beta_device = cuda.device_array(1, n_float64)
	sum_device = cuda.device_array(1, n_float64)
	#norm_temp1_device = cuda.device_array(1, n_float64)
	#norm_temp2_device = cuda.device_array(1, n_float64)
	
	local_data_elements = m_ceil(npbparams.NA / device_prop.WARP_SIZE)
	global_data_device = cuda.device_array(local_data_elements, n_float64)
	global_data_two_device = cuda.device_array(local_data_elements, n_float64)
	
	colidx_device = cuda.to_device(colidx)
	rowstr_device = cuda.to_device(rowstr)
	a_device = cuda.to_device(a)
	p_device = cuda.to_device(p)
	q_device = cuda.to_device(q)
	r_device = cuda.to_device(r)
	x_device = cuda.to_device(x)
	z_device = cuda.to_device(z)
	
	size_shared_data_on_kernel_two = threads_per_block_on_kernel_two * global_data_device.dtype.itemsize
	size_shared_data_on_kernel_three = threads_per_block_on_kernel_three * a_device.dtype.itemsize
	size_shared_data_on_kernel_four = threads_per_block_on_kernel_four * global_data_device.dtype.itemsize
	size_shared_data_on_kernel_six = threads_per_block_on_kernel_six * global_data_device.dtype.itemsize
	size_shared_data_on_kernel_eight = threads_per_block_on_kernel_eight * global_data_device.dtype.itemsize
	size_shared_data_on_kernel_nine = threads_per_block_on_kernel_nine * global_data_device.dtype.itemsize
	size_shared_data_on_kernel_ten = threads_per_block_on_kernel_ten * global_data_device.dtype.itemsize
#END setup_gpu()


#*****************************************************************
#************************* GPU FUNCTIONS *************************
#*****************************************************************
@cuda.jit('void(float64, float64[:], float64[:], int32)')
def gpu_kernel_eleven_gpu(norm_temp2, x, z, NA_aux):
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if j >= NA_aux:
		return
	x[j] = norm_temp2 * z[j]
#END gpu_kernel_eleven_gpu()


def gpu_kernel_eleven(norm_temp2):   
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_ELEVEN)

	gpu_kernel_eleven_gpu[blocks_per_grid_on_kernel_eleven,
		threads_per_block_on_kernel_eleven](norm_temp2,
										x_device,
										z_device,
										npbparams.NA)
		
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_ELEVEN)
#END gpu_kernel_eleven()


@cuda.jit('void(float64[:], float64[:], float64[:], int32)')
def gpu_kernel_ten_1_gpu(norm_temp, 
						x, 
						z,
						NA_aux):
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)

	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	local_id = cuda.threadIdx.x  

	share_data[cuda.threadIdx.x] = 0.0

	if thread_id >= NA_aux:
		return

	share_data[cuda.threadIdx.x] = x[thread_id]*z[thread_id]

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break

	if local_id == 0:
		norm_temp[cuda.blockIdx.x] = share_data[0]
#END gpu_kernel_ten_1_gpu


@cuda.jit('void(float64[:], float64[:], float64[:], int32)')
def gpu_kernel_ten_2_gpu(norm_temp,
						x,
						z,
						NA_aux):
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)

	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	local_id = cuda.threadIdx.x

	share_data[cuda.threadIdx.x] = 0.0

	if thread_id >= NA_aux:
		return

	share_data[cuda.threadIdx.x] = z[thread_id]*z[thread_id]

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break

	if local_id == 0:
		norm_temp[cuda.blockIdx.x] = share_data[0]
#END gpu_kernel_ten_2_gpu


def gpu_kernel_ten():
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_TEN)

	gpu_kernel_ten_1_gpu[blocks_per_grid_on_kernel_ten,
				  threads_per_block_on_kernel_ten, 
				  stream,
				  size_shared_data_on_kernel_ten](global_data_device, x_device, z_device, npbparams.NA)
	gpu_kernel_ten_2_gpu[blocks_per_grid_on_kernel_ten,
				  threads_per_block_on_kernel_ten,
				  stream,
				  size_shared_data_on_kernel_ten](global_data_two_device, x_device, z_device, npbparams.NA)

	local_data_reduce = 0.0
	local_data_two_reduce = 0.0
	local_data = global_data_device.copy_to_host()
	local_data_two = global_data_two_device.copy_to_host() 
	for i in range (blocks_per_grid_on_kernel_ten):
		local_data_reduce += local_data[i]
		local_data_two_reduce += local_data_two[i]

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_TEN)
		
	return local_data_reduce, local_data_two_reduce
#END gpu_kernel_ten()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32)')
def gpu_kernel_nine_gpu(r, x, summ, global_data, NA_aux):
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)

	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	local_id = cuda.threadIdx.x

	share_data[local_id] = 0.0

	if thread_id >= NA_aux: 
		return

	share_data[local_id] = x[thread_id] - r[thread_id]
	share_data[local_id] = share_data[local_id] * share_data[local_id]

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if local_id == 0:
		global_data[cuda.blockIdx.x] = share_data[0]
#END gpu_kernel_nine_gpu


def gpu_kernel_nine():
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_NINE)

	gpu_kernel_nine_gpu[blocks_per_grid_on_kernel_nine,
		threads_per_block_on_kernel_nine,
		stream,
		size_shared_data_on_kernel_nine](r_device, 
										x_device, 
										sum_device,
										global_data_device,
										npbparams.NA)
	local_data_reduce = 0.0
	local_data = global_data_device.copy_to_host()
	for i in range(blocks_per_grid_on_kernel_nine):
		local_data_reduce += local_data[i]
	
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_NINE)
	
	return local_data_reduce
#END gpu_kernel_nine()


@cuda.jit('void(int32[:], int32[:], float64[:], float64[:], float64[:])')
def gpu_kernel_eight_gpu(colidx,
						rowstr,
						a, 
						r, 
						z):
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)

	j = int((cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) / cuda.blockDim.x)
	local_id = cuda.threadIdx.x

	begin = rowstr[j]
	end = rowstr[j+1]
	summ = 0.0
	for k in range(begin+local_id, end, cuda.blockDim.x):
		summ = summ + a[k]*z[colidx[k]]
	share_data[local_id] = summ

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if local_id == 0:
		r[j] = share_data[0]
#END gpu_kernel_eight_gpu()


def gpu_kernel_eight():
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_EIGHT)

	gpu_kernel_eight_gpu[blocks_per_grid_on_kernel_eight,
		threads_per_block_on_kernel_eight,
		stream,
		size_shared_data_on_kernel_eight](colidx_device, 
										rowstr_device, 
										a_device, 
										r_device, 
										z_device)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_EIGHT)
#END gpu_kernel_eight()


@cuda.jit('void(float64, float64[:], float64[:], int32)')
def gpu_kernel_seven_gpu(beta, 
						p, 
						r,
						NA_aux):
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if j >= NA_aux: 
		return
	p[j] = r[j] + beta*p[j]
#END gpu_kernel_seven_gpu


def gpu_kernel_seven(beta_host):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_SEVEN)

	gpu_kernel_seven_gpu[blocks_per_grid_on_kernel_seven,
		threads_per_block_on_kernel_seven](beta_host,
										p_device,
										r_device,
										npbparams.NA)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_SEVEN)
#END gpu_kernel_seven()


@cuda.jit('void(float64[:], float64[:], int32)')
def gpu_kernel_six_gpu(r, 
					global_data,
					NA_aux):
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)
	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	local_id = cuda.threadIdx.x
	share_data[local_id] = 0.0
	if thread_id >= NA_aux:
		return
	r_value = r[thread_id]
	share_data[local_id] = r_value * r_value
	cuda.syncthreads()
	
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if local_id==0: 
		global_data[cuda.blockIdx.x] = share_data[0]
#END gpu_kernel_six_gpu


def gpu_kernel_six():
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_SIX)

	gpu_kernel_six_gpu[blocks_per_grid_on_kernel_six,
		threads_per_block_on_kernel_six,
		stream,
		size_shared_data_on_kernel_six](r_device, 
										global_data_device,
										npbparams.NA)
	local_data_reduce = 0.0
	local_data = global_data_device.copy_to_host()
	for i in range(blocks_per_grid_on_kernel_six):
		local_data_reduce += local_data[i]
		
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_SIX)
	
	return local_data_reduce
#END gpu_kernel_six


@cuda.jit('void(float64, float64[:], float64[:], int32)')
def gpu_kernel_five_1_gpu(alpha, 
						p, 
						z,
						NA_aux):
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if j >= NA_aux:
		return
	z[j] += alpha * p[j]
#END gpu_kernel_five_1_gpu()


@cuda.jit('void(float64, float64[:], float64[:], int32)')
def gpu_kernel_five_2_gpu(alpha, 
						q, 
						r,
						NA_aux):
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if j >= NA_aux:
		return
	r[j] -= alpha * q[j]
#END gpu_kernel_five_2_gpu()


def gpu_kernel_five(alpha_host):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_FIVE)

	gpu_kernel_five_1_gpu[blocks_per_grid_on_kernel_five,
		threads_per_block_on_kernel_five](alpha_host,
										p_device,
										z_device,
										npbparams.NA)
	gpu_kernel_five_2_gpu[blocks_per_grid_on_kernel_five,
		threads_per_block_on_kernel_five](alpha_host,
										q_device,
										r_device,
										npbparams.NA)
	
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_FIVE)
#END gpu_kernel_five()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32)')
def gpu_kernel_four_gpu(d, 
						p, 
						q, 
						global_data,
						NA_aux):
	#double* share_data = (double*)extern_share_data; 
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)

	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	local_id = cuda.threadIdx.x

	share_data[local_id] = 0.0

	if	thread_id >= NA_aux: 
		return

	share_data[cuda.threadIdx.x] = p[thread_id] * q[thread_id]

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if local_id == 0:
		global_data[cuda.blockIdx.x] = share_data[0]
#END gpu_kernel_four_gpu


def gpu_kernel_four():   
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_FOUR)

	gpu_kernel_four_gpu[blocks_per_grid_on_kernel_four,
		threads_per_block_on_kernel_four,
		stream,
		size_shared_data_on_kernel_four](d_device, 
										p_device,
										q_device,
										global_data_device,
										npbparams.NA)
	local_data_reduce = 0.0
	local_data = global_data_device.copy_to_host()
	for i in range(blocks_per_grid_on_kernel_four):
		local_data_reduce += local_data[i]
	
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_FOUR)
		
	return local_data_reduce
#END gpu_kernel_four


@cuda.jit('void(int32[:], int32[:], float64[:], float64[:], float64[:])')
def gpu_kernel_three_gpu(colidx, 
					rowstr, 
					a, 
					p, 
					q): 
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)
	
	j = int((cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) / cuda.blockDim.x)
	local_id = cuda.threadIdx.x

	begin = rowstr[j]
	end = rowstr[j+1]
	summ = 0.0
	for k in range(begin+local_id, end, cuda.blockDim.x):
		summ = summ + a[k]*p[colidx[k]]
	share_data[local_id] = summ

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if local_id == 0: 
		q[j] = share_data[0]
#END gpu_kernel_three_gpu


def gpu_kernel_three():
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_THREE)

	gpu_kernel_three_gpu[blocks_per_grid_on_kernel_three,
		threads_per_block_on_kernel_three,
		stream,
		size_shared_data_on_kernel_three](colidx_device,
										rowstr_device,
										a_device,
										p_device,
										q_device)

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_THREE)
#END gpu_kernel_three


@cuda.jit('void(float64[:], float64[:], float64[:], int32)')
def gpu_kernel_two_gpu(r,
					rho, 
					global_data,
					NA_aux):
	#double* share_data = (double*)extern_share_data;
	share_data = cuda.shared.array(shape=0, dtype=numba.float64)

	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	local_id = cuda.threadIdx.x

	share_data[local_id] = 0.0

	if thread_id >= NA_aux:
		return

	r_value = r[thread_id]
	share_data[local_id] = r_value * r_value

	cuda.syncthreads()
	i = int(cuda.blockDim.x / 2) #for(int i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if local_id < i:
			share_data[local_id] += share_data[local_id+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if local_id == 0: 
		global_data[cuda.blockIdx.x] = share_data[0]
#END gpu_kernel_two_gpu


def gpu_kernel_two():
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_TWO)
	
	gpu_kernel_two_gpu[blocks_per_grid_on_kernel_two,
		threads_per_block_on_kernel_two,
		stream,
		size_shared_data_on_kernel_two](r_device, 
									rho_device, 
									global_data_device,
									npbparams.NA)
	local_data_reduce = 0.0
	local_data = global_data_device.copy_to_host()
	for i in range(blocks_per_grid_on_kernel_two):
		local_data_reduce += local_data[i]

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_TWO)
		
	return local_data_reduce
#END gpu_kernel_two


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], int32)')
def gpu_kernel_one_gpu(p, 
					q, 
					r, 
					x, 
					z,
					NA_aux):
	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if thread_id >= NA_aux:
		return
	q[thread_id] = 0.0
	z[thread_id] = 0.0
	x_value = x[thread_id]
	r[thread_id] = x_value
	p[thread_id] = x_value
#def gpu_kernel_one_gpu()


def gpu_kernel_one():  
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_KERNEL_ONE)

	gpu_kernel_one_gpu[blocks_per_grid_on_kernel_one,
		threads_per_block_on_kernel_one](
				p_device, 
				q_device, 
				r_device, 
				x_device, 
				z_device,
				npbparams.NA)
	
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_KERNEL_ONE)
#END gpu_kernel_one()

#static void conj_grad_gpu(double* rnorm)
def conj_grad_gpu():
	cgitmax = 25
	
	# initialize the CG algorithm
	gpu_kernel_one()
	
	# rho = r.r - now, obtain the norm of r: first, sum squares of r elements locally
	rho = gpu_kernel_two()
	
	# the conj grad iteration loop
	for cgit in range(1, cgitmax+1):
		# q = A.p
		gpu_kernel_three()
		
		# obtain p.q
		d = gpu_kernel_four()

		alpha = rho / d

		# save a temporary of rho
		rho0 = rho
		
		#obtain (z = z + alpha*p) and (r = r - alpha*q)
		gpu_kernel_five(alpha)

		#rho = r.r - now, obtain the norm of r: first, sum squares of r elements locally
		rho = gpu_kernel_six()

		# obtain beta
		beta = rho / rho0

		#p = r + beta*p
		gpu_kernel_seven(beta)
	#END for cgit in range(1, cgitmax+1)
	
	# compute residual norm explicitly:  ||r|| = ||x - A.z||
	gpu_kernel_eight()

	# at this point, r contains A.z
	summ = gpu_kernel_nine()
	
	rnorm = math.sqrt(summ)
	return rnorm
#END conj_grad_gpu(rnorm)



#*****************************************************************
#************************* CPU FUNCTIONS *************************
#*****************************************************************
# ---------------------------------------------------------------------
# floating point arrays here are named as in NPB1 spec discussion of 
# CG algorithm
# ---------------------------------------------------------------------
#static void conj_grad(int colidx[], int rowstr[], double x[], double z[], 
#double a[], double p[], double q[], double r[], double* rnorm)
@njit
def conj_grad(colidx,
			rowstr,
			x,
			z,
			a,
			p,
			q,
			r): 
	cgitmax = 25
	rho = 0.0

	#initialize the CG algorithm 
	for j in range(naa+1):
		q[j] = 0.0
		z[j] = 0.0
		r[j] = x[j]
		p[j] = r[j]

	# --------------------------------------------------------------------
	# rho = r.r
	# now, obtain the norm of r: First, sum squares of r elements locally...
	# --------------------------------------------------------------------
	end = lastcol - firstcol + 1
	for j in range(end):
		rho = rho + r[j]*r[j]

	#the conj grad iteration loop
	for cgit in range(1, cgitmax+1):
		# ---------------------------------------------------------------------
		# q = A.p
		# the partition submatrix-vector multiply: use workspace w
		# ---------------------------------------------------------------------
		# 
		# note: this version of the multiply is actually (slightly: maybe %5) 
		# faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
		# below. on the Cray t3d, the reverse is TRUE, i.e., the 
		# unrolled-by-two version is some 10% faster.  
		# the unrolled-by-8 version below is significantly faster
		# on the Cray t3d - overall speed of code is 1.5 times faster.
		end = lastrow - firstrow + 1
		for j in range(end):
			summ = 0.0
			for k in range (rowstr[j], rowstr[j+1]):
				summ = summ + a[k]*p[colidx[k]]
			q[j] = summ

		# --------------------------------------------------------------------
		# obtain p.q
		# --------------------------------------------------------------------
		d = 0.0
		end = lastcol - firstcol + 1
		for j in range(end):
			d = d + p[j]*q[j]

		# --------------------------------------------------------------------
		# obtain alpha = rho / (p.q)
		# -------------------------------------------------------------------
		alpha = rho / d

		# --------------------------------------------------------------------
		# save a temporary of rho
		# --------------------------------------------------------------------
		rho0 = rho

		# ---------------------------------------------------------------------
		# obtain z = z + alpha*p
		# and    r = r - alpha*q
		# ---------------------------------------------------------------------
		rho = 0.0
		end = lastcol - firstcol + 1
		for j in range(end):
			z[j] = z[j] + alpha*p[j]
			r[j] = r[j] - alpha*q[j]

		# ---------------------------------------------------------------------
		# rho = r.r
		# now, obtain the norm of r: first, sum squares of r elements locally...
		# ---------------------------------------------------------------------
		end = lastcol - firstcol + 1 
		for j in range(end):
			rho = rho + r[j]*r[j]

		# ---------------------------------------------------------------------
		# obtain beta
		# ---------------------------------------------------------------------
		beta = rho / rho0

		# ---------------------------------------------------------------------
		# p = r + beta*p
		# ---------------------------------------------------------------------
		end = lastcol - firstcol + 1 
		for j in range(end):
			p[j] = r[j] + beta*p[j]
	#END for cgit in range(1, cgitmax+1):

	# ---------------------------------------------------------------------
	# compute residual norm explicitly: ||r|| = ||x - A.z||
	# first, form A.z
	# the partition submatrix-vector multiply
	# ---------------------------------------------------------------------
	summ = 0.0
	end = lastrow - firstrow + 1
	for j in range(end):
		d = 0.0
		for k in range(rowstr[j], rowstr[j+1]):
			d = d + a[k]*z[colidx[k]]
		r[j] = d

	# ---------------------------------------------------------------------
	# at this point, r contains A.z
	# ---------------------------------------------------------------------
	end = lastcol - firstcol + 1
	for j in range(end):
		d = x[j] - r[j]
		summ = summ + d*d

	rnorm = math.sqrt(summ)
	return rnorm
#END conj_grad()


# ---------------------------------------------------------------------
# rows range from firstrow to lastrow
# the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
# ---------------------------------------------------------------------
# static void sparse(double a[], int colidx[], int rowstr[], int n, int nz, int nozer,
#int arow[], int acol[][NONZER+1], double aelt[][NONZER+1], int firstrow, int lastrow,
#int nzloc[], double rcond, double shift)
@njit
def sparse(a,
		colidx,
		rowstr,
		n,
		nz,
		nozer,
		arow,
		acol, #int acol[][NONZER+1]
		aelt, #double aelt[][NONZER+1]
		firstrow,
		lastrow,
		nzloc,
		rcond,
		shift):
	
	NONZER_aux = npbparams.NONZER + 1
	# --------------------------------------------------------------------
	# how many rows of result
	# --------------------------------------------------------------------
	nrows = lastrow - firstrow + 1

	# --------------------------------------------------------------------
	# ...count the number of triples in each row
	# --------------------------------------------------------------------
	for j in range(nrows+1):
		rowstr[j] = 0
	
	for i in range(n):
		for nza in range(arow[i]):
			j = acol[i*NONZER_aux+nza] + 1
			rowstr[j] = rowstr[j] + arow[i]
	
	rowstr[0] = 0
	for j in range(1, nrows+1):
		rowstr[j] = rowstr[j] + rowstr[j-1]
	nza = rowstr[nrows] - 1

	# ---------------------------------------------------------------------
	# ... rowstr(j) now is the location of the first nonzero
	# of row j of a
	# ---------------------------------------------------------------------
	if nza > nz:
		print("Space for matrix elements exceeded in sparse")
		print("nza, nzmax = ", nza, ", " , nz)
		#sys.exit() #TODO: Search for an exit function supported by numba jit

	# ---------------------------------------------------------------------
	# ... preload data pages
	# ---------------------------------------------------------------------
	for j in range(nrows):
		for k in range(rowstr[j], rowstr[j+1]):
			a[k] = 0.0
			colidx[k] = -1
		nzloc[j] = 0

	# ---------------------------------------------------------------------
	# ... generate actual values by summing duplicates
	# ---------------------------------------------------------------------
	size = 1.0
	ratio = pow(rcond, (1.0 / n))
	for i in range(n):
		for nza in range(arow[i]):
			j = acol[i*NONZER_aux+nza]

			scale = size * aelt[i*NONZER_aux+nza]
			for nzrow in range(arow[i]):
				jcol = acol[i*NONZER_aux+nzrow]
				va = aelt[i*NONZER_aux+nzrow] * scale

				# --------------------------------------------------------------------
				# ... add the identity * rcond to the generated matrix to bound
				# the smallest eigenvalue from below by rcond
				# --------------------------------------------------------------------
				if jcol == j and j == i:
					va = va + rcond - shift

				goto_40 = False
				for k in range(rowstr[j], rowstr[j+1]):
					if colidx[k] > jcol:
						# ----------------------------------------------------------------
						# ... insert colidx here orderly
						# ----------------------------------------------------------------
						start = rowstr[j+1]-2
						for kk in range(start, k-1, -1): #for(kk = rowstr[j+1]-2; kk >= k; kk--)
							if colidx[kk] > -1:
								a[kk+1] = a[kk]
								colidx[kk+1] = colidx[kk]

						colidx[k] = jcol
						a[k] = 0.0
						goto_40 = True
						break
					elif colidx[k] == -1:
						colidx[k] = jcol
						goto_40 = True
						break
					elif colidx[k] == jcol:
						# --------------------------------------------------------------
						# ... mark the duplicated entry
						# -------------------------------------------------------------
						nzloc[j] = nzloc[j] + 1
						goto_40 = True
						break
				#END for k in range(rowstr[j], rowstr[j+1]):
				if (not goto_40):
					print("internal error in sparse: i=", i)
					#sys.exit() #TODO: Search for an exit function supported by numba jit
				a[k] = a[k] + va
			#END for nzrow in range(arow[i]):
		#END for nza in range(arow[i]):
		size = size * ratio
	#END for i in range(n):

	# ---------------------------------------------------------------------
	# ... remove empty entries and generate final results
	# ---------------------------------------------------------------------
	for j in range(1, nrows):
		nzloc[j] = nzloc[j] + nzloc[j-1]

	for j in range(nrows):
		if j > 0:
			j1 = rowstr[j] - nzloc[j-1]
		else:
			j1 = 0
		j2 = rowstr[j+1] - nzloc[j]
		nza = rowstr[j]
		for k in range(j1, j2):
			a[k] = a[nza]
			colidx[k] = colidx[nza]
			nza = nza + 1

	for j in range(1, nrows+1):
		rowstr[j] = rowstr[j] - nzloc[j-1]

	nza = rowstr[nrows] - 1
#END sparse()


# ---------------------------------------------------------------------
# scale a double precision number x in (0,1) by a power of 2 and chop it
# ---------------------------------------------------------------------
#static int icnvrt(double x, int ipwr2)
@njit
def icnvrt(x, ipwr2):
	return int(ipwr2 * x)
#END icnvrt() 


# --------------------------------------------------------------------
# set ith element of sparse vector (v, iv) with
# nzv nonzeros to val
# --------------------------------------------------------------------
#static void vecset(int n, double v[], int iv[], int* nzv, int i, double val)
@njit
def vecset(n, v, iv, nzv, i, val):
	sett = False
	for k in range(nzv):
		if iv[k] == i:
			v[k] = val
			sett  = True

	if (not sett):
		v[nzv]  = val
		iv[nzv] = i
		nzv = nzv + 1
	
	return nzv
#END vecset()


# ---------------------------------------------------------------------
# generate a sparse n-vector (v, iv)
# having nzv nonzeros
#
# mark(i) is set to 1 if position i is nonzero.
# mark is all zero on entry and is reset to all zero before exit
# this corrects a performance bug found by John G. Lewis, caused by
# reinitialization of mark on every one of the n calls to sprnvc
# ---------------------------------------------------------------------
#static void sprnvc(int n, int nz, int nn1, double v[], int iv[])
@njit
def sprnvc(n, nz, nn1, v, iv, tran_aux):
	nzv = 0
	vecelt = 0.0
	vecloc = 0.0

	while nzv < nz:
		vecelt, tran_aux = randlc(tran_aux, amult)

		# --------------------------------------------------------------------
		# generate an integer between 1 and n in a portable manner
		# --------------------------------------------------------------------
		vecloc, tran_aux = randlc(tran_aux, amult)
		i = icnvrt(vecloc, nn1) + 1
		if i>n: 
			continue

		# --------------------------------------------------------------------
		# was this integer generated already?
		# --------------------------------------------------------------------
		was_gen = False
		for ii in range(nzv):
			if iv[ii] == i:
				was_gen = True
				break

		if was_gen: 
			continue
		v[nzv] = vecelt
		iv[nzv] = i
		nzv = nzv + 1
	#END while nzv < nz 
	return tran_aux
#END sprnvc()


# ---------------------------------------------------------------------
# generate the test problem for benchmark 6
# makea generates a sparse matrix with a
# prescribed sparsity distribution
#
# parameter    type        usage
#
# input
#
# n            i           number of cols/rows of matrix
# nz           i           nonzeros as declared array size
# rcond        r*8         condition number
# shift        r*8         main diagonal shift
#
# output
#
# a            r*8         array for nonzeros
# colidx       i           col indices
# rowstr       i           row pointers
#
# workspace
#
# iv, arow, acol i
# aelt           r*8
# ---------------------------------------------------------------------
#static void makea(int n, int nz, double a[], int colidx[], int rowstr[], 
#int firstrow, int lastrow, int firstcol, int lastcol, int arow[], int acol[][NONZER+1], double aelt[][NONZER+1], int iv[])
@njit
def makea(n,
		nz,
		a,
		colidx,
		rowstr,
		firstrow,
		lastrow,
#		firstcol,
#		lastcol,
		arow,
		acol, 
		aelt, 
		iv,
		tran_aux):
	
	NONZER_aux = npbparams.NONZER+1
	ivc = numpy.empty(NONZER_aux, dtype=numpy.int32)
	vc = numpy.empty(NONZER_aux, dtype=numpy.float64)
	
	# --------------------------------------------------------------------
	# nonzer is approximately (int(sqrt(nnza /n)));
	# --------------------------------------------------------------------
	# nn1 is the smallest power of two not less than n
	# --------------------------------------------------------------------
	nn1 = 1
	while True:
		nn1 = 2 * nn1
		if nn1 >= n:
			break
	
	# -------------------------------------------------------------------
	# generate nonzero positions and save for the use in sparse
	# -------------------------------------------------------------------
	for iouter in range(n):
		nzv = npbparams.NONZER
		tran_aux = sprnvc(n, nzv, nn1, vc, ivc, tran_aux)
		nzv = vecset(n, vc, ivc, nzv, iouter+1, 0.5)
		arow[iouter] = nzv
		for ivelt in range(nzv):
			acol[iouter*NONZER_aux+ivelt] = ivc[ivelt] - 1 #acol[iouter][ivelt] = ivc[ivelt] - 1
			aelt[iouter*NONZER_aux+ivelt] = vc[ivelt] # aelt[iouter][ivelt] = vc[ivelt]
			
	# ---------------------------------------------------------------------
	# ... make the sparse matrix from list of elements with duplicates
	# (iv is used as  workspace)
	# ---------------------------------------------------------------------
	sparse(a,
		colidx,
		rowstr,
		n,
		nz,
		npbparams.NONZER,
		arow,
		acol,
		aelt,
		firstrow,
		lastrow,
		iv,
		npbparams.RCOND,
		npbparams.SHIFT)
	
	return tran_aux
#END makea()


def main():
	global naa, nzz, firstrow, lastrow, firstcol, lastcol
	global tran, amult
	global colidx, rowstr, iv, arow, acol, aelt
	global a, x, z, p, q, r
	
	if gpu_config.PROFILING:
		print(" PROFILING mode on")
	
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		c_timers.timer_clear(PROFILING_KERNEL_ONE)
		c_timers.timer_clear(PROFILING_KERNEL_TWO)
		c_timers.timer_clear(PROFILING_KERNEL_THREE)
		c_timers.timer_clear(PROFILING_KERNEL_FOUR)
		c_timers.timer_clear(PROFILING_KERNEL_FIVE)
		c_timers.timer_clear(PROFILING_KERNEL_SIX)
		c_timers.timer_clear(PROFILING_KERNEL_SEVEN)
		c_timers.timer_clear(PROFILING_KERNEL_EIGHT)
		c_timers.timer_clear(PROFILING_KERNEL_NINE)
		c_timers.timer_clear(PROFILING_KERNEL_TEN)
		c_timers.timer_clear(PROFILING_KERNEL_ELEVEN)
		
	firstrow = 0
	lastrow  = npbparams.NA-1
	firstcol = 0
	lastcol  = npbparams.NA-1
	
	zeta_verify_value = create_zeta_verify_value()
		
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - CG Benchmark\n")
	print(" Size: %11d" % (npbparams.NA))
	print(" Iterations: %5d" % (npbparams.NITER))
	
	naa = npbparams.NA
	nzz = NZ
	
	# initialize random number generator
	tran  = 314159265.0
	amult = 1220703125.0
	zeta, tran = randlc(tran, amult)
	
	tran = makea(naa, 
				nzz, 
				a, 
				colidx, 
				rowstr, 
				firstrow, 
				lastrow, #firstcol, lastcol, 
				arow, 
				acol, #(int(*)[NONZER+1])(void*)acol, 
				aelt, #(double(*)[NONZER+1])(void*)aelt
				iv,
				tran)
	
	# ---------------------------------------------------------------------
	# note: as a result of the above call to makea:
	# values of j used in indexing rowstr go from 0 --> lastrow-firstrow
	# values of colidx which are col indexes go from firstcol --> lastcol
	# so:
	# shift the col index vals from actual (firstcol --> lastcol) 
	# to local, i.e., (0 --> lastcol-firstcol)
	# ---------------------------------------------------------------------
	end = lastrow - firstrow + 1 
	for j in range(end): #for(j = 0; j < lastrow - firstrow + 1; j++){
		for k in range(rowstr[j], rowstr[j+1]):
			colidx[k] = colidx[k] - firstcol
	
	#Block commented, arrays already innitialized
	# set starting vector to (1, 1, .... 1)
	#for i in range(npbparams.NA+1): 
	#	x[i] = 1.0
	#end = lastcol - firstcol + 1
	#for j in range(end):
	#	q[j] = 0.0
	#	z[j] = 0.0
	#	r[j] = 0.0
	#	p[j] = 0.0  
	zeta = 0.0
	
	# -------------------------------------------------------------------
	# ---->
	# do one iteration untimed to init all code and data page tables
	# ----> (then reinit, start timing, to niter its)
	# -------------------------------------------------------------------*/
	for it in range(1, 1+1):
		# the call to the conjugate gradient routine
		rnorm = conj_grad(colidx, rowstr, x, z, a, p, q, r)

		# --------------------------------------------------------------------
		# zeta = shift + 1/(x.z)
		# so, first: (x.z)
		# also, find norm of z
		# so, first: (z.z)
		# --------------------------------------------------------------------
		norm_temp1 = 0.0
		norm_temp2 = 0.0
		end = lastcol - firstcol + 1
		for j in range(0, end):
			norm_temp1 = norm_temp1 + x[j] * z[j]
			norm_temp2 = norm_temp2 + z[j] * z[j]
		norm_temp2 = 1.0 / math.sqrt(norm_temp2)

		# normalize z to obtain x
		for j in range(0, end):
			x[j] = norm_temp2 * z[j]
	#END for it in range(1, 1+1) - end of do one iteration untimed 

	# set starting vector to (1, 1, .... 1)
	for i in range(npbparams.NA+1):
		x[i] = 1.0
	zeta = 0.0

	setup_gpu()
	c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	# --------------------------------------------------------------------
	# ---->
	# main iteration for inverse power method
	# ---->
	# --------------------------------------------------------------------
	for it in range(1, npbparams.NITER+1):
		# the call to the conjugate gradient routine
		rnorm = conj_grad_gpu()

		# --------------------------------------------------------------------
		# zeta = shift + 1/(x.z)
		# so, first: (x.z)
		# also, find norm of z
		# so, first: (z.z)
		# --------------------------------------------------------------------
		norm_temp1, norm_temp2 = gpu_kernel_ten()
		norm_temp2 = 1.0 / math.sqrt(norm_temp2)
		zeta = npbparams.SHIFT + 1.0 / norm_temp1
		if it == 1:
			print("\n   iteration           ||r||                 zeta")
		print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

		# normalize z to obtain x
		gpu_kernel_eleven(norm_temp2)
	# end of main iter inv pow meth

	c_timers.timer_stop(PROFILING_TOTAL_TIME)

	# --------------------------------------------------------------------
	# end of timed section
	# --------------------------------------------------------------------

	t = c_timers.timer_read(PROFILING_TOTAL_TIME)

	print(" Benchmark completed")
	
	verified = False
	epsilon = 1.0e-10
	err = abs(zeta - zeta_verify_value) / zeta_verify_value
	if err <= epsilon:
		verified = True
		print(" VERIFICATION SUCCESSFUL")
		print(" Zeta is    %20.13e" % (zeta))
		print(" Error is   %20.13e" % (err))
	else:
		print(" VERIFICATION FAILED")
		print(" Zeta                %20.13e" % (zeta))
		print(" The correct zeta is %20.13e" % (zeta_verify_value))

	mflops = 0.0
	if t != 0.0:
		mflops = ( (2.0 * npbparams.NITER * npbparams.NA) 
			* ( 3.0 + (npbparams.NONZER * (npbparams.NONZER+1))
				+ 25.0
				* (5.0 + (npbparams.NONZER * (npbparams.NONZER+1))) + 3.0 )
			/ t / 1000000.0 )
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_KERNEL_ONE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" one", threads_per_block_on_kernel_one, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_TWO)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" two", threads_per_block_on_kernel_two, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_THREE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" three", threads_per_block_on_kernel_three, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_FOUR)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" four", threads_per_block_on_kernel_four, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_FIVE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" five", threads_per_block_on_kernel_five, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_SIX)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" six", threads_per_block_on_kernel_six, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_SEVEN)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" seven", threads_per_block_on_kernel_seven, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_EIGHT)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" eight", threads_per_block_on_kernel_eight, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_NINE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" nine", threads_per_block_on_kernel_nine, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_TEN)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" ten", threads_per_block_on_kernel_ten, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_KERNEL_ELEVEN)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" eleven", threads_per_block_on_kernel_eleven, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" one", threads_per_block_on_kernel_one)
		gpu_config_string += "%29s\t%25d\n" % (" two", threads_per_block_on_kernel_two)
		gpu_config_string += "%29s\t%25d\n" % (" three", threads_per_block_on_kernel_three)
		gpu_config_string += "%29s\t%25d\n" % (" four", threads_per_block_on_kernel_four)
		gpu_config_string += "%29s\t%25d\n" % (" five", threads_per_block_on_kernel_five)
		gpu_config_string += "%29s\t%25d\n" % (" six", threads_per_block_on_kernel_six)
		gpu_config_string += "%29s\t%25d\n" % (" seven", threads_per_block_on_kernel_seven)
		gpu_config_string += "%29s\t%25d\n" % (" eight", threads_per_block_on_kernel_eight)
		gpu_config_string += "%29s\t%25d\n" % (" nine", threads_per_block_on_kernel_nine)
		gpu_config_string += "%29s\t%25d\n" % (" ten", threads_per_block_on_kernel_ten)
		gpu_config_string += "%29s\t%25d\n" % (" eleven", threads_per_block_on_kernel_eleven)
	
	c_print_results.c_print_results("CG",
			npbparams.CLASS,
			npbparams.NA, 
			0,
			0,
			npbparams.NITER,
			t,
			mflops,
			"          floating point",
			verified,
			device_prop.name,
			gpu_config_string)
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA CG')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_cg_info(args.CLASS)
	set_global_variables()
	
	main()
