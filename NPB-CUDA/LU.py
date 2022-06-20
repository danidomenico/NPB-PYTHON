# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	S. Weeratunga
#	V. Venkatakrishnan
#	E. Barszcz
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
from math import ceil as m_ceil
import numba
from numba import njit
from numba import cuda
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
import c_timers
import c_print_results

sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')))
import gpu_config


# ---------------------------------------------------------------------
# driver for the performance evaluation of the solver for
# five coupled parabolic/elliptic partial differential equations
# ---------------------------------------------------------------------
# parameters which can be overridden in runtime config file
# isiz1,isiz2,isiz3 give the maximum size
# ipr = 1 to print out verbose information
# omega = 2.0 is correct for all classes
# tolrsd is tolerance levels for steady state residuals
# ---------------------------------------------------------------------
# field variables and residuals
# to improve cache performance, second two dimensions padded by 1 
# for even number sizes only.
# note: corresponding array (called "v") in routines blts, buts, 
# and l2norm are similarly padded
# ---------------------------------------------------------------------

# Global variables
IPR_DEFAULT = 1
OMEGA_DEFAULT = 1.2
TOLRSD1_DEF = 1.0e-08
TOLRSD2_DEF = 1.0e-08
TOLRSD3_DEF = 1.0e-08
TOLRSD4_DEF = 1.0e-08
TOLRSD5_DEF = 1.0e-08
C1 = 1.40e+00
C2 = 0.40e+00
C3 = 1.00e-01
C4 = 1.00e+00
C5 = 1.40e+00

PROFILING_TOTAL_TIME = 0

PROFILING_ERHS_1 = 1
PROFILING_ERHS_2 = 2
PROFILING_ERHS_3 = 3
PROFILING_ERHS_4 = 4
PROFILING_ERROR = 5
PROFILING_NORM = 6
PROFILING_JACLD_BLTS = 7
PROFILING_JACU_BUTS = 8
PROFILING_L2NORM = 9
PROFILING_PINTGR_1 = 10
PROFILING_PINTGR_2 = 11
PROFILING_PINTGR_3 = 12
PROFILING_PINTGR_4 = 13
PROFILING_RHS_1 = 14
PROFILING_RHS_2 = 15
PROFILING_RHS_3 = 16
PROFILING_RHS_4 = 17
PROFILING_SETBV_1 = 18
PROFILING_SETBV_2 = 19
PROFILING_SETBV_3 = 20
PROFILING_SETIV = 21
PROFILING_SSOR_1 = 22
PROFILING_SSOR_2 = 23

u_host = None
rsd_host = None
frct_host = None
flux_host = None
qs_host = None
rho_i_host = None
a_host = None
b_host = None
c_host = None
d_host = None

#grid
dxi, deta, dzeta = 0.0, 0.0, 0.0
tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
# dissipation
dx1, dx2, dx3, dx4, dx5 = 0.0, 0.0, 0.0, 0.0, 0.0
dy1, dy2, dy3, dy4, dy5 = 0.0, 0.0, 0.0, 0.0, 0.0
dz1, dz2, dz3, dz4, dz5 = 0.0, 0.0, 0.0, 0.0, 0.0
dssp = 0.0

ce = numpy.empty((13, 5), dtype=numpy.float64())

# output control parameters */
ipr, inorm = 0, 0
# newton-raphson iteration control parameters
dt_host, omega_host = 0.0, 0.0 
tolrsd = numpy.empty(5, dtype=numpy.float64) 
rsdnm = numpy.empty(5, dtype=numpy.float64) 
errnm = numpy.empty(5, dtype=numpy.float64) 
frc = 0.0
itmax = 0, 0

# timer
maxtime = 1.0

# GPU variables
u_device = None
rsd_device = None 
frct_device = None 
rho_i_device = None 
qs_device = None  
norm_buffer_device = None 
ce_device = None

const_jac_device = None

nx = 0
ny = 0
nz = 0
THREADS_PER_BLOCK_ON_ERHS_1 = 0
THREADS_PER_BLOCK_ON_ERHS_2 = 0
THREADS_PER_BLOCK_ON_ERHS_3 = 0
THREADS_PER_BLOCK_ON_ERHS_4 = 0
THREADS_PER_BLOCK_ON_ERROR = 0
THREADS_PER_BLOCK_ON_NORM = 0
THREADS_PER_BLOCK_ON_JACLD_BLTS = 0
THREADS_PER_BLOCK_ON_JACU_BUTS = 0
THREADS_PER_BLOCK_ON_L2NORM = 0
THREADS_PER_BLOCK_ON_PINTGR_1 = 0
THREADS_PER_BLOCK_ON_PINTGR_2 = 0
THREADS_PER_BLOCK_ON_PINTGR_3 = 0
THREADS_PER_BLOCK_ON_PINTGR_4 = 0
THREADS_PER_BLOCK_ON_RHS_1 = 0
THREADS_PER_BLOCK_ON_RHS_2 = 0
THREADS_PER_BLOCK_ON_RHS_3 = 0
THREADS_PER_BLOCK_ON_RHS_4 = 0
THREADS_PER_BLOCK_ON_SETBV_1 = 0
THREADS_PER_BLOCK_ON_SETBV_2 = 0
THREADS_PER_BLOCK_ON_SETBV_3 = 0
THREADS_PER_BLOCK_ON_SETIV = 0
THREADS_PER_BLOCK_ON_SSOR_1 = 0
THREADS_PER_BLOCK_ON_SSOR_2 = 0

stream = 0

gpu_device_id = 0
total_devices = 0
device_prop = None


def setup_gpu():
	global gpu_device_id, total_devices
	global device_prop
	
	global THREADS_PER_BLOCK_ON_ERHS_1, THREADS_PER_BLOCK_ON_ERHS_2, THREADS_PER_BLOCK_ON_ERHS_3, THREADS_PER_BLOCK_ON_ERHS_4
	global THREADS_PER_BLOCK_ON_ERROR, THREADS_PER_BLOCK_ON_NORM
	global THREADS_PER_BLOCK_ON_JACLD_BLTS, THREADS_PER_BLOCK_ON_JACU_BUTS, THREADS_PER_BLOCK_ON_L2NORM
	
	global THREADS_PER_BLOCK_ON_PINTGR_1, THREADS_PER_BLOCK_ON_PINTGR_2, THREADS_PER_BLOCK_ON_PINTGR_3, THREADS_PER_BLOCK_ON_PINTGR_4
	global THREADS_PER_BLOCK_ON_RHS_1, THREADS_PER_BLOCK_ON_RHS_2, THREADS_PER_BLOCK_ON_RHS_3, THREADS_PER_BLOCK_ON_RHS_4
	
	global THREADS_PER_BLOCK_ON_SETBV_1, THREADS_PER_BLOCK_ON_SETBV_2, THREADS_PER_BLOCK_ON_SETBV_3
	global THREADS_PER_BLOCK_ON_SETIV
	global THREADS_PER_BLOCK_ON_SSOR_1, THREADS_PER_BLOCK_ON_SSOR_2
	
	global u_device, rsd_device, frct_device
	global rho_i_device, qs_device
	global norm_buffer_device
	
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
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_ERHS_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERHS_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERHS_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_ERHS_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERHS_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERHS_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_ERHS_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERHS_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERHS_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_ERHS_4
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERHS_4 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERHS_4 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_ERROR
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERROR = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERROR = device_prop.WARP_SIZE
	
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_NORM
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_NORM = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_NORM = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_JACLD_BLTS
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_JACLD_BLTS = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_JACLD_BLTS = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_JACU_BUTS
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_JACU_BUTS = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_JACU_BUTS = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_L2NORM
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_L2NORM = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_L2NORM = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_PINTGR_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_PINTGR_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_PINTGR_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_PINTGR_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_PINTGR_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_PINTGR_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_PINTGR_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_PINTGR_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_PINTGR_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_PINTGR_4
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_PINTGR_4 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_PINTGR_4 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_RHS_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_RHS_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_RHS_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_RHS_4
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_4 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_4 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_SETBV_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_SETBV_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_SETBV_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_SETBV_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_SETBV_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_SETBV_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_SETBV_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_SETBV_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_SETBV_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_SETIV
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_SETIV = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_SETIV = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_SSOR_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_SSOR_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_SSOR_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.LU_THREADS_PER_BLOCK_ON_SSOR_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_SSOR_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_SSOR_2 = device_prop.WARP_SIZE

	n_float64 = numpy.float64
	u_device = cuda.device_array((5*nx*ny*nz), n_float64)
	rsd_device = cuda.device_array((5*nx*ny*nz), n_float64)
	frct_device = cuda.device_array((5*nx*ny*nz), n_float64)
	rho_i_device = cuda.device_array((nx*ny*nz), n_float64)
	qs_device = cuda.device_array((nx*ny*nz), n_float64)
	
	max_tpb = device_prop.MAX_THREADS_PER_BLOCK
	size_norm_buffer_device = max( 5*(ny-2)*(nz-2), 
								   ((nx-3)*(ny-3)+(nx-3)*(nz-3)+(ny-3)*(nz-3)) / ((max_tpb-1)*(max_tpb-1)) + 3 )
	norm_buffer_device = cuda.device_array(size_norm_buffer_device, n_float64)
#END setup_gpu()


#*****************************************************************
#************************* GPU FUNCTIONS *************************
#*****************************************************************

@cuda.jit('void(float64[:], int32)')
def pintgr_gpu_kernel_4(frc,
					num):
	#double* buffer = (double*)extern_share_data;
	buffer_v = cuda.shared.array(shape=0, dtype=numba.float64)

	i = cuda.threadIdx.x

	buffer_v[i] = 0.0
	while i < num:
		buffer_v[cuda.threadIdx.x] += frc[i]
		i += cuda.blockDim.x

	loc_max = cuda.blockDim.x
	dist = int((loc_max+1)/2)
	i = cuda.threadIdx.x
	cuda.syncthreads()
	while loc_max > 1:
		if (i<dist) and ((i+dist)<loc_max):
			buffer_v[i] += buffer_v[i+dist]
		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if i==0:
		frc[0] = .25 * buffer_v[0]
#END pintgr_gpu_kernel_4()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64)')
def pintgr_gpu_kernel_3(u,
					frc,
					nx,
					ny,
					nz,
					deta, dzeta):
	#double* phi1 = (double*)extern_share_data;
	phi1 = cuda.shared.array(shape=0, dtype=numba.float64)
	phi2 = phi1[(cuda.blockDim.x*cuda.blockDim.y):] 
	frc3 = phi2[(cuda.blockDim.x*cuda.blockDim.y):]

	j = cuda.blockIdx.x*(cuda.blockDim.x-1)+1
	k = cuda.blockIdx.y*(cuda.blockDim.x-1)+2
	kp = cuda.threadIdx.y
	jp = cuda.threadIdx.x

	# ---------------------------------------------------------------------
	# initialize
	# ---------------------------------------------------------------------
	if ((k+kp)<(nz-1)) and ((j+jp)<(ny-2)):
		phi1[kp+(jp*cuda.blockDim.x)] = ( C2*(u[(4)+5*((1)+nx*((j+jp)+ny*(k+kp)))]-0.5
						*(u[(1)+5*((1)+nx*((j+jp)+ny*(k+kp)))]*u[(1)+5*((1)+nx*((j+jp)+ny*(k+kp)))]+u[(2)+5*((1)+nx*((j+jp)+ny*(k+kp)))]*u[(2)+5*((1)+nx*((j+jp)+ny*(k+kp)))]+u[(3)+5*((1)+nx*((j+jp)+ny*(k+kp)))]*u[(3)+5*((1)+nx*((j+jp)+ny*(k+kp)))])/u[(0)+5*((1)+nx*((j+jp)+ny*(k+kp)))]) )
		phi2[kp+(jp*cuda.blockDim.x)] = ( C2*(u[(4)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]-0.5
			*(u[(1)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]*u[(1)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]+u[(2)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]*u[(2)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]+u[(3)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]*u[(3)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))])/u[(0)+5*((nx-2)+nx*((j+jp)+ny*(k+kp)))]) )

	cuda.syncthreads()
	frc3[kp*cuda.blockDim.x+jp] = 0.0
	if ((k+kp)<(nz-2)) and ((j+jp)<(ny-3)) and (kp<(cuda.blockDim.x-1)) and (jp<cuda.blockDim.x-1):
		frc3[kp*cuda.blockDim.x+jp] = phi1[kp+(jp*cuda.blockDim.x)]+phi1[(kp+1)+(jp*cuda.blockDim.x)]+phi1[kp+((jp+1)*cuda.blockDim.x)]+phi1[(kp+1)+((jp+1)*cuda.blockDim.x)]+phi2[kp+(jp*cuda.blockDim.x)]+phi2[(kp+1)+(jp*cuda.blockDim.x)]+phi2[(kp)+((jp+1)*cuda.blockDim.x)]+phi2[(kp+1)+((jp+1)*cuda.blockDim.x)]
	loc_max = cuda.blockDim.x * cuda.blockDim.y
	dist = int((loc_max+1)/2)
	j = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
	cuda.syncthreads()
	while loc_max > 1:
		if (j<dist) and ((j+dist)<loc_max):
			frc3[j] += frc3[j+dist]
		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if j==0:
		frc[cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x] = frc3[0]*deta*dzeta
#END pintgr_gpu_kernel_3()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64)')
def pintgr_gpu_kernel_2(u,
					frc,
					nx,
					ny,
					nz,
					dxi, dzeta):
	#double* phi1 = (double*)extern_share_data;
	phi1 = cuda.shared.array(shape=0, dtype=numba.float64)
	phi2 = phi1[(cuda.blockDim.x*cuda.blockDim.y):] 
	frc2 = phi2[(cuda.blockDim.x*cuda.blockDim.y):]

	i = cuda.blockIdx.x * (cuda.blockDim.x-1) + 1
	k = cuda.blockIdx.y * (cuda.blockDim.x-1) + 2
	kp = cuda.threadIdx.y
	ip = cuda.threadIdx.x

	# ---------------------------------------------------------------------
	# initialize
	# ---------------------------------------------------------------------
	if ((k+kp)<(nz-1)) and ((i+ip)<(nx-1)):
		j = 1
		phi1[kp+(ip*cuda.blockDim.x)] = ( C2*(u[(4)+5*((i+ip)+nx*((j)+ny*(k+kp)))]-0.5
					*(u[(1)+5*((i+ip)+nx*((j)+ny*(k+kp)))]*u[(1)+5*((i+ip)+nx*((j)+ny*(k+kp)))]+u[(2)+5*((i+ip)+nx*((j)+ny*(k+kp)))]*u[(2)+5*((i+ip)+nx*((j)+ny*(k+kp)))]+u[(3)+5*((i+ip)+nx*((j)+ny*(k+kp)))]*u[(3)+5*((i+ip)+nx*((j)+ny*(k+kp)))])/u[(0)+5*((i+ip)+nx*((j)+ny*(k+kp)))]) )
		j = ny-3
		phi2[kp+(ip*cuda.blockDim.x)] = ( C2*(u[(4)+5*((i+ip)+nx*((j)+ny*(k+kp)))]-0.5
					*(u[(1)+5*((i+ip)+nx*((j)+ny*(k+kp)))]*u[(1)+5*((i+ip)+nx*((j)+ny*(k+kp)))]+u[(2)+5*((i+ip)+nx*((j)+ny*(k+kp)))]*u[(2)+5*((i+ip)+nx*((j)+ny*(k+kp)))]+u[(3)+5*((i+ip)+nx*((j)+ny*(k+kp)))]*u[(3)+5*((i+ip)+nx*((j)+ny*(k+kp)))])/u[(0)+5*((i+ip)+nx*((j)+ny*(k+kp)))]) )

	cuda.syncthreads()
	frc2[kp*cuda.blockDim.x+ip] = 0.0;
	if ((k+kp)<(nz-2)) and ((i+ip)<(nx-2)) and (kp<(cuda.blockDim.x-1)) and (ip<(cuda.blockDim.x-1)):
		frc2[kp*cuda.blockDim.x+ip] += phi1[kp+(ip*cuda.blockDim.x)]+phi1[(kp+1)+(ip*cuda.blockDim.x)]+phi1[kp+((ip+1)*cuda.blockDim.x)]+phi1[(kp+1)+((ip+1)*cuda.blockDim.x)]+phi2[kp+(ip*cuda.blockDim.x)]+phi2[(kp+1)+(ip*cuda.blockDim.x)]+phi2[kp+((ip+1)*cuda.blockDim.x)]+phi2[(kp+1)+((ip+1)*cuda.blockDim.x)]
	loc_max = cuda.blockDim.x * cuda.blockDim.y
	dist = int((loc_max+1)/2)
	i = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
	cuda.syncthreads()
	while loc_max > 1:
		if (i<dist) and ((i+dist)<loc_max):
			frc2[i] += frc2[i+dist]
		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if i==0:
		frc[cuda.blockIdx.y * cuda.gridDim.x + cuda.blockIdx.x] = frc2[0]*dxi*dzeta
#END pintgr_gpu_kernel_2()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64)')
def pintgr_gpu_kernel_1(u,
		frc,
		nx,
		ny,
		nz,
		dxi, deta):
	#double* phi1 = (double*)extern_share_data;
	phi1 = cuda.shared.array(shape=0, dtype=numba.float64)
	phi2 = phi1[(cuda.blockDim.x*cuda.blockDim.y):] 
	frc1 = phi2[(cuda.blockDim.x*cuda.blockDim.y):]

	i = cuda.blockIdx.x * (cuda.blockDim.x-1) + cuda.threadIdx.x + 1
	j = cuda.blockIdx.y * (cuda.blockDim.x-1) + cuda.threadIdx.y + 1

	# ---------------------------------------------------------------------
	# initialize
	# ---------------------------------------------------------------------
	if j<ny-2 and i<nx-1:
		k = 2
		phi1[cuda.threadIdx.x+(cuda.threadIdx.y*cuda.blockDim.x)] = ( C2*(u[(4)+5*((i)+nx*((j)+ny*(k)))]-0.5
									*(u[(1)+5*((i)+nx*((j)+ny*(k)))]*u[(1)+5*((i)+nx*((j)+ny*(k)))]+u[(2)+5*((i)+nx*((j)+ny*(k)))]*u[(2)+5*((i)+nx*((j)+ny*(k)))]+u[(3)+5*((i)+nx*((j)+ny*(k)))]*u[(3)+5*((i)+nx*((j)+ny*(k)))])/u[(0)+5*((i)+nx*((j)+ny*(k)))]) )
		k = nz-2
		phi2[cuda.threadIdx.x+(cuda.threadIdx.y*cuda.blockDim.x)] = ( C2*(u[(4)+5*((i)+nx*((j)+ny*(k)))]-0.5
									*(u[(1)+5*((i)+nx*((j)+ny*(k)))]*u[(1)+5*((i)+nx*((j)+ny*(k)))]+u[(2)+5*((i)+nx*((j)+ny*(k)))]*u[(2)+5*((i)+nx*((j)+ny*(k)))]+u[(3)+5*((i)+nx*((j)+ny*(k)))]*u[(3)+5*((i)+nx*((j)+ny*(k)))])/u[(0)+5*((i)+nx*((j)+ny*(k)))]) )

	cuda.syncthreads()
	frc1[cuda.threadIdx.y*cuda.blockDim.x+cuda.threadIdx.x] = 0.0
	if (j<(ny-3)) and (i<(nx-2)) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (cuda.threadIdx.y<(cuda.blockDim.x-1)):
		frc1[cuda.threadIdx.y*cuda.blockDim.x+cuda.threadIdx.x] = phi1[cuda.threadIdx.x+(cuda.threadIdx.y*cuda.blockDim.x)]+phi1[(cuda.threadIdx.x+1)+(cuda.threadIdx.y*cuda.blockDim.x)]+phi1[cuda.threadIdx.x+((cuda.threadIdx.y+1)*cuda.blockDim.x)]+phi1[(cuda.threadIdx.x+1)+((cuda.threadIdx.y+1)*cuda.blockDim.x)]+phi2[cuda.threadIdx.x+(cuda.threadIdx.y*cuda.blockDim.x)]+phi2[(cuda.threadIdx.x+1)+(cuda.threadIdx.y*cuda.blockDim.x)]+phi2[(cuda.threadIdx.x)+((cuda.threadIdx.y+1)*cuda.blockDim.x)]+phi2[(cuda.threadIdx.x+1)+((cuda.threadIdx.y+1)*cuda.blockDim.x)]
	loc_max = cuda.blockDim.x*cuda.blockDim.y
	dist = int((loc_max+1)/2)
	i = cuda.threadIdx.y*cuda.blockDim.x+cuda.threadIdx.x
	cuda.syncthreads()
	while loc_max > 1:
		if (i<dist) and ((i+dist)<loc_max):
			frc1[i] += frc1[i+dist]
		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if i==0:
		frc[cuda.blockIdx.y*cuda.gridDim.x+cuda.blockIdx.x] = frc1[0]*dxi*deta
#END pintgr_gpu_kernel_1()


def pintgr_gpu(u_device,
			norm_buffer_device):
	m_ceil = math.ceil
	m_sqrt = math.sqrt
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_PINTGR_1)
	# #KERNEL PINTGR 1
	pintgr_1_threads_per_block = THREADS_PER_BLOCK_ON_PINTGR_1
	pintgr_1_blocks_per_grid = (nx, ny)
	grid_1_size = nx * ny

	# dimensions must fit the gpu
	while (pintgr_1_threads_per_block*pintgr_1_threads_per_block) > device_prop.MAX_THREADS_PER_BLOCK:
		pintgr_1_threads_per_block = m_ceil(m_sqrt(pintgr_1_threads_per_block))

	# shared memory must fit the gpu
	aux = u_device.dtype.itemsize * 3 * (pintgr_1_threads_per_block*pintgr_1_threads_per_block)
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		pintgr_1_threads_per_block = m_ceil(pintgr_1_threads_per_block / 2.0)

	final_pintgr_1_threads_per_block = (pintgr_1_threads_per_block, pintgr_1_threads_per_block)
	size_shared_data = u_device.dtype.itemsize * 3 * (pintgr_1_threads_per_block*pintgr_1_threads_per_block)
	#print("threadSize=[%d, %d]" % (final_pintgr_1_threads_per_block[0], final_pintgr_1_threads_per_block[0]))
	#print("blockSize=[%d, %d]" % (pintgr_1_blocks_per_grid[0], pintgr_1_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	pintgr_gpu_kernel_1[pintgr_1_blocks_per_grid, 
		final_pintgr_1_threads_per_block,
		stream,
		size_shared_data](u_device, 
						norm_buffer_device, 
						nx, 
						ny, 
						nz,
						dxi, deta)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_PINTGR_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_PINTGR_2)
	# #KERNEL PINTGR 2
	pintgr_2_threads_per_block = THREADS_PER_BLOCK_ON_PINTGR_2
	pintgr_2_blocks_per_grid = (nx, nz)
	grid_2_size = nx * nz

	# dimensions must fit the gpu
	while (pintgr_2_threads_per_block*pintgr_2_threads_per_block) > device_prop.MAX_THREADS_PER_BLOCK:
		pintgr_2_threads_per_block = m_ceil(m_sqrt(pintgr_2_threads_per_block))

	# shared memory must fit the gpu
	aux = u_device.dtype.itemsize * 3 * (pintgr_2_threads_per_block*pintgr_2_threads_per_block)
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		pintgr_2_threads_per_block = m_ceil(pintgr_2_threads_per_block / 2.0)

	final_pintgr_2_threads_per_block = (pintgr_2_threads_per_block, pintgr_2_threads_per_block)
	size_shared_data = u_device.dtype.itemsize * 3 * (pintgr_2_threads_per_block*pintgr_2_threads_per_block)
	#print("threadSize=[%d, %d]" % (final_pintgr_2_threads_per_block[0], final_pintgr_2_threads_per_block[0]))
	#print("blockSize=[%d, %d]" % (pintgr_2_blocks_per_grid[0], pintgr_2_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	pintgr_gpu_kernel_2[pintgr_2_blocks_per_grid, 
		final_pintgr_2_threads_per_block,
		stream,
		size_shared_data](u_device, 
						norm_buffer_device[grid_1_size:], 
						nx, 
						ny, 
						nz,
						dxi, dzeta)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_PINTGR_2)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_PINTGR_3)
	# #KERNEL PINTGR 3
	pintgr_3_threads_per_block = THREADS_PER_BLOCK_ON_PINTGR_3
	pintgr_3_blocks_per_grid = (ny, nz)
	grid_3_size = ny * nz

	# dimensions must fit the gpu
	while (pintgr_3_threads_per_block*pintgr_3_threads_per_block) > device_prop.MAX_THREADS_PER_BLOCK:
		pintgr_3_threads_per_block = m_ceil(m_sqrt(pintgr_3_threads_per_block));

	# shared memory must fit the gpu
	aux = u_device.dtype.itemsize * 3 * (pintgr_3_threads_per_block*pintgr_3_threads_per_block)
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		pintgr_3_threads_per_block = m_ceil(pintgr_3_threads_per_block / 2.0)

	final_pintgr_3_threads_per_block = (pintgr_3_threads_per_block, pintgr_3_threads_per_block)
	size_shared_data = u_device.dtype.itemsize * 3 * (pintgr_3_threads_per_block*pintgr_3_threads_per_block)
	#print("threadSize=[%d, %d]" % (final_pintgr_3_threads_per_block[0], final_pintgr_3_threads_per_block[0]))
	#print("blockSize=[%d, %d]" % (pintgr_3_blocks_per_grid[0], pintgr_3_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	pintgr_gpu_kernel_3[pintgr_3_blocks_per_grid, 
		final_pintgr_3_threads_per_block,
		stream,
		size_shared_data](u_device, 
						norm_buffer_device[(grid_1_size+grid_2_size):],
						nx, 
						ny, 
						nz,
						deta, dzeta)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_PINTGR_3)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_PINTGR_4)
	# #KERNEL PINTGR 4
	pintgr_4_threads_per_block = THREADS_PER_BLOCK_ON_PINTGR_4
	pintgr_4_blocks_per_grid = 1;

	# shared memory must fit the gpu
	aux = u_device.dtype.itemsize * pintgr_4_threads_per_block 
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		pintgr_4_threads_per_block = int(pintgr_4_threads_per_block / 2)
	size_shared_data = u_device.dtype.itemsize * pintgr_4_threads_per_block 
	#print("threadSize=[%d]" % (pintgr_4_threads_per_block))
	#print("blockSize=[%d]" % (pintgr_4_blocks_per_grid))
	#print("sharedMemory=%d" % (size_shared_data))

	pintgr_gpu_kernel_4[pintgr_4_blocks_per_grid, 
		pintgr_4_threads_per_block,
		stream,
		size_shared_data](norm_buffer_device,
						(grid_1_size+grid_2_size+grid_3_size))
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_PINTGR_4)

	#cudaMemcpy(&frc, norm_buffer_device, sizeof(double), cudaMemcpyDeviceToHost);
	norm_buffer = norm_buffer_device.copy_to_host() #Numba requires to copy the full array
	
	return norm_buffer[0] #frc value
#END pintgr_gpu()


# ---------------------------------------------------------------------
# compute the exact solution at (i,j,k)
# ---------------------------------------------------------------------
@cuda.jit('void(int32, int32, int32, float64[:], int32, int32, int32, float64[:, :])', device=True)
def exact_gpu_device(i,
					j,
					k,
					u000ijk,
					nx,
					ny,
					nz,
					ce):
	xi = i / (nx-1)
	eta = j / (ny-1)
	zeta = k / (nz-1)
	for m in range(5):
		u000ijk[m] = ( ce[0][m]+
			(ce[1][m]+
			 (ce[4][m]+
			  (ce[7][m]+
			   ce[10][m]*xi)*xi)*xi)*xi+ 
			(ce[2][m]+
			 (ce[5][m]+
			  (ce[8][m]+
			   ce[11][m]*eta)*eta)*eta)*eta+ 
			(ce[3][m]+
			 (ce[6][m]+
			  (ce[9][m]+
			   ce[12][m]*zeta)*zeta)*zeta)*zeta )
#END exact_gpu_device()


# ---------------------------------------------------------------------
# compute the solution error
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64[:, :])')
def error_gpu_kernel(u,
					errnm,
					nx,
					ny,
					nz,
					ce):
	u000ijk = cuda.local.array(5, numba.float64)

	errnm_loc = cuda.shared.array(shape=0, dtype=numba.float64)

	k = cuda.blockIdx.x+1
	j = cuda.blockIdx.y+1
	i = cuda.threadIdx.x+1

	for m in range(5):
		errnm_loc[m+5*cuda.threadIdx.x] = 0.0
	while i < nx-1:
		exact_gpu_device(i, j, k, u000ijk, nx, ny, nz, ce)
		for m in range(5):
			tmp = u000ijk[m] - u[(m)+5*((i)+nx*((j)+ny*(k)))]
			errnm_loc[m+5*cuda.threadIdx.x] += tmp*tmp

		i += cuda.blockDim.x

	i = cuda.threadIdx.x
	loc_max = cuda.blockDim.x
	dist = int((loc_max+1)/2)
	cuda.syncthreads()
	while loc_max > 1:
		if i<dist and i+dist<loc_max:
			for m in range(5):
				errnm_loc[m+5*i] += errnm_loc[m+5*(i+dist)]
		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if i==0:
		for m in range(5):
			errnm[m+5*(cuda.blockIdx.y+cuda.gridDim.y*cuda.blockIdx.x)] = errnm_loc[m]
#END error_gpu_kernel()


# ---------------------------------------------------------------------
# compute the solution error
# ---------------------------------------------------------------------
def error_gpu(u_device,
			norm_buffer_device,
			errnm,
			ce_device):
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERROR)
	# #KERNEL ERROR
	error_threads_per_block = THREADS_PER_BLOCK_ON_ERROR
	error_blocks_per_grid = (nz-2, ny-2)

	# shared memory must fit the gpu
	aux = u_device.dtype.itemsize * 5 * error_threads_per_block
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		error_threads_per_block = int(error_threads_per_block / 2)
	size_shared_data = u_device.dtype.itemsize * 5 * min(nx-2,error_threads_per_block)

	error_gpu_kernel[error_blocks_per_grid,
		(min(nx-2,error_threads_per_block)),
		stream,
		size_shared_data](u_device, 
						norm_buffer_device, 
						nx, 
						ny, 
						nz,
						ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERROR)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_NORM)
	# #KERNEL NORM
	norm_threads_per_block = THREADS_PER_BLOCK_ON_NORM
	norm_blocks_per_grid = 1

	# shared memory must fit the gpu
	aux = u_device.dtype.itemsize * 5 * norm_threads_per_block
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		norm_threads_per_block = int(norm_threads_per_block / 2)
	size_shared_data = u_device.dtype.itemsize * 5 * norm_threads_per_block

	norm_gpu_kernel[norm_blocks_per_grid, 
		norm_threads_per_block,
		stream,
		size_shared_data](norm_buffer_device, 
						(nz-2)*(ny-2))
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_NORM)

	#cudaMemcpy(errnm, norm_buffer_device, 5*sizeof(double), cudaMemcpyDeviceToHost);
	norm_buffer = norm_buffer_device.copy_to_host() #Numba requires to copy the full array
	
	m_sqrt = math.sqrt
	for m in range(5):
		errnm[m] = m_sqrt(norm_buffer[m] / ((nz-2)*(ny-2)*(nx-2)))
#END error_gpu()


@cuda.jit('void(float64[:], int32)')
def norm_gpu_kernel(rms,
					size):
	buffer_v = cuda.shared.array(shape=0, dtype=numba.float64)

	i = cuda.threadIdx.x

	for m in range(5):
		buffer_v[m+5*i] = 0.0
	while i < size:
		for m in range(5):
			buffer_v[m+5*cuda.threadIdx.x] += rms[m+5*i]
		i += cuda.blockDim.x

	loc_max = cuda.blockDim.x
	dist = int((loc_max+1)/2)
	i = cuda.threadIdx.x
	cuda.syncthreads()
	while loc_max > 1:
		if (i<dist) and ((i+dist)<loc_max):
			for m in range(5):
				buffer_v[m+5*i] += buffer_v[m+5*(i+dist)]
		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if cuda.threadIdx.x < 5:
		rms[cuda.threadIdx.x] = buffer_v[cuda.threadIdx.x]
#END norm_gpu_kernel()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32)')
def l2norm_gpu_kernel(v,
		sum_v,
		nx,
		ny,
		nz):
	sum_loc = cuda.shared.array(shape=0, dtype=numba.float64)

	k = cuda.blockIdx.x + 1
	j = cuda.blockIdx.y + 1
	i = cuda.threadIdx.x + 1

	for m in range(5):
		sum_loc[m+5*cuda.threadIdx.x] = 0.0
	while i < (nx-1):
		for m in range(5):
			sum_loc[m+5*cuda.threadIdx.x] += v[(m)+5*((i)+nx*((j)+ny*(k)))] * v[(m)+5*((i)+nx*((j)+ny*(k)))]
		i += cuda.blockDim.x

	i = cuda.threadIdx.x
	loc_max = cuda.blockDim.x
	dist = int((loc_max+1)/2)
	cuda.syncthreads()
	while loc_max > 1:
		if (i<dist) and (i+dist<loc_max):
			for m in range(5):
				sum_loc[m+5*i] += sum_loc[m+5*(i+dist)]

		loc_max = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	if i==0:
		for m in range(5):
			sum_v[m+5*(cuda.blockIdx.y+cuda.gridDim.y*cuda.blockIdx.x)] = sum_loc[m]
#END l2norm_gpu_kernel()


# ---------------------------------------------------------------------
# to compute the l2-norm of vector v.
# ---------------------------------------------------------------------
# to improve cache performance, second two dimensions padded by 1 
# for even number sizes only.  Only needed in v.
# ---------------------------------------------------------------------
def l2norm_gpu(v,
			sum_v,
			norm_buffer_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_L2NORM)
	# #KERNEL L2NORM
	l2norm_threads_per_block = THREADS_PER_BLOCK_ON_L2NORM
	l2norm_blocks_per_grid = (nz-2, ny-2)

	# shared memory must fit the gpu
	aux = v.dtype.itemsize * 5 * l2norm_threads_per_block
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		l2norm_threads_per_block = int(l2norm_threads_per_block / 2)
	size_shared_data = v.dtype.itemsize * 5 * min(nx-2, l2norm_threads_per_block)
	#print("threadSize=[%d]" % (l2norm_threads_per_block))
	#print("blockSize=[%d, %d]" % (l2norm_blocks_per_grid[0], l2norm_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	l2norm_gpu_kernel[l2norm_blocks_per_grid, 
		min(nx-2,l2norm_threads_per_block),
		stream,
		size_shared_data](v, 
						norm_buffer_device, 
						nx, 
						ny, 
						nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_L2NORM)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_NORM)
	# #KERNEL NORM
	norm_threads_per_block = THREADS_PER_BLOCK_ON_NORM
	norm_blocks_per_grid = 1

	# shared memory must fit the gpu
	aux = v.dtype.itemsize * 5 * norm_threads_per_block
	while aux > device_prop.MAX_SHARED_MEMORY_PER_BLOCK:
		norm_threads_per_block = int(norm_threads_per_block / 2)
	size_shared_data = v.dtype.itemsize * 5 * norm_threads_per_block
	#print("threadSize=[%d]" % (norm_threads_per_block))
	#print("blockSize=[%d]" % (norm_blocks_per_grid))
	#print("sharedMemory=%d" % (size_shared_data))

	norm_gpu_kernel[norm_blocks_per_grid, 
		norm_threads_per_block,
		stream,
		size_shared_data](norm_buffer_device, 
						(nz-2)*(ny-2))
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_NORM)

	#cudaMemcpy(sum_v, norm_buffer_device, 5*sizeof(double), cudaMemcpyDeviceToHost);
	norm_buffer_host = norm_buffer_device.copy_to_host() #Numba requires to copy the full array
	
	m_sqrt = math.sqrt
	for m in range(5):
		sum_v[m] = m_sqrt(norm_buffer_host[m] / ((nz-2)*(ny-2)*(nx-2)))
#END l2norm_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def rhs_gpu_kernel_4(u,
					rsd,
					qs,
					rho_i,
					nx,
					ny,
					nz,
					tz1, tz2, tz3,
					dz1, dz2, dz3, dz4, dz5,
					dssp):
	#double* flux = (double*)extern_share_data;
	flux = cuda.shared.array(shape=0, dtype=numba.float64)
	utmp = flux[(cuda.blockDim.x*5):] 
	rtmp = utmp[(cuda.blockDim.x*5):]
	rhotmp = rtmp[(cuda.blockDim.x*5):]
	u21k = rhotmp[(cuda.blockDim.x):]
	u31k = u21k[(cuda.blockDim.x):]
	u41k = u31k[(cuda.blockDim.x):]
	u51k = u41k[(cuda.blockDim.x):]

	j = cuda.blockIdx.x + 1
	i = cuda.blockIdx.y + 1
	k = cuda.threadIdx.x

	while k < nz:
		nthreads = (nz-(k-cuda.threadIdx.x))
		if nthreads>cuda.blockDim.x:
			nthreads = cuda.blockDim.x
		m = cuda.threadIdx.x
		utmp[m] = u[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		rhotmp[cuda.threadIdx.x] = rho_i[(i)+nx*((j)+ny*(k))]
		cuda.syncthreads()
		# ---------------------------------------------------------------------
		# zeta-direction flux differences
		# ---------------------------------------------------------------------
		flux[cuda.threadIdx.x+(0*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+3]
		u41 = utmp[cuda.threadIdx.x*5+3]*rhotmp[cuda.threadIdx.x]
		q = qs[(i)+nx*((j)+ny*(k))]
		flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+1]*u41
		flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+2]*u41
		flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+3]*u41+C2*(utmp[cuda.threadIdx.x*5+4]-q)
		flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = (C1*utmp[cuda.threadIdx.x*5+4]-C2*q)*u41
		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (k<(nz-1)):
			for m in range(5):
				rtmp[cuda.threadIdx.x*5+m] = rtmp[cuda.threadIdx.x*5+m]-tz2*(flux[(cuda.threadIdx.x+1)+(m*cuda.blockDim.x)]-flux[(cuda.threadIdx.x-1)+(m*cuda.blockDim.x)])
		u21k[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+1]
		u31k[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+2]
		u41k[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+3]
		u51k[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+4]
		cuda.syncthreads()
		if cuda.threadIdx.x >= 1:
			flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = tz3*(u21k[cuda.threadIdx.x]-u21k[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = tz3*(u31k[cuda.threadIdx.x]-u31k[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = (4.0/3.0)*tz3*(u41k[cuda.threadIdx.x]-u41k[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = 0.5*(1.0-C1*C5)*tz3*((u21k[cuda.threadIdx.x]*u21k[cuda.threadIdx.x]+u31k[cuda.threadIdx.x]*u31k[cuda.threadIdx.x]+u41k[cuda.threadIdx.x]*u41k[cuda.threadIdx.x])-(u21k[cuda.threadIdx.x-1]*u21k[cuda.threadIdx.x-1]+u31k[cuda.threadIdx.x-1]*u31k[cuda.threadIdx.x-1]+u41k[cuda.threadIdx.x-1]*u41k[cuda.threadIdx.x-1]))+(1.0/6.0)*tz3*(u41k[cuda.threadIdx.x]*u41k[cuda.threadIdx.x]-u41k[cuda.threadIdx.x-1]*u41k[cuda.threadIdx.x-1])+C1*C5*tz3*(u51k[cuda.threadIdx.x]-u51k[cuda.threadIdx.x-1])

		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (k<(nz-1)):
			rtmp[cuda.threadIdx.x*5+0] += dz1*tz1*(utmp[cuda.threadIdx.x*5-5]-2.0*utmp[cuda.threadIdx.x*5+0]+utmp[cuda.threadIdx.x*5+5])
			rtmp[cuda.threadIdx.x*5+1] += tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(1*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(1*cuda.blockDim.x)])+dz2*tz1*(utmp[5*cuda.threadIdx.x-4]-2.0*utmp[cuda.threadIdx.x*5+1]+utmp[cuda.threadIdx.x*5+6])
			rtmp[cuda.threadIdx.x*5+2] += tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(2*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(2*cuda.blockDim.x)])+dz3*tz1*(utmp[5*cuda.threadIdx.x-3]-2.0*utmp[cuda.threadIdx.x*5+2]+utmp[cuda.threadIdx.x*5+7])
			rtmp[cuda.threadIdx.x*5+3] += tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(3*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(3*cuda.blockDim.x)])+dz4*tz1*(utmp[5*cuda.threadIdx.x-2]-2.0*utmp[cuda.threadIdx.x*5+3]+utmp[cuda.threadIdx.x*5+8])
			rtmp[cuda.threadIdx.x*5+4] += tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(4*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(4*cuda.blockDim.x)])+dz5*tz1*(utmp[5*cuda.threadIdx.x-1]-2.0*utmp[cuda.threadIdx.x*5+4]+utmp[cuda.threadIdx.x*5+9])
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			if k==1:
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(5.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((i)+nx*((j)+ny*(3)))])
			if k==2:
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= ( dssp*(-4.0
								*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((i)+nx*((j)+ny*(4)))]) )
			if (k>=3) and (k<(nz-3)):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= ( dssp*(u[(m)+5*((i)+nx*((j)+ny*(k-2)))]
								-4.0*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((i)+nx*((j)+ny*(k+2)))]) )
			if k==(nz-3):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= ( dssp*(u[(m)+5*((i)+nx*((j)+ny*(nz-5)))]
								-4.0*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]) )
			if k==(nz-2):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(u[(m)+5*((i)+nx*((j)+ny*(nz-4)))]-4.0*utmp[cuda.threadIdx.x*5+m-5]+5.0*utmp[cuda.threadIdx.x*5+m])

		m = cuda.threadIdx.x
		rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))] = rtmp[m]
		k += cuda.blockDim.x-2
	#END while k < nz:
#END rhs_gpu_kernel_4()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def rhs_gpu_kernel_3(u,
					rsd,
					qs,
					rho_i,
					nx,
					ny,
					nz,
					ty1, ty2, ty3,
					dy1, dy2, dy3, dy4, dy5,
					dssp):
	#double* flux = (double*)extern_share_data;
	flux = cuda.shared.array(shape=0, dtype=numba.float64)
	utmp = flux[(cuda.blockDim.x*5):] 
	rtmp = utmp[(cuda.blockDim.x*5):]
	rhotmp = rtmp[(cuda.blockDim.x*5):]
	u21j = rhotmp[(cuda.blockDim.x):]
	u31j = u21j[(cuda.blockDim.x):]
	u41j = u31j[(cuda.blockDim.x):]
	u51j = u41j[(cuda.blockDim.x):]

	k = cuda.blockIdx.x+1
	i = cuda.blockIdx.y+1
	j = cuda.threadIdx.x

	while j < ny:
		nthreads = ny-(j-cuda.threadIdx.x)
		if nthreads > cuda.blockDim.x:
			nthreads = cuda.blockDim.x
		m = cuda.threadIdx.x;
		utmp[m] = u[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		rhotmp[cuda.threadIdx.x] = rho_i[(i)+nx*((j)+ny*(k))]
		cuda.syncthreads()
		# ---------------------------------------------------------------------
		# eta-direction flux differences
		# ---------------------------------------------------------------------
		flux[cuda.threadIdx.x+(0*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+2]
		u31 = utmp[cuda.threadIdx.x*5+2]*rhotmp[cuda.threadIdx.x]
		q = qs[(i)+nx*((j)+ny*(k))]
		flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+1]*u31
		flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+2]*u31+C2*(utmp[cuda.threadIdx.x*5+4]-q)
		flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+3]*u31
		flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = (C1*utmp[cuda.threadIdx.x*5+4]-C2*q)*u31
		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (j<(ny-1)):
			for m in range(5):
				rtmp[cuda.threadIdx.x*5+m] = rtmp[cuda.threadIdx.x*5+m]-ty2*(flux[(cuda.threadIdx.x+1)+(m*cuda.blockDim.x)]-flux[(cuda.threadIdx.x-1)+(m*cuda.blockDim.x)])
		u21j[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+1]
		u31j[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+2]
		u41j[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+3]
		u51j[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+4]
		cuda.syncthreads()
		if cuda.threadIdx.x >= 1:
			flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = ty3*(u21j[cuda.threadIdx.x]-u21j[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = (4.0/3.0)*ty3*(u31j[cuda.threadIdx.x]-u31j[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = ty3*(u41j[cuda.threadIdx.x]-u41j[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = 0.5*(1.0-C1*C5)*ty3*((u21j[cuda.threadIdx.x]*u21j[cuda.threadIdx.x]+u31j[cuda.threadIdx.x]*u31j[cuda.threadIdx.x]+u41j[cuda.threadIdx.x]*u41j[cuda.threadIdx.x])-(u21j[cuda.threadIdx.x-1]*u21j[cuda.threadIdx.x-1]+u31j[cuda.threadIdx.x-1]*u31j[cuda.threadIdx.x-1]+u41j[cuda.threadIdx.x-1]*u41j[cuda.threadIdx.x-1]))+(1.0/6.0)*ty3*(u31j[cuda.threadIdx.x]*u31j[cuda.threadIdx.x]-u31j[cuda.threadIdx.x-1]*u31j[cuda.threadIdx.x-1])+C1*C5*ty3*(u51j[cuda.threadIdx.x]-u51j[cuda.threadIdx.x-1])

		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)  and (j<(ny-1))):
			rtmp[cuda.threadIdx.x*5+0] += dy1*ty1*(utmp[5*(cuda.threadIdx.x-1)]-2.0*utmp[cuda.threadIdx.x*5+0]+utmp[5*(cuda.threadIdx.x+1)])
			rtmp[cuda.threadIdx.x*5+1] += ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(1*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(1*cuda.blockDim.x)])+dy2*ty1*(utmp[5*cuda.threadIdx.x-4]-2.0*utmp[cuda.threadIdx.x*5+1]+utmp[5*cuda.threadIdx.x+6])
			rtmp[cuda.threadIdx.x*5+2] += ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(2*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(2*cuda.blockDim.x)])+dy3*ty1*(utmp[5*cuda.threadIdx.x-3]-2.0*utmp[cuda.threadIdx.x*5+2]+utmp[5*cuda.threadIdx.x+7])
			rtmp[cuda.threadIdx.x*5+3] += ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(3*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(3*cuda.blockDim.x)])+dy4*ty1*(utmp[5*cuda.threadIdx.x-2]-2.0*utmp[cuda.threadIdx.x*5+3]+utmp[5*cuda.threadIdx.x+8])
			rtmp[cuda.threadIdx.x*5+4] += ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(4*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(4*cuda.blockDim.x)])+dy5*ty1*(utmp[5*cuda.threadIdx.x-1]-2.0*utmp[cuda.threadIdx.x*5+4]+utmp[5*cuda.threadIdx.x+9])
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			if j==1:
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(5.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[5*cuda.threadIdx.x+m+5]+u[(m)+5*((i)+nx*((3)+ny*(k)))])
			if j==2:
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(-4.0*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((i)+nx*((4)+ny*(k)))])
			if (j>=3) and(j<(ny-3)):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= ( dssp*(u[(m)+5*((i)+nx*((j-2)+ny*(k)))]-4.0
								*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((i)+nx*((j+2)+ny*(k)))]) )
			if j==(ny-3):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= ( dssp*(u[(m)+5*((i)+nx*((ny-5)+ny*(k)))]-4.0
								*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]) )
			if j==(ny-2):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(u[(m)+5*((i)+nx*((ny-4)+ny*(k)))]-4.0*utmp[cuda.threadIdx.x*5+m-5]+5.0*utmp[cuda.threadIdx.x*5+m])

		m = cuda.threadIdx.x
		rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))] = rtmp[m]
		m += nthreads                              
		rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))] = rtmp[m]
		j += cuda.blockDim.x-2
	#END while j < ny:
#END rhs_gpu_kernel_3()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def rhs_gpu_kernel_2(u,
					rsd,
					qs,
					rho_i,
					nx,
					ny,
					nz,
					tx1, tx2, tx3,
					dx1, dx2, dx3, dx4, dx5,
					dssp):
	#double* flux = (double*)extern_share_data;
	flux = cuda.shared.array(shape=0, dtype=numba.float64)
	utmp = flux[(cuda.blockDim.x*5):] 
	rtmp = utmp[(cuda.blockDim.x*5):]
	rhotmp = rtmp[(cuda.blockDim.x*5):]
	u21i = rhotmp[(cuda.blockDim.x):]
	u31i = u21i[(cuda.blockDim.x):]
	u41i = u31i[(cuda.blockDim.x):]
	u51i = u41i[(cuda.blockDim.x):]
	
	k = cuda.blockIdx.x+1
	j = cuda.blockIdx.y+1
	i = cuda.threadIdx.x

	while i < nx:
		nthreads = nx-(i-cuda.threadIdx.x)
		if nthreads > cuda.blockDim.x:
			nthreads = cuda.blockDim.x
		m = cuda.threadIdx.x;
		utmp[m] = u[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		utmp[m] = u[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		rhotmp[cuda.threadIdx.x] = rho_i[(i)+nx*((j)+ny*(k))]
		cuda.syncthreads()
		# ---------------------------------------------------------------------
		# xi-direction flux differences
		# ---------------------------------------------------------------------
		flux[cuda.threadIdx.x+(0*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+1]
		u21 = utmp[cuda.threadIdx.x*5+1]*rhotmp[cuda.threadIdx.x]
		q = qs[(i)+nx*((j)+ny*(k))]
		flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+1]*u21+C2*(utmp[cuda.threadIdx.x*5+4]-q)
		flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+2]*u21
		flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = utmp[cuda.threadIdx.x*5+3]*u21
		flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = (C1*utmp[cuda.threadIdx.x*5+4]-C2*q)*u21
		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (i<(nx-1)):
			for m in range(5):
				rtmp[cuda.threadIdx.x*5+m] = rtmp[cuda.threadIdx.x*5+m]-tx2*(flux[(cuda.threadIdx.x+1)+(m*cuda.blockDim.x)]-flux[(cuda.threadIdx.x-1)+(m*cuda.blockDim.x)])
		u21i[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+1]
		u31i[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+2]
		u41i[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+3]
		u51i[cuda.threadIdx.x] = rhotmp[cuda.threadIdx.x]*utmp[cuda.threadIdx.x*5+4]
		cuda.syncthreads()
		if cuda.threadIdx.x >= 1:
			flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = (4.0/3.0)*tx3*(u21i[cuda.threadIdx.x]-u21i[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = tx3*(u31i[cuda.threadIdx.x]-u31i[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = tx3*(u41i[cuda.threadIdx.x]-u41i[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = ( 0.5*(1.0-C1*C5)*tx3*((u21i[cuda.threadIdx.x]*u21i[cuda.threadIdx.x]+u31i[cuda.threadIdx.x]*u31i[cuda.threadIdx.x]+u41i[cuda.threadIdx.x]*u41i[cuda.threadIdx.x])-(u21i[cuda.threadIdx.x-1]*u21i[cuda.threadIdx.x-1]+u31i[cuda.threadIdx.x-1]*u31i[cuda.threadIdx.x-1]+u41i[cuda.threadIdx.x-1]*u41i[cuda.threadIdx.x-1]))+(1.0/6.0)*tx3*(u21i[cuda.threadIdx.x]*u21i[cuda.threadIdx.x]-u21i[cuda.threadIdx.x-1]*u21i[cuda.threadIdx.x-1]) + C1*C5*tx3*(u51i[cuda.threadIdx.x]-u51i[cuda.threadIdx.x-1]) )

		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (i<(nx-1)):
			rtmp[cuda.threadIdx.x*5+0] += dx1*tx1*(utmp[cuda.threadIdx.x*5-5]-2.0*utmp[cuda.threadIdx.x*5+0]+utmp[cuda.threadIdx.x*5+5])
			rtmp[cuda.threadIdx.x*5+1] += tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(1*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(1*cuda.blockDim.x)])+dx2*tx1*(utmp[cuda.threadIdx.x*5-4]-2.0*utmp[cuda.threadIdx.x*5+1]+utmp[cuda.threadIdx.x*5+6])
			rtmp[cuda.threadIdx.x*5+2] += tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(2*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(2*cuda.blockDim.x)])+dx3*tx1*(utmp[cuda.threadIdx.x*5-3]-2.0*utmp[cuda.threadIdx.x*5+2]+utmp[cuda.threadIdx.x*5+7])
			rtmp[cuda.threadIdx.x*5+3] += tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(3*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(3*cuda.blockDim.x)])+dx4*tx1*(utmp[cuda.threadIdx.x*5-2]-2.0*utmp[cuda.threadIdx.x*5+3]+utmp[cuda.threadIdx.x*5+8])
			rtmp[cuda.threadIdx.x*5+4] += tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(4*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(4*cuda.blockDim.x)])+dx5*tx1*(utmp[cuda.threadIdx.x*5-1]-2.0*utmp[cuda.threadIdx.x*5+4]+utmp[cuda.threadIdx.x*5+9])
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			if i==1: 
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(5.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((3)+nx*((j)+ny*(k)))])
			if i==2: 
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(-4.0*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((4)+nx*((j)+ny*(k)))])
			if (i>=3) and (i<(nx-3)):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= ( dssp*(u[(m)+5*((i-2)+nx*((j)+ny*(k)))]-4.0
									*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]+u[(m)+5*((i+2)+nx*((j)+ny*(k)))]) )
			if i==(nx-3):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= (  dssp*(u[(m)+5*((nx-5)+nx*((j)+ny*(k)))]-4.0
									*utmp[cuda.threadIdx.x*5+m-5]+6.0*utmp[cuda.threadIdx.x*5+m]-4.0*utmp[cuda.threadIdx.x*5+m+5]) )
			if i==(nx-2):
				for m in range(5):
					rtmp[cuda.threadIdx.x*5+m] -= dssp*(u[(m)+5*((nx-4)+nx*((j)+ny*(k)))]-4.0*utmp[cuda.threadIdx.x*5+m-5]+5.0*utmp[cuda.threadIdx.x*5+m])

		m = cuda.threadIdx.x
		rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))] = rtmp[m]
		m += nthreads
		rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))] = rtmp[m]
		m += nthreads
		rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))] = rtmp[m]
		m += nthreads
		rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))] = rtmp[m]
		m += nthreads
		rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))] = rtmp[m]
		i += cuda.blockDim.x-2
	#END while i < nx:
#END rhs_gpu_kernel_2()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32)')
def rhs_gpu_kernel_1(u,
					rsd,
					frct,
					qs,
					rho_i,
					nx,
					ny,
					nz):
	i_j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j_k % nx
	j = int(i_j_k / nx) % ny
	k = int(i_j_k / (nx * ny))

	if i_j_k >= (nx*ny*nz):
		return

	for m in range(5):
		rsd[(m)+5*(( i)+nx*(( j)+ny*( k)))] = -frct[(m)+5*(( i)+nx*(( j)+ny*( k)))]
	rho_i[(i)+nx*(( j)+ny*( k))] = tmp = 1.0 / u[(0)+5*(( i)+nx*(( j)+ny*( k)))]
	qs[(i)+nx*(( j)+ny*( k))] = 0.5*(u[(1)+5*((i)+nx*((j)+ny*(k)))]*u[(1)+5*((i)+nx*((j)+ny*(k)))]+u[(2)+5*((i)+nx*((j)+ny*(k)))]*u[(2)+5*((i)+nx*((j)+ny*(k)))]+u[(3)+5*((i)+nx*((j)+ny*(k)))]*u[(3)+5*((i)+nx*((j)+ny*(k)))])*tmp
#END rhs_gpu_kernel_1()


def rhs_gpu(u_device, 
			rsd_device,
			frct_device,
			qs_device, 
			rho_i_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_1)
	# #KERNEL RHS 1
	rhs_1_workload = nx * ny * nz
	rhs_1_threads_per_block = THREADS_PER_BLOCK_ON_RHS_1
	rhs_1_blocks_per_grid = math.ceil(rhs_1_workload / rhs_1_threads_per_block)

	rhs_gpu_kernel_1[rhs_1_blocks_per_grid, 
		rhs_1_threads_per_block](u_device, 
								rsd_device, 
								frct_device, 
								qs_device, 
								rho_i_device,
								nx, 
								ny, 
								nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_1)

	# ---------------------------------------------------------------------
	# xi-direction flux differences
	# ---------------------------------------------------------------------
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_2)
	# #KERNEL RHS 2
	rhs_2_blocks_per_grid = (nz-2, ny-2)
	rhs_2_threads_per_block = THREADS_PER_BLOCK_ON_RHS_2
	if THREADS_PER_BLOCK_ON_RHS_2 != device_prop.WARP_SIZE:
		rhs_2_threads_per_block = device_prop.WARP_SIZE
	size_shared_data = rsd_device.dtype.itemsize * ( (3*(min(nx,rhs_2_threads_per_block))*5) + (5*(min(nx,rhs_2_threads_per_block))) )
	#print("threadSize=[%d]" % (rhs_2_threads_per_block))
	#print("blockSize=[%d, %d]" % (rhs_2_blocks_per_grid[0], rhs_2_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	rhs_gpu_kernel_2[rhs_2_blocks_per_grid, 
		(min(nx,rhs_2_threads_per_block)),
		stream,
		size_shared_data](u_device, 
						rsd_device, 
						qs_device, 
						rho_i_device, 
						nx, 
						ny, 
						nz,
						tx1, tx2, tx3,
						dx1, dx2, dx3, dx4, dx5,
						dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_2)

	# ---------------------------------------------------------------------
	# eta-direction flux differences
	# ---------------------------------------------------------------------
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_3)
	# #KERNEL RHS 3
	rhs_3_blocks_per_grid = (nz-2, nx-2)
	rhs_3_threads_per_block = THREADS_PER_BLOCK_ON_RHS_3
	if THREADS_PER_BLOCK_ON_RHS_3 != device_prop.WARP_SIZE:
		rhs_3_threads_per_block = device_prop.WARP_SIZE
	size_shared_data = rsd_device.dtype.itemsize * ( (3*(min(ny,rhs_3_threads_per_block))*5) + (5*(min(ny,rhs_3_threads_per_block))) )
	#print("threadSize=[%d]" % (rhs_3_threads_per_block))
	#print("blockSize=[%d, %d]" % (rhs_3_blocks_per_grid[0], rhs_3_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	rhs_gpu_kernel_3[rhs_3_blocks_per_grid, 
		(min(ny,rhs_3_threads_per_block)),
		stream,
		size_shared_data](u_device, 
						rsd_device, 
						qs_device, 
						rho_i_device, 
						nx, 
						ny, 
						nz,
						ty1, ty2, ty3,
						dy1, dy2, dy3, dy4, dy5,
						dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_3)

	# ---------------------------------------------------------------------
	# zeta-direction flux differences
	# ---------------------------------------------------------------------
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_4)
	# #KERNEL RHS 4
	rhs_4_blocks_per_grid = (ny-2, nx-2)
	rhs_4_threads_per_block = THREADS_PER_BLOCK_ON_RHS_4
	if THREADS_PER_BLOCK_ON_RHS_4 != device_prop.WARP_SIZE:
		rhs_4_threads_per_block = device_prop.WARP_SIZE
	size_shared_data = rsd_device.dtype.itemsize * ( (3*(min(nz,rhs_4_threads_per_block))*5) + (5*(min(nz,rhs_4_threads_per_block))) )
	#print("threadSize=[%d]" % (rhs_4_threads_per_block))
	#print("blockSize=[%d, %d]" % (rhs_4_blocks_per_grid[0], rhs_4_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	rhs_gpu_kernel_4[rhs_4_blocks_per_grid, 
		(min(nz,rhs_4_threads_per_block)),
		stream,
		size_shared_data](u_device, 
						rsd_device, 
						qs_device, 
						rho_i_device, 
						nx, 
						ny, 
						nz,
						tz1, tz2, tz3,
						dz1, dz2, dz3, dz4, dz5,
						dssp);
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_4)
#END rhs_gpu()


@cuda.jit('void(int32, int32, int32, float64[:],  float64[:],  float64[:], float64[:], int32, int32, int32, float64[:])')
def jacu_buts_gpu_kernel(plane,
						klower,
						jlower,
						u,
						rho_i,
						qs,
						v,
						nx,
						ny,
						nz,
						const_arr):
	tmat = cuda.local.array(5*5, numba.float64)
	tv = cuda.local.array(5, numba.float64)

	k = klower + cuda.blockIdx.x + 1
	j = jlower + cuda.threadIdx.x + 1

	i = plane-j-k+3

	if (i<1) or (i>(nx-2)) or (j>(ny-2)):
		return

	r43 = 4.0/3.0
	c1345 = C1*C3*C4*C5
	c34 = C3*C4
	
	#Load constants
	dt, omega = const_arr[0], const_arr[1]
	tx1, tx2, ty1, ty2, tz1, tz2 = const_arr[2], const_arr[3], const_arr[4], const_arr[5], const_arr[6], const_arr[7]
	dx1, dx2, dx3, dx4, dx5 = const_arr[8], const_arr[9], const_arr[10], const_arr[11], const_arr[12]
	dy1, dy2, dy3, dy4, dy5 = const_arr[13], const_arr[14], const_arr[15], const_arr[16], const_arr[17]
	dz1, dz2, dz3, dz4, dz5 = const_arr[18], const_arr[19], const_arr[20], const_arr[21], const_arr[22]
	
	# ---------------------------------------------------------------------
	# form the first block sub-diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i+1)+nx*((j)+ny*(k))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = -dt*tx1*dx1
	tmat[0+5*1] = dt*tx2
	tmat[0+5*2] = 0.0
	tmat[0+5*3] = 0.0
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = dt*tx2*(-(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)+C2*qs[(i+1)+nx*((j)+ny*(k))]*tmp1)-dt*tx1*(-r43*c34*tmp2*u[(1)+5*((i+1)+nx*((j)+ny*(k)))])
	tmat[1+5*1] = dt*tx2*((2.0-C2)*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1))-dt*tx1*(r43*c34*tmp1)-dt*tx1*dx2
	tmat[1+5*2] = dt*tx2*(-C2*(u[(2)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1))
	tmat[1+5*3] = dt*tx2*(-C2*(u[(3)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1))
	tmat[1+5*4] = dt*tx2*C2
	tmat[2+5*0] = dt*tx2*(-(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*u[(2)+5*((i+1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(-c34*tmp2*u[(2)+5*((i+1)+nx*((j)+ny*(k)))])
	tmat[2+5*1] = dt*tx2*(u[(2)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)
	tmat[2+5*2] = dt*tx2*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx3
	tmat[2+5*3] = 0.0
	tmat[2+5*4] = 0.0
	tmat[3+5*0] = dt*tx2*(-(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*u[(3)+5*((i+1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(-c34*tmp2*u[(3)+5*((i+1)+nx*((j)+ny*(k)))])
	tmat[3+5*1] = dt*tx2*(u[(3)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)
	tmat[3+5*2] = 0.0
	tmat[3+5*3] = dt*tx2*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx4
	tmat[3+5*4] = 0.0
	tmat[4+5*0] = dt*tx2*((C2*2.0*qs[(i+1)+nx*((j)+ny*(k))]-C1*u[(4)+5*((i+1)+nx*((j)+ny*(k)))])*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp2))-dt*tx1*(-(r43*c34-c1345)*tmp3*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*u[(1)+5*((i+1)+nx*((j)+ny*(k)))])-(c34-c1345)*tmp3*(u[(2)+5*((i+1)+nx*((j)+ny*(k)))]*u[(2)+5*((i+1)+nx*((j)+ny*(k)))])-(c34-c1345)*tmp3*(u[(3)+5*((i+1)+nx*((j)+ny*(k)))]*u[(3)+5*((i+1)+nx*((j)+ny*(k)))])-c1345*tmp2*u[(4)+5*((i+1)+nx*((j)+ny*(k)))])
	tmat[4+5*1] = dt*tx2*(C1*(u[(4)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1)-C2*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp2+qs[(i+1)+nx*((j)+ny*(k))]*tmp1))-dt*tx1*(r43*c34-c1345)*tmp2*u[(1)+5*((i+1)+nx*((j)+ny*(k)))]
	tmat[4+5*2] = dt*tx2*(-C2*(u[(2)+5*((i+1)+nx*((j)+ny*(k)))]*u[(1)+5*((i+1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(c34-c1345)*tmp2*u[(2)+5*((i+1)+nx*((j)+ny*(k)))]
	tmat[4+5*3] = dt*tx2*(-C2*(u[(3)+5*((i+1)+nx*((j)+ny*(k)))]*u[(1)+5*((i+1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(c34-c1345)*tmp2*u[(3)+5*((i+1)+nx*((j)+ny*(k)))]
	tmat[4+5*4] = dt*tx2*(C1*(u[(1)+5*((i+1)+nx*((j)+ny*(k)))]*tmp1))-dt*tx1*c1345*tmp1-dt*tx1*dx5
	for m in range(5):
		tv[m] = omega*(tmat[m+5*0]*v[(0)+5*((i+1)+nx*((j)+ny*(k)))]+tmat[m+5*1]*v[(1)+5*((i+1)+nx*((j)+ny*(k)))]+tmat[m+5*2]*v[(2)+5*((i+1)+nx*((j)+ny*(k)))]+tmat[m+5*3]*v[(3)+5*((i+1)+nx*((j)+ny*(k)))]+tmat[m+5*4]*v[(4)+5*((i+1)+nx*((j)+ny*(k)))])
	# ---------------------------------------------------------------------
	# form the second block sub-diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i)+nx*((j+1)+ny*(k))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = -dt*ty1*dy1
	tmat[0+5*1] = 0.0
	tmat[0+5*2] = dt*ty2
	tmat[0+5*3] = 0.0
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = dt*ty2*(-(u[(1)+5*((i)+nx*((j+1)+ny*(k)))]*u[(2)+5*((i)+nx*((j+1)+ny*(k)))])*tmp2)-dt*ty1*(-c34*tmp2*u[(1)+5*((i)+nx*((j+1)+ny*(k)))])
	tmat[1+5*1] = dt*ty2*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy2
	tmat[1+5*2] = dt*ty2*(u[(1)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)
	tmat[1+5*3] = 0.0
	tmat[1+5*4] = 0.0
	tmat[2+5*0] = dt*ty2*(-(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)+C2*(qs[(i)+nx*((j+1)+ny*(k))]*tmp1))-dt*ty1*(-r43*c34*tmp2*u[(2)+5*((i)+nx*((j+1)+ny*(k)))])
	tmat[2+5*1] = dt*ty2*(-C2*(u[(1)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1))
	tmat[2+5*2] = dt*ty2*((2.0-C2)*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1))-dt*ty1*(r43*c34*tmp1)-dt*ty1*dy3
	tmat[2+5*3] = dt*ty2*(-C2*(u[(3)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1))
	tmat[2+5*4] = dt*ty2*C2
	tmat[3+5*0] = dt*ty2*(-(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*u[(3)+5*((i)+nx*((j+1)+ny*(k)))])*tmp2)-dt*ty1*(-c34*tmp2*u[(3)+5*((i)+nx*((j+1)+ny*(k)))])
	tmat[3+5*1] = 0.0
	tmat[3+5*2] = dt*ty2*(u[(3)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)
	tmat[3+5*3] = dt*ty2*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy4
	tmat[3+5*4] = 0.0
	tmat[4+5*0] = dt*ty2*((C2*2.0*qs[(i)+nx*((j+1)+ny*(k))]-C1*u[(4)+5*((i)+nx*((j+1)+ny*(k)))])*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp2))-dt*ty1*(-(c34-c1345)*tmp3*(u[(1)+5*((i)+nx*((j+1)+ny*(k)))]*u[(1)+5*((i)+nx*((j+1)+ny*(k)))])-(r43*c34-c1345)*tmp3*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*u[(2)+5*((i)+nx*((j+1)+ny*(k)))])-(c34-c1345)*tmp3*(u[(3)+5*((i)+nx*((j+1)+ny*(k)))]*u[(3)+5*((i)+nx*((j+1)+ny*(k)))])-c1345*tmp2*u[(4)+5*((i)+nx*((j+1)+ny*(k)))])
	tmat[4+5*1] = dt*ty2*(-C2*(u[(1)+5*((i)+nx*((j+1)+ny*(k)))]*u[(2)+5*((i)+nx*((j+1)+ny*(k)))])*tmp2)-dt*ty1*(c34-c1345)*tmp2*u[(1)+5*((i)+nx*((j+1)+ny*(k)))]
	tmat[4+5*2] = dt*ty2*(C1*(u[(4)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1)-C2*(qs[(i)+nx*((j+1)+ny*(k))]*tmp1+u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp2))-dt*ty1*(r43*c34-c1345)*tmp2*u[(2)+5*((i)+nx*((j+1)+ny*(k)))]
	tmat[4+5*3] = dt*ty2*(-C2*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*u[(3)+5*((i)+nx*((j+1)+ny*(k)))])*tmp2)-dt*ty1*(c34-c1345)*tmp2*u[(3)+5*((i)+nx*((j+1)+ny*(k)))]
	tmat[4+5*4] = dt*ty2*(C1*(u[(2)+5*((i)+nx*((j+1)+ny*(k)))]*tmp1))-dt*ty1*c1345*tmp1-dt*ty1*dy5
	for m in range(5):
		tv[m] = tv[m]+omega*(tmat[m+5*0]*v[(0)+5*((i)+nx*((j+1)+ny*(k)))]+tmat[m+5*1]*v[(1)+5*((i)+nx*((j+1)+ny*(k)))]+tmat[m+5*2]*v[(2)+5*((i)+nx*((j+1)+ny*(k)))]+tmat[m+5*3]*v[(3)+5*((i)+nx*((j+1)+ny*(k)))]+tmat[m+5*4]*v[(4)+5*((i)+nx*((j+1)+ny*(k)))])
	# ---------------------------------------------------------------------
	# form the third block sub-diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i)+nx*((j)+ny*(k+1))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = -dt*tz1*dz1
	tmat[0+5*1] = 0.0
	tmat[0+5*2] = 0.0
	tmat[0+5*3] = dt*tz2
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = dt*tz2*(-(u[(1)+5*((i)+nx*((j)+ny*(k+1)))]*u[(3)+5*((i)+nx*((j)+ny*(k+1)))])*tmp2)-dt*tz1*(-c34*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k+1)))])
	tmat[1+5*1] = dt*tz2*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)-dt*tz1*c34*tmp1-dt*tz1*dz2
	tmat[1+5*2] = 0.0
	tmat[1+5*3] = dt*tz2*(u[(1)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)
	tmat[1+5*4] = 0.0
	tmat[2+5*0] = dt*tz2*(-(u[(2)+5*((i)+nx*((j)+ny*(k+1)))]*u[(3)+5*((i)+nx*((j)+ny*(k+1)))])*tmp2)-dt*tz1*(-c34*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k+1)))])
	tmat[2+5*1] = 0.0
	tmat[2+5*2] = dt*tz2*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)-dt*tz1*(c34*tmp1)-dt*tz1*dz3
	tmat[2+5*3] = dt*tz2*(u[(2)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)
	tmat[2+5*4] = 0.0
	tmat[3+5*0] = dt*tz2*(-(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)+C2*(qs[(i)+nx*((j)+ny*(k+1))]*tmp1))-dt*tz1*(-r43*c34*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k+1)))])
	tmat[3+5*1] = dt*tz2*(-C2*(u[(1)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1))
	tmat[3+5*2] = dt*tz2*(-C2*(u[(2)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1))
	tmat[3+5*3] = dt*tz2*(2.0-C2)*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)-dt*tz1*(r43*c34*tmp1)-dt*tz1*dz4
	tmat[3+5*4] = dt*tz2*C2
	tmat[4+5*0] = dt*tz2*((C2*2.0*qs[(i)+nx*((j)+ny*(k+1))]-C1*u[(4)+5*((i)+nx*((j)+ny*(k+1)))])*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp2))-dt*tz1*(-(c34-c1345)*tmp3*(u[(1)+5*((i)+nx*((j)+ny*(k+1)))]*u[(1)+5*((i)+nx*((j)+ny*(k+1)))])-(c34-c1345)*tmp3*(u[(2)+5*((i)+nx*((j)+ny*(k+1)))]*u[(2)+5*((i)+nx*((j)+ny*(k+1)))])-(r43*c34-c1345)*tmp3*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*u[(3)+5*((i)+nx*((j)+ny*(k+1)))])-c1345*tmp2*u[(4)+5*((i)+nx*((j)+ny*(k+1)))])
	tmat[4+5*1] = dt*tz2*(-C2*(u[(1)+5*((i)+nx*((j)+ny*(k+1)))]*u[(3)+5*((i)+nx*((j)+ny*(k+1)))])*tmp2)-dt*tz1*(c34-c1345)*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k+1)))]
	tmat[4+5*2] = dt*tz2*(-C2*(u[(2)+5*((i)+nx*((j)+ny*(k+1)))]*u[(3)+5*((i)+nx*((j)+ny*(k+1)))])*tmp2)-dt*tz1*(c34-c1345)*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k+1)))]
	tmat[4+5*3] = dt*tz2*(C1*(u[(4)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1)-C2*(qs[(i)+nx*((j)+ny*(k+1))]*tmp1+u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp2))-dt*tz1*(r43*c34-c1345)*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k+1)))]
	tmat[4+5*4] = dt*tz2*(C1*(u[(3)+5*((i)+nx*((j)+ny*(k+1)))]*tmp1))-dt*tz1*c1345*tmp1-dt*tz1*dz5
	for m in range(5):
		tv[m] = tv[m]+omega*(tmat[m+5*0]*v[(0)+5*((i)+nx*((j)+ny*(k+1)))]+tmat[m+5*1]*v[(1)+5*((i)+nx*((j)+ny*(k+1)))]+tmat[m+5*2]*v[(2)+5*((i)+nx*((j)+ny*(k+1)))]+tmat[m+5*3]*v[(3)+5*((i)+nx*((j)+ny*(k+1)))]+tmat[m+5*4]*v[(4)+5*((i)+nx*((j)+ny*(k+1)))])
	# ---------------------------------------------------------------------
	# form the block daigonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i)+nx*((j)+ny*(k))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = 1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1)
	tmat[0+5*1] = 0.0
	tmat[0+5*2] = 0.0
	tmat[0+5*3] = 0.0
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = dt*2.0*(-tx1*r43-ty1-tz1)*(c34*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k)))])
	tmat[1+5*1] = 1.0+dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1)+dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2)
	tmat[1+5*2] = 0.0
	tmat[1+5*3] = 0.0
	tmat[1+5*4] = 0.0
	tmat[2+5*0] = dt*2.0*(-tx1-ty1*r43-tz1)*(c34*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k)))])
	tmat[2+5*1] = 0.0
	tmat[2+5*2] = 1.0+dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1)+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3)
	tmat[2+5*3] = 0.0
	tmat[2+5*4] = 0.0
	tmat[3+5*0] = dt*2.0*(-tx1-ty1-tz1*r43)*(c34*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k)))])
	tmat[3+5*1] = 0.0
	tmat[3+5*2] = 0.0
	tmat[3+5*3] = 1.0+dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43)+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4)
	tmat[3+5*4] = 0.0
	tmat[4+5*0] = -dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u[(1)+5*((i)+nx*((j)+ny*(k)))]*u[(1)+5*((i)+nx*((j)+ny*(k)))])+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u[(2)+5*((i)+nx*((j)+ny*(k)))]*u[(2)+5*((i)+nx*((j)+ny*(k)))])+(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u[(3)+5*((i)+nx*((j)+ny*(k)))]*u[(3)+5*((i)+nx*((j)+ny*(k)))]))*tmp3+(tx1+ty1+tz1)*c1345*tmp2*u[(4)+5*((i)+nx*((j)+ny*(k)))])
	tmat[4+5*1] = dt*2.0*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k)))]
	tmat[4+5*2] = dt*2.0*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k)))]
	tmat[4+5*3] = dt*2.0*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k)))]
	tmat[4+5*4] = 1.0 + dt*2.0*(tx1+ty1+tz1)*c1345*tmp1+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5)
	# ---------------------------------------------------------------------
	# diagonal block inversion
	# ---------------------------------------------------------------------
	tmp1 = 1.0/tmat[0+0*5]
	tmp = tmp1*tmat[1+0*5]
	tmat[1+1*5] -= tmp*tmat[0+1*5]
	tmat[1+2*5] -= tmp*tmat[0+2*5]
	tmat[1+3*5] -= tmp*tmat[0+3*5]
	tmat[1+4*5] -= tmp*tmat[0+4*5]
	tv[1] -= tmp*tv[0]
	tmp = tmp1*tmat[2+0*5]
	tmat[2+1*5] -= tmp*tmat[0+1*5]
	tmat[2+2*5] -= tmp*tmat[0+2*5]
	tmat[2+3*5] -= tmp*tmat[0+3*5]
	tmat[2+4*5] -= tmp*tmat[0+4*5]
	tv[2] -= tmp*tv[0]
	tmp = tmp1*tmat[3+0*5]
	tmat[3+1*5] -= tmp*tmat[0+1*5]
	tmat[3+2*5] -= tmp*tmat[0+2*5]
	tmat[3+3*5] -= tmp*tmat[0+3*5]
	tmat[3+4*5] -= tmp*tmat[0+4*5]
	tv[3] -= tmp*tv[0]
	tmp = tmp1*tmat[4+0*5]
	tmat[4+1*5] -= tmp*tmat[0+1*5]
	tmat[4+2*5] -= tmp*tmat[0+2*5]
	tmat[4+3*5] -= tmp*tmat[0+3*5]
	tmat[4+4*5] -= tmp*tmat[0+4*5]
	tv[4] -= tmp*tv[0]
	tmp1 = 1.0/tmat[1+1*5]
	tmp = tmp1*tmat[2+1*5]
	tmat[2+2*5] -= tmp*tmat[1+2*5]
	tmat[2+3*5] -= tmp*tmat[1+3*5]
	tmat[2+4*5] -= tmp*tmat[1+4*5]
	tv[2] -= tmp*tv[1]
	tmp = tmp1*tmat[3+1*5]
	tmat[3+2*5] -= tmp*tmat[1+2*5]
	tmat[3+3*5] -= tmp*tmat[1+3*5]
	tmat[3+4*5] -= tmp*tmat[1+4*5]
	tv[3] -= tmp*tv[1]
	tmp = tmp1*tmat[4+1*5]
	tmat[4+2*5] -= tmp*tmat[1+2*5]
	tmat[4+3*5] -= tmp*tmat[1+3*5]
	tmat[4+4*5] -= tmp*tmat[1+4*5]
	tv[4] -= tmp*tv[1]
	tmp1 = 1.0/tmat[2+2*5]
	tmp = tmp1*tmat[3+2*5]
	tmat[3+3*5] -= tmp*tmat[2+3*5]
	tmat[3+4*5] -= tmp*tmat[2+4*5]
	tv[3] -= tmp*tv[2]
	tmp = tmp1*tmat[4+2*5]
	tmat[4+3*5] -= tmp*tmat[2+3*5]
	tmat[4+4*5] -= tmp*tmat[2+4*5]
	tv[4] -= tmp*tv[2]
	tmp1 = 1.0/tmat[3+3*5]
	tmp = tmp1*tmat[4+3*5]
	tmat[4+4*5] -= tmp*tmat[3+4*5]
	tv[4] -= tmp*tv[3]
	# ---------------------------------------------------------------------
	# back substitution
	# ---------------------------------------------------------------------
	tv[4] = tv[4]/tmat[4+4*5]
	tv[3] = tv[3]-tmat[3+4*5]*tv[4]
	tv[3] = tv[3]/tmat[3+3*5]
	tv[2] = tv[2]-tmat[2+3*5]*tv[3]-tmat[2+4*5]*tv[4]
	tv[2] = tv[2]/tmat[2+2*5]
	tv[1] = tv[1]-tmat[1+2*5]*tv[2]-tmat[1+3*5]*tv[3]-tmat[1+4*5]*tv[4]
	tv[1] = tv[1]/tmat[1+1*5]
	tv[0] = tv[0]-tmat[0+1*5]*tv[1]-tmat[0+2*5]*tv[2]-tmat[0+3*5]*tv[3]-tmat[0+4*5]*tv[4]
	tv[0] = tv[0]/tmat[0+0*5]
	v[(0)+5*((i)+nx*((j)+ny*(k)))] -= tv[0]
	v[(1)+5*((i)+nx*((j)+ny*(k)))] -= tv[1]
	v[(2)+5*((i)+nx*((j)+ny*(k)))] -= tv[2]
	v[(3)+5*((i)+nx*((j)+ny*(k)))] -= tv[3]
	v[(4)+5*((i)+nx*((j)+ny*(k)))] -= tv[4]
#END jacu_buts_gpu_kernel()


@cuda.jit('void(int32, int32, int32, float64[:],  float64[:],  float64[:], float64[:], int32, int32, int32, float64[:])')
def jacld_blts_gpu_kernel(plane,
		klower,
		jlower,
		u,
		rho_i,
		qs,
		v,
		nx,
		ny,
		nz,
		const_arr):
	tmat = cuda.local.array(5*5, numba.float64)
	tv = cuda.local.array(5, numba.float64)
	
	k = klower + cuda.blockIdx.x + 1
	j = jlower + cuda.threadIdx.x + 1

	i = plane-k-j+3

	if (j>(ny-2)) or (i>(nx-2)) or (i<1):
		return

	r43 = 4.0/3.0
	c1345 = C1*C3*C4*C5
	c34 = C3*C4
	
	#Load constants
	dt, omega = const_arr[0], const_arr[1]
	tx1, tx2, ty1, ty2, tz1, tz2 = const_arr[2], const_arr[3], const_arr[4], const_arr[5], const_arr[6], const_arr[7]
	dx1, dx2, dx3, dx4, dx5 = const_arr[8], const_arr[9], const_arr[10], const_arr[11], const_arr[12]
	dy1, dy2, dy3, dy4, dy5 = const_arr[13], const_arr[14], const_arr[15], const_arr[16], const_arr[17]
	dz1, dz2, dz3, dz4, dz5 = const_arr[18], const_arr[19], const_arr[20], const_arr[21], const_arr[22]
	
	# ---------------------------------------------------------------------
	# form the first block sub-diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i)+nx*((j)+ny*(k-1))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] =  -dt*tz1*dz1
	tmat[0+5*1] = 0.0
	tmat[0+5*2] = 0.0
	tmat[0+5*3] = -dt*tz2
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = -dt*tz2*(-(u[(1)+5*((i)+nx*((j)+ny*(k-1)))]*u[(3)+5*((i)+nx*((j)+ny*(k-1)))])*tmp2)-dt*tz1*(-c34*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k-1)))])
	tmat[1+5*1] = -dt*tz2*(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)-dt*tz1*c34*tmp1-dt*tz1*dz2
	tmat[1+5*2] = 0.0
	tmat[1+5*3] = -dt*tz2*(u[(1)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)
	tmat[1+5*4] = 0.0
	tmat[2+5*0] = -dt*tz2*(-(u[(2)+5*((i)+nx*((j)+ny*(k-1)))]*u[(3)+5*((i)+nx*((j)+ny*(k-1)))])*tmp2)-dt*tz1*(-c34*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k-1)))])
	tmat[2+5*1] = 0.0
	tmat[2+5*2] = -dt*tz2*(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)-dt*tz1*(c34*tmp1)-dt*tz1*dz3
	tmat[2+5*3] = -dt*tz2*(u[(2)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)
	tmat[2+5*4] = 0.0
	tmat[3+5*0] = -dt*tz2*(-(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)*(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)+C2*qs[(i)+nx*((j)+ny*(k-1))]*tmp1)-dt*tz1*(-r43*c34*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k-1)))])
	tmat[3+5*1] = -dt*tz2*(-C2*(u[(1)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1))
	tmat[3+5*2] = -dt*tz2*(-C2*(u[(2)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1))
	tmat[3+5*3] = -dt*tz2*(2.0-C2)*(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)-dt*tz1*(r43*c34*tmp1)-dt*tz1*dz4
	tmat[3+5*4] = -dt*tz2*C2
	tmat[4+5*0] = -dt*tz2*((C2*2.0*qs[(i)+nx*((j)+ny*(k-1))]-C1*u[(4)+5*((i)+nx*((j)+ny*(k-1)))])*u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp2)-dt*tz1*(-(c34-c1345)*tmp3*(u[(1)+5*((i)+nx*((j)+ny*(k-1)))]*u[(1)+5*((i)+nx*((j)+ny*(k-1)))])-(c34-c1345)*tmp3*(u[(2)+5*((i)+nx*((j)+ny*(k-1)))]*u[(2)+5*((i)+nx*((j)+ny*(k-1)))])-(r43*c34-c1345)*tmp3*(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*u[(3)+5*((i)+nx*((j)+ny*(k-1)))])-c1345*tmp2*u[(4)+5*((i)+nx*((j)+ny*(k-1)))])
	tmat[4+5*1] = -dt*tz2*(-C2*(u[(1)+5*((i)+nx*((j)+ny*(k-1)))]*u[(3)+5*((i)+nx*((j)+ny*(k-1)))])*tmp2)-dt*tz1*(c34-c1345)*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k-1)))]
	tmat[4+5*2] = -dt*tz2*(-C2*(u[(2)+5*((i)+nx*((j)+ny*(k-1)))]*u[(3)+5*((i)+nx*((j)+ny*(k-1)))])*tmp2)-dt*tz1*(c34-c1345)*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k-1)))]
	tmat[4+5*3] = -dt*tz2*(C1*(u[(4)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1)-C2*(qs[(i)+nx*((j)+ny*(k-1))]*tmp1+u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp2))-dt*tz1*(r43*c34-c1345)*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k-1)))]
	tmat[4+5*4] = -dt*tz2*(C1*(u[(3)+5*((i)+nx*((j)+ny*(k-1)))]*tmp1))-dt*tz1*c1345*tmp1-dt*tz1*dz5
	for m in range(5):
		tv[m] = v[(m)+5*((i)+nx*((j)+ny*(k)))]-omega*(tmat[m+5*0]*v[(0)+5*((i)+nx*((j)+ny*(k-1)))]+tmat[m+5*1]*v[(1)+5*((i)+nx*((j)+ny*(k-1)))]+tmat[m+5*2]*v[(2)+5*((i)+nx*((j)+ny*(k-1)))]+tmat[m+5*3]*v[(3)+5*((i)+nx*((j)+ny*(k-1)))]+tmat[m+5*4]*v[(4)+5*((i)+nx*((j)+ny*(k-1)))])
	# ---------------------------------------------------------------------
	# form the second block sub-diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i)+nx*((j-1)+ny*(k))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = -dt*ty1*dy1
	tmat[0+5*1] = 0.0
	tmat[0+5*2] = -dt*ty2
	tmat[0+5*3] = 0.0
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = -dt*ty2*(-(u[(1)+5*((i)+nx*((j-1)+ny*(k)))]*u[(2)+5*((i)+nx*((j-1)+ny*(k)))])*tmp2)-dt*ty1*(-c34*tmp2*u[(1)+5*((i)+nx*((j-1)+ny*(k)))])
	tmat[1+5*1] = -dt*ty2*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy2
	tmat[1+5*2] = -dt*ty2*(u[(1)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)
	tmat[1+5*3] = 0.0
	tmat[1+5*4] = 0.0
	tmat[2+5*0] = -dt*ty2*(-(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)+C2*(qs[(i)+nx*((j-1)+ny*(k))]*tmp1))-dt*ty1*(-r43*c34*tmp2*u[(2)+5*((i)+nx*((j-1)+ny*(k)))])
	tmat[2+5*1] = -dt*ty2*(-C2*(u[(1)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1))
	tmat[2+5*2] = -dt*ty2*((2.0-C2)*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1))-dt*ty1*(r43*c34*tmp1)-dt*ty1*dy3
	tmat[2+5*3] = -dt*ty2*(-C2*(u[(3)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1))
	tmat[2+5*4] = -dt*ty2*C2
	tmat[3+5*0] = -dt*ty2*(-(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*u[(3)+5*((i)+nx*((j-1)+ny*(k)))])*tmp2)-dt*ty1*(-c34*tmp2*u[(3)+5*((i)+nx*((j-1)+ny*(k)))])
	tmat[3+5*1] = 0.0
	tmat[3+5*2] = -dt*ty2*(u[(3)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)
	tmat[3+5*3] = -dt*ty2*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)-dt*ty1*(c34*tmp1)-dt*ty1*dy4
	tmat[3+5*4] = 0.0
	tmat[4+5*0] = -dt*ty2*((C2*2.0*qs[(i)+nx*((j-1)+ny*(k))]-C1*u[(4)+5*((i)+nx*((j-1)+ny*(k)))])*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp2))-dt*ty1*(-(c34-c1345)*tmp3*(u[(1)+5*((i)+nx*((j-1)+ny*(k)))]*u[(1)+5*((i)+nx*((j-1)+ny*(k)))])-(r43*c34-c1345)*tmp3*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*u[(2)+5*((i)+nx*((j-1)+ny*(k)))])-(c34-c1345)*tmp3*(u[(3)+5*((i)+nx*((j-1)+ny*(k)))]*u[(3)+5*((i)+nx*((j-1)+ny*(k)))])-c1345*tmp2*u[(4)+5*((i)+nx*((j-1)+ny*(k)))])
	tmat[4+5*1] = -dt*ty2*(-C2*(u[(1)+5*((i)+nx*((j-1)+ny*(k)))]*u[(2)+5*((i)+nx*((j-1)+ny*(k)))])*tmp2)-dt*ty1*(c34-c1345)*tmp2*u[(1)+5*((i)+nx*((j-1)+ny*(k)))]
	tmat[4+5*2] = -dt*ty2*(C1*(u[(4)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1)-C2*(qs[(i)+nx*((j-1)+ny*(k))]*tmp1+u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp2))-dt*ty1*(r43*c34-c1345)*tmp2*u[(2)+5*((i)+nx*((j-1)+ny*(k)))]
	tmat[4+5*3] = -dt*ty2*(-C2*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*u[(3)+5*((i)+nx*((j-1)+ny*(k)))])*tmp2) - dt*ty1*(c34-c1345)*tmp2*u[(3)+5*((i)+nx*((j-1)+ny*(k)))]
	tmat[4+5*4] = -dt*ty2*(C1*(u[(2)+5*((i)+nx*((j-1)+ny*(k)))]*tmp1))-dt*ty1*c1345*tmp1-dt*ty1*dy5
	for m in range(5):
		tv[m] = tv[m]-omega*(tmat[m+5*0]*v[(0)+5*((i)+nx*((j-1)+ny*(k)))]+tmat[m+5*1]*v[(1)+5*((i)+nx*((j-1)+ny*(k)))]+tmat[m+5*2]*v[(2)+5*((i)+nx*((j-1)+ny*(k)))]+tmat[m+5*3]*v[(3)+5*((i)+nx*((j-1)+ny*(k)))]+tmat[m+5*4]*v[(4)+5*((i)+nx*((j-1)+ny*(k)))])
	# ---------------------------------------------------------------------
	# form the third block sub-diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i-1)+nx*((j)+ny*(k))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = -dt*tx1*dx1
	tmat[0+5*1] = -dt*tx2
	tmat[0+5*2] = 0.0
	tmat[0+5*3] = 0.0
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = -dt*tx2*(-(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)+C2*qs[(i-1)+nx*((j)+ny*(k))]*tmp1)-dt*tx1*(-r43*c34*tmp2*u[(1)+5*((i-1)+nx*((j)+ny*(k)))])
	tmat[1+5*1] = -dt*tx2*((2.0-C2)*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1))-dt*tx1*(r43*c34*tmp1)-dt*tx1*dx2
	tmat[1+5*2] = -dt*tx2*(-C2*(u[(2)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1))
	tmat[1+5*3] = -dt*tx2*(-C2*(u[(3)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1))
	tmat[1+5*4] = -dt*tx2*C2
	tmat[2+5*0] = -dt*tx2*(-(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*u[(2)+5*((i-1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(-c34*tmp2*u[(2)+5*((i-1)+nx*((j)+ny*(k)))])
	tmat[2+5*1] = -dt*tx2*(u[(2)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)
	tmat[2+5*2] = -dt*tx2*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx3
	tmat[2+5*3] = 0.0
	tmat[2+5*4] = 0.0
	tmat[3+5*0] = -dt*tx2*(-(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*u[(3)+5*((i-1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(-c34*tmp2*u[(3)+5*((i-1)+nx*((j)+ny*(k)))])
	tmat[3+5*1] = -dt*tx2*(u[(3)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)
	tmat[3+5*2] = 0.0
	tmat[3+5*3] = -dt*tx2*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)-dt*tx1*(c34*tmp1)-dt*tx1*dx4
	tmat[3+5*4] = 0.0
	tmat[4+5*0] = -dt*tx2*((C2*2.0*qs[(i-1)+nx*((j)+ny*(k))]-C1*u[(4)+5*((i-1)+nx*((j)+ny*(k)))])*u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp2)-dt*tx1*(-(r43*c34-c1345)*tmp3*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*u[(1)+5*((i-1)+nx*((j)+ny*(k)))])-(c34-c1345)*tmp3*(u[(2)+5*((i-1)+nx*((j)+ny*(k)))]*u[(2)+5*((i-1)+nx*((j)+ny*(k)))])-(c34-c1345)*tmp3*(u[(3)+5*((i-1)+nx*((j)+ny*(k)))]*u[(3)+5*((i-1)+nx*((j)+ny*(k)))])-c1345*tmp2*u[(4)+5*((i-1)+nx*((j)+ny*(k)))])
	tmat[4+5*1] = -dt*tx2*(C1*(u[(4)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1)-C2*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp2+qs[(i-1)+nx*((j)+ny*(k))]*tmp1))-dt*tx1*(r43*c34-c1345)*tmp2*u[(1)+5*((i-1)+nx*((j)+ny*(k)))]
	tmat[4+5*2] = -dt*tx2*(-C2*(u[(2)+5*((i-1)+nx*((j)+ny*(k)))]*u[(1)+5*((i-1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(c34-c1345)*tmp2*u[(2)+5*((i-1)+nx*((j)+ny*(k)))]
	tmat[4+5*3] = -dt*tx2*(-C2*(u[(3)+5*((i-1)+nx*((j)+ny*(k)))]*u[(1)+5*((i-1)+nx*((j)+ny*(k)))])*tmp2)-dt*tx1*(c34-c1345)*tmp2*u[(3)+5*((i-1)+nx*((j)+ny*(k)))]
	tmat[4+5*4] = -dt*tx2*(C1*(u[(1)+5*((i-1)+nx*((j)+ny*(k)))]*tmp1))-dt*tx1*c1345*tmp1-dt*tx1*dx5
	for m in range(5):
		tv[m] = tv[m]-omega*(tmat[m+0*5]*v[(0)+5*((i-1)+nx*((j)+ny*(k)))]+tmat[m+5*1]*v[(1)+5*((i-1)+nx*((j)+ny*(k)))]+tmat[m+5*2]*v[(2)+5*((i-1)+nx*((j)+ny*(k)))]+tmat[m+5*3]*v[(3)+5*((i-1)+nx*((j)+ny*(k)))]+tmat[m+5*4]*v[(4)+5*((i-1)+nx*((j)+ny*(k)))])
	# ---------------------------------------------------------------------
	# form the block diagonal
	# ---------------------------------------------------------------------
	tmp1 = rho_i[(i)+nx*((j)+ny*(k))]
	tmp2 = tmp1*tmp1
	tmp3 = tmp1*tmp2
	tmat[0+5*0] = 1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1)
	tmat[0+5*1] = 0.0
	tmat[0+5*2] = 0.0
	tmat[0+5*3] = 0.0
	tmat[0+5*4] = 0.0
	tmat[1+5*0] = -dt*2.0*(tx1*r43+ty1+tz1)*c34*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k)))]
	tmat[1+5*1] = 1.0+dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1) + dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2)
	tmat[1+5*2] = 0.0
	tmat[1+5*3] = 0.0
	tmat[1+5*4] = 0.0
	tmat[2+5*0] = -dt*2.0*(tx1+ty1*r43+tz1)*c34*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k)))]
	tmat[2+5*1] = 0.0
	tmat[2+5*2] = 1.0+dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1)+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3)
	tmat[2+5*3] = 0.0
	tmat[2+5*4] = 0.0
	tmat[3+5*0] = -dt*2.0*(tx1+ty1+tz1*r43)*c34*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k)))]
	tmat[3+5*1] = 0.0
	tmat[3+5*2] = 0.0
	tmat[3+5*3] = 1.0+dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43)+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4)
	tmat[3+5*4] = 0.0
	tmat[4+5*0] = -dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u[(1)+5*((i)+nx*((j)+ny*(k)))]*u[(1)+5*((i)+nx*((j)+ny*(k)))])+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u[(2)+5*((i)+nx*((j)+ny*(k)))]*u[(2)+5*((i)+nx*((j)+ny*(k)))])+(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u[(3)+5*((i)+nx*((j)+ny*(k)))]*u[(3)+5*((i)+nx*((j)+ny*(k)))]))*tmp3+(tx1+ty1+tz1)*c1345*tmp2*u[(4)+5*((i)+nx*((j)+ny*(k)))])
	tmat[4+5*1] = dt*2.0*tmp2*u[(1)+5*((i)+nx*((j)+ny*(k)))]*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))
	tmat[4+5*2] = dt*2.0*tmp2*u[(2)+5*((i)+nx*((j)+ny*(k)))]*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))
	tmat[4+5*3] = dt*2.0*tmp2*u[(3)+5*((i)+nx*((j)+ny*(k)))]*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))
	tmat[4+5*4] = 1.0+dt*2.0*(tx1+ty1+tz1)*c1345*tmp1+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5)
	# ---------------------------------------------------------------------
	# diagonal block inversion
	# --------------------------------------------------------------------- 
	# forward elimination
	# ---------------------------------------------------------------------
	tmp1 = 1.0/tmat[0+0*5]
	tmp2 = tmp1*tmat[1+0*5]
	tmat[1+1*5] -= tmp2*tmat[0+1*5]
	tmat[1+2*5] -= tmp2*tmat[0+2*5]
	tmat[1+3*5] -= tmp2*tmat[0+3*5]
	tmat[1+4*5] -= tmp2*tmat[0+4*5]
	tv[1]-=tmp2*tv[0]
	tmp2=tmp1*tmat[2+0*5]
	tmat[2+1*5] -= tmp2*tmat[0+1*5]
	tmat[2+2*5] -= tmp2*tmat[0+2*5]
	tmat[2+3*5] -= tmp2*tmat[0+3*5]
	tmat[2+4*5] -= tmp2*tmat[0+4*5]
	tv[2] -= tmp2*tv[0]
	tmp2 = tmp1*tmat[3+0*5]
	tmat[3+1*5] -= tmp2*tmat[0+1*5]
	tmat[3+2*5] -= tmp2*tmat[0+2*5]
	tmat[3+3*5] -= tmp2*tmat[0+3*5]
	tmat[3+4*5] -= tmp2*tmat[0+4*5]
	tv[3] -= tmp2*tv[0]
	tmp2 = tmp1*tmat[4+0*5]
	tmat[4+1*5] -= tmp2*tmat[0+1*5]
	tmat[4+2*5] -= tmp2*tmat[0+2*5]
	tmat[4+3*5] -= tmp2*tmat[0+3*5]
	tmat[4+4*5] -= tmp2*tmat[0+4*5]
	tv[4] -= tmp2*tv[0]
	tmp1 = 1.0/tmat[1+1*5]
	tmp2 = tmp1*tmat[2+1*5]
	tmat[2+2*5] -= tmp2*tmat[1+2*5]
	tmat[2+3*5] -= tmp2*tmat[1+3*5]
	tmat[2+4*5] -= tmp2*tmat[1+4*5]
	tv[2] -= tmp2*tv[1]
	tmp2 = tmp1*tmat[3+1*5]
	tmat[3+2*5] -= tmp2*tmat[1+2*5]
	tmat[3+3*5] -= tmp2*tmat[1+3*5]
	tmat[3+4*5] -= tmp2*tmat[1+4*5]
	tv[3] -= tmp2*tv[1]
	tmp2 = tmp1*tmat[4+1*5]
	tmat[4+2*5] -= tmp2*tmat[1+2*5]
	tmat[4+3*5] -= tmp2*tmat[1+3*5]
	tmat[4+4*5] -= tmp2*tmat[1+4*5]
	tv[4] -= tmp2*tv[1]
	tmp1 = 1.0/tmat[2+2*5]
	tmp2 = tmp1*tmat[3+2*5]
	tmat[3+3*5] -= tmp2*tmat[2+3*5]
	tmat[3+4*5] -= tmp2*tmat[2+4*5]
	tv[3] -= tmp2*tv[2]
	tmp2 =tmp1*tmat[4+2*5]
	tmat[4+3*5] -= tmp2*tmat[2+3*5]
	tmat[4+4*5] -= tmp2*tmat[2+4*5]
	tv[4] -= tmp2*tv[2]
	tmp1 = 1.0/tmat[3+3*5]
	tmp2 = tmp1*tmat[4+3*5]
	tmat[4+4*5] -= tmp2*tmat[3+4*5]
	tv[4] -= tmp2*tv[3]
	# ---------------------------------------------------------------------
	# back substitution
	# ---------------------------------------------------------------------
	v[(4)+5*((i)+nx*((j)+ny*(k)))] = tv[4]/tmat[4+4*5]
	tv[3] = tv[3]-tmat[3+4*5]*v[(4)+5*((i)+nx*((j)+ny*(k)))]
	v[(3)+5*((i)+nx*((j)+ny*(k)))] = tv[3]/tmat[3+3*5]
	tv[2] = tv[2]-tmat[2+3*5]*v[(3)+5*((i)+nx*((j)+ny*(k)))]-tmat[2+4*5]*v[(4)+5*((i)+nx*((j)+ny*(k)))]
	v[(2)+5*((i)+nx*((j)+ny*(k)))] = tv[2]/tmat[2+2*5]
	tv[1] = tv[1]-tmat[1+2*5]*v[(2)+5*((i)+nx*((j)+ny*(k)))]-tmat[1+3*5]*v[(3)+5*((i)+nx*((j)+ny*(k)))]-tmat[1+4*5]*v[(4)+5*((i)+nx*((j)+ny*(k)))]
	v[(1)+5*((i)+nx*((j)+ny*(k)))] = tv[1]/tmat[1+1*5]
	tv[0] = tv[0]-tmat[0+1*5]*v[(1)+5*((i)+nx*((j)+ny*(k)))]-tmat[0+2*5]*v[(2)+5*((i)+nx*((j)+ny*(k)))]-tmat[0+3*5]*v[(3)+5*((i)+nx*((j)+ny*(k)))]-tmat[0+4*5]*v[(4)+5*((i)+nx*((j)+ny*(k)))]
	v[(0)+5*((i)+nx*((j)+ny*(k)))] = tv[0]/tmat[0+0*5]
#END jacld_blts_gpu_kernel()


@cuda.jit('void(float64[:], float64[:], float64, int32, int32, int32)')
def ssor_gpu_kernel_2(u,
					rsd,
					tmp,
					nx,
					ny,
					nz):

	if cuda.threadIdx.x >= (nx-2):
		return

	i = cuda.threadIdx.x + 1
	j = cuda.blockIdx.y + 1
	k = cuda.blockIdx.x + 1

	u[(0)+5*(( i)+nx*(( j)+ny*( k)))] += tmp * rsd[(0)+5*(( i)+nx*(( j)+ny*( k)))]
	u[(1)+5*(( i)+nx*(( j)+ny*( k)))] += tmp * rsd[(1)+5*(( i)+nx*(( j)+ny*( k)))]
	u[(2)+5*(( i)+nx*(( j)+ny*( k)))] += tmp * rsd[(2)+5*(( i)+nx*(( j)+ny*( k)))]
	u[(3)+5*(( i)+nx*(( j)+ny*( k)))] += tmp * rsd[(3)+5*(( i)+nx*(( j)+ny*( k)))]
	u[(4)+5*(( i)+nx*(( j)+ny*( k)))] += tmp * rsd[(4)+5*(( i)+nx*(( j)+ny*( k)))]
#END ssor_gpu_kernel_2()


@cuda.jit('void(float64[:], int32, int32, int32, float64)')
def ssor_gpu_kernel_1(rsd,
					nx,
					ny,
					nz,
					dt):

	if cuda.threadIdx.x >= (nx-2):
		return

	i = cuda.threadIdx.x + 1
	j = cuda.blockIdx.y + 1
	k = cuda.blockIdx.x + 1

	rsd[(0)+5*(( i)+nx*(( j)+ny*( k)))] *= dt
	rsd[(1)+5*(( i)+nx*(( j)+ny*( k)))] *= dt
	rsd[(2)+5*(( i)+nx*(( j)+ny*( k)))] *= dt
	rsd[(3)+5*(( i)+nx*(( j)+ny*( k)))] *= dt
	rsd[(4)+5*(( i)+nx*(( j)+ny*( k)))] *= dt
#END ssor_gpu_kernel_1()


# ---------------------------------------------------------------------
# to perform pseudo-time stepping SSOR iterations
# for five nonlinear pde's.
# ---------------------------------------------------------------------
def ssor_gpu(niter,
			u_device, rsd_device, frct_device,
			rho_i_device, qs_device,
			norm_buffer_device, rsdnm,
			const_jac_device):
	global maxtime
	
	omega = omega_host
	tmp = 1.0/(omega*(2.0-omega))
	# ---------------------------------------------------------------------
	# compute the steady-state residuals
	# ---------------------------------------------------------------------
	rhs_gpu(u_device, rsd_device, frct_device, qs_device, rho_i_device)
	# ---------------------------------------------------------------------
	# compute the L2 norms of newton iteration residuals
	# ---------------------------------------------------------------------
	l2norm_gpu(rsd_device, rsdnm, norm_buffer_device)
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		#c_timers.timer_clear(PROFILING_ERHS_1) #Not clean time, because this routine alreary have been executed
		#c_timers.timer_clear(PROFILING_ERHS_2) #Not clean time, because this routine alreary have been executed
		#c_timers.timer_clear(PROFILING_ERHS_3) #Not clean time, because this routine alreary have been executed
		#c_timers.timer_clear(PROFILING_ERHS_4) #Not clean time, because this routine alreary have been executed
		c_timers.timer_clear(PROFILING_ERROR)
		c_timers.timer_clear(PROFILING_NORM)
		c_timers.timer_clear(PROFILING_JACLD_BLTS)
		c_timers.timer_clear(PROFILING_JACU_BUTS)
		c_timers.timer_clear(PROFILING_L2NORM)
		c_timers.timer_clear(PROFILING_PINTGR_1)
		c_timers.timer_clear(PROFILING_PINTGR_2)
		c_timers.timer_clear(PROFILING_PINTGR_3)
		c_timers.timer_clear(PROFILING_PINTGR_4)
		c_timers.timer_clear(PROFILING_RHS_1)
		c_timers.timer_clear(PROFILING_RHS_2)
		c_timers.timer_clear(PROFILING_RHS_3)
		c_timers.timer_clear(PROFILING_RHS_4)
		c_timers.timer_clear(PROFILING_SETBV_1)
		c_timers.timer_clear(PROFILING_SETBV_2)
		c_timers.timer_clear(PROFILING_SETBV_3)
		c_timers.timer_clear(PROFILING_SETIV)
		c_timers.timer_clear(PROFILING_SSOR_1)
		c_timers.timer_clear(PROFILING_SSOR_2)
	c_timers.timer_start(PROFILING_TOTAL_TIME) #start_timer
	# ---------------------------------------------------------------------
	# the timestep loop
	# ---------------------------------------------------------------------
	for istep in range(1, niter+1):
		if (istep%20)==0 or istep==itmax or istep==1:
			if niter > 1:
				print(" Time step %4d" % (istep))
		# ---------------------------------------------------------------------
		# perform SSOR iteration
		# ---------------------------------------------------------------------
		if gpu_config.PROFILING:
			c_timers.timer_start(PROFILING_SSOR_1)
		ssor_1_threads_per_block = THREADS_PER_BLOCK_ON_SSOR_1
		ssor_1_blocks_per_grid = (nz-2, ny-2)

		ssor_gpu_kernel_1[ssor_1_blocks_per_grid, 
				max(nx-2, ssor_1_threads_per_block)](rsd_device, 
													nx, 
													ny, 
													nz,
													dt_host)
		if gpu_config.PROFILING:
			c_timers.timer_stop(PROFILING_SSOR_1)
		# ---------------------------------------------------------------------
		# form the lower triangular part of the jacobian matrix
		# ---------------------------------------------------------------------
		# perform the lower triangular solution
		# ---------------------------------------------------------------------
		if gpu_config.PROFILING:
			c_timers.timer_start(PROFILING_JACLD_BLTS)

		for plane in range((nx+ny+nz-9)+1):
			klower = max(0, plane-(nx-3)-(ny-3))
			kupper = min(plane, nz-3)
			jlowermin = max(0, plane-kupper-(nx-3))
			juppermax = min(plane, ny-3)

			# #KERNEL JACLD BLTS
			jacld_blts_blocks_per_grid = kupper-klower+1
			jacld_blts_threads_per_block = THREADS_PER_BLOCK_ON_JACLD_BLTS
			if THREADS_PER_BLOCK_ON_JACLD_BLTS != (juppermax-jlowermin+1):
				jacld_blts_threads_per_block = juppermax-jlowermin+1

			jacld_blts_gpu_kernel[jacld_blts_blocks_per_grid, 
					jacld_blts_threads_per_block](plane, 
												klower, 
												jlowermin, 
												u_device, 
												rho_i_device, 
												qs_device, 
												rsd_device, 
												nx, 
												ny, 
												nz,
												const_jac_device)
		if gpu_config.PROFILING:
			c_timers.timer_stop(PROFILING_JACLD_BLTS)
		# ---------------------------------------------------------------------
		# form the strictly upper triangular part of the jacobian matrix
		# ---------------------------------------------------------------------
		# perform the upper triangular solution
		# ---------------------------------------------------------------------
		if gpu_config.PROFILING:
			c_timers.timer_start(PROFILING_JACU_BUTS)
		for plane in range((nx+ny+nz-9), -1, -1):
			klower = max(0, plane-(nx-3)-(ny-3))
			kupper = min(plane, nz-3)
			jlowermin = max(0, plane-kupper-(nx-3))
			juppermax = min(plane, ny-3)

			# #KERNEL JACLD BLTS
			jacu_buts_blocks_per_grid = kupper-klower+1
			jacu_buts_threads_per_block = THREADS_PER_BLOCK_ON_JACU_BUTS
			if THREADS_PER_BLOCK_ON_JACU_BUTS != (juppermax-jlowermin+1):
				jacu_buts_threads_per_block = juppermax-jlowermin+1

			jacu_buts_gpu_kernel[jacu_buts_blocks_per_grid, 
					jacu_buts_threads_per_block](plane, 
												klower, 
												jlowermin, 
												u_device, 
												rho_i_device, 
												qs_device, 
												rsd_device, 
												nx, 
												ny, 
												nz,
												const_jac_device)
		if gpu_config.PROFILING:
			c_timers.timer_stop(PROFILING_JACU_BUTS)
		# ---------------------------------------------------------------------
		# update the variables
		# ---------------------------------------------------------------------
		if gpu_config.PROFILING:
			c_timers.timer_start(PROFILING_SSOR_2)
		ssor_2_threads_per_block = THREADS_PER_BLOCK_ON_SSOR_2
		ssor_2_blocks_per_grid = (nz-2, ny-2)

		ssor_gpu_kernel_2[ssor_2_blocks_per_grid, 
			max(nx-2, ssor_2_threads_per_block)](u_device, 
												rsd_device, 
												tmp, 
												nx, 
												ny, 
												nz)
		if gpu_config.PROFILING:
			c_timers.timer_stop(PROFILING_SSOR_2)
		# ---------------------------------------------------------------------
		# compute the max-norms of newton iteration corrections
		# ---------------------------------------------------------------------
		if istep%inorm == 0:
			delunm = numpy.empty(5, dtype=numpy.float64)
			l2norm_gpu(rsd_device, delunm, norm_buffer_device)
		# ---------------------------------------------------------------------
		# compute the steady-state residuals
		# ---------------------------------------------------------------------
		rhs_gpu(u_device, rsd_device, frct_device, qs_device, rho_i_device)
		# ---------------------------------------------------------------------
		# compute the max-norms of newton iteration residuals
		# ---------------------------------------------------------------------
		if istep%inorm == 0:
			l2norm_gpu(rsd_device, rsdnm, norm_buffer_device)
		# ---------------------------------------------------------------------
		# check the newton-iteration residuals against the tolerance levels
		# ---------------------------------------------------------------------
		if (rsdnm[0]<tolrsd[0]) and (rsdnm[1]<tolrsd[1]) and (rsdnm[2]<tolrsd[2]) and (rsdnm[3]<tolrsd[3]) and (rsdnm[4]<tolrsd[4]):
			print("\n convergence was achieved after %4d pseudo-time steps" % istep)
			break
	#END for istep in range(1, niter+1):
	
	c_timers.timer_stop(PROFILING_TOTAL_TIME) #stop_timer
	maxtime = c_timers.timer_read(PROFILING_TOTAL_TIME)
#END ssor_gpu()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def erhs_gpu_kernel_4(frct,
					rsd,
					nx,
					ny,
					nz,
					tz1, tz2, tz3,
					dz1, dz2, dz3, dz4, dz5,
					dssp):
	#double* flux = (double*)extern_share_data;
	flux = cuda.shared.array(shape=0, dtype=numba.float64)
	rtmp = flux[(cuda.blockDim.x*5):] 
	u21k = rtmp[(cuda.blockDim.x*5):]
	u31k = u21k[(cuda.blockDim.x):]
	u41k = u31k[(cuda.blockDim.x):]
	u51k = u41k[(cuda.blockDim.x):]
	
	utmp = cuda.local.array(5, numba.float64)

	j = cuda.blockIdx.x + 1
	i = cuda.blockIdx.y + 1
	k = cuda.threadIdx.x

	while k < nz:
		nthreads = nz-(k-cuda.threadIdx.x)
		if nthreads > cuda.blockDim.x:
			nthreads = cuda.blockDim.x
		m = cuda.threadIdx.x
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( j)+ny*( (k-cuda.threadIdx.x)+int(m/5))))]
		cuda.syncthreads()
		# ---------------------------------------------------------------------
		# zeta-direction flux differences
		# ---------------------------------------------------------------------
		flux[cuda.threadIdx.x+(0*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+3]
		u41 = rtmp[cuda.threadIdx.x*5+3]/rtmp[cuda.threadIdx.x*5+0]
		q = ( 0.5*(rtmp[cuda.threadIdx.x*5+1]*rtmp[cuda.threadIdx.x*5+1]+rtmp[cuda.threadIdx.x*5+2]*rtmp[cuda.threadIdx.x*5+2]+rtmp[cuda.threadIdx.x*5+3]*rtmp[cuda.threadIdx.x*5+3])/rtmp[cuda.threadIdx.x*5+0] )
		flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+1]*u41
		flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+2]*u41
		flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+3]*u41+C2*(rtmp[cuda.threadIdx.x*5+4]-q)
		flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = (C1*rtmp[cuda.threadIdx.x*5+4]-C2*q)*u41
		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (k<(nz-1)):
			for m in range(5):
				utmp[m] = frct[(m)+5*(( i)+nx*(( j)+ny*( k)))]-tz2*(flux[(cuda.threadIdx.x+1)+(m*cuda.blockDim.x)]-flux[(cuda.threadIdx.x-1)+(m*cuda.blockDim.x)])
		u41 = 1.0/rtmp[cuda.threadIdx.x*5+0]
		u21k[cuda.threadIdx.x] = u41*rtmp[cuda.threadIdx.x*5+1]
		u31k[cuda.threadIdx.x] = u41*rtmp[cuda.threadIdx.x*5+2]
		u41k[cuda.threadIdx.x] = u41*rtmp[cuda.threadIdx.x*5+3]
		u51k[cuda.threadIdx.x] = u41*rtmp[cuda.threadIdx.x*5+4]
		cuda.syncthreads()
		if cuda.threadIdx.x>=1:
			flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = tz3*(u21k[cuda.threadIdx.x]-u21k[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = tz3*(u31k[cuda.threadIdx.x]-u31k[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = (4.0/3.0)*tz3*(u41k[cuda.threadIdx.x]-u41k[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = ( 0.5*(1.0-C1*C5)*tz3*((u21k[cuda.threadIdx.x]*u21k[cuda.threadIdx.x]+u31k[cuda.threadIdx.x]*u31k[cuda.threadIdx.x]+u41k[cuda.threadIdx.x]*u41k[cuda.threadIdx.x])-(u21k[cuda.threadIdx.x-1]*u21k[cuda.threadIdx.x-1]+u31k[cuda.threadIdx.x-1]*u31k[cuda.threadIdx.x-1]+u41k[cuda.threadIdx.x-1]*u41k[cuda.threadIdx.x-1]))+(1.0/6.0)*tz3*(u41k[cuda.threadIdx.x]*u41k[cuda.threadIdx.x]-u41k[cuda.threadIdx.x-1]*u41k[cuda.threadIdx.x-1])+C1*C5*tz3*(u51k[cuda.threadIdx.x]-u51k[cuda.threadIdx.x-1]) )

		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (k<(nz-1)):
			utmp[0] += dz1*tz1*(rtmp[cuda.threadIdx.x*5-5]-2.0*rtmp[cuda.threadIdx.x*5+0]+rtmp[cuda.threadIdx.x*5+5])
			utmp[1] += ( tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(1*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(1*cuda.blockDim.x)])+dz2*tz1*(rtmp[cuda.threadIdx.x*5-4]-2.0*rtmp[cuda.threadIdx.x*5+1]+rtmp[cuda.threadIdx.x*5+6]) )
			utmp[2] += ( tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(2*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(2*cuda.blockDim.x)])+dz3*tz1*(rtmp[cuda.threadIdx.x*5-3]-2.0*rtmp[cuda.threadIdx.x*5+2]+rtmp[cuda.threadIdx.x*5+7]) )
			utmp[3] += ( tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(3*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(3*cuda.blockDim.x)])+dz4*tz1*(rtmp[cuda.threadIdx.x*5-2]-2.0*rtmp[cuda.threadIdx.x*5+3]+rtmp[cuda.threadIdx.x*5+8]) )
			utmp[4] += ( tz3*C3*C4*(flux[(cuda.threadIdx.x+1)+(4*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(4*cuda.blockDim.x)])+dz5*tz1*(rtmp[cuda.threadIdx.x*5-1]-2.0*rtmp[cuda.threadIdx.x*5+4]+rtmp[cuda.threadIdx.x*5+9]) )
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			if k==1:
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( 1)))] = utmp[m]-dssp*(+5.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( i)+nx*(( j)+ny*( 3)))])
			if k==2:
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( 2)))] = ( utmp[m]-dssp*(-4.0
									*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( i)+nx*(( j)+ny*( 4)))]) )
			if (k>=3) and (k<(nz-3)):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( k)))] = ( utmp[m]-dssp*(rsd[(m)+5*(( i)+nx*(( j)+ny*( k-2)))]
									-4.0*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( i)+nx*(( j)+ny*( k+2)))]) )
			if k==(nz-3):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( nz-3)))] = ( utmp[m]-dssp*(rsd[(m)+5*(( i)+nx*(( j)+ny*( nz-5)))]
										-4.0*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]) )
			if k==(nz-2):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( nz-2)))] = utmp[m]-dssp*(rsd[(m)+5*(( i)+nx*(( j)+ny*( nz-4)))]-4.0*rtmp[cuda.threadIdx.x*5+m-5]+5.0*rtmp[cuda.threadIdx.x*5+m])

		k+=cuda.blockDim.x-2
	#END while k < nz:
#END erhs_gpu_kernel_4


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def erhs_gpu_kernel_3(frct,
					rsd,
					nx,
					ny,
					nz,
					ty1, ty2, ty3,
					dy1, dy2, dy3, dy4, dy5,
					dssp):
	#double* flux = (double*)extern_share_data;
	flux = cuda.shared.array(shape=0, dtype=numba.float64)
	rtmp = flux[(cuda.blockDim.x*5):] 
	u21j = rtmp[(cuda.blockDim.x*5):]
	u31j = u21j[(cuda.blockDim.x):]
	u41j = u31j[(cuda.blockDim.x):]
	u51j = u41j[(cuda.blockDim.x):]

	utmp = cuda.local.array(5, numba.float64)

	k = cuda.blockIdx.x+1
	i = cuda.blockIdx.y+1
	j = cuda.threadIdx.x

	while j < ny:
		nthreads = ny-(j-cuda.threadIdx.x)
		if nthreads>cuda.blockDim.x:
			nthreads = cuda.blockDim.x
		m = cuda.threadIdx.x
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( i)+nx*(( (j-cuda.threadIdx.x)+int(m/5))+ny*( k)))]
		cuda.syncthreads()
		# ---------------------------------------------------------------------
		# eta-direction flux differences
		# ---------------------------------------------------------------------
		flux[cuda.threadIdx.x+(0*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+2]
		u31 = rtmp[cuda.threadIdx.x*5+2]/rtmp[cuda.threadIdx.x*5+0]
		q = ( 0.5*(rtmp[cuda.threadIdx.x*5+1]*rtmp[cuda.threadIdx.x*5+1]+rtmp[cuda.threadIdx.x*5+2]*rtmp[cuda.threadIdx.x*5+2]+rtmp[cuda.threadIdx.x*5+3]*rtmp[cuda.threadIdx.x*5+3])/rtmp[cuda.threadIdx.x*5+0] )
		flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+1]*u31
		flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+2]*u31+C2*(rtmp[cuda.threadIdx.x*5+4]-q)
		flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+3]*u31
		flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = (C1*rtmp[cuda.threadIdx.x*5+4]-C2*q)*u31
		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (j<(ny-1)):
			for m in range(5):
				utmp[m] = frct[(m)+5*(( i)+nx*(( j)+ny*( k)))]-ty2*(flux[(cuda.threadIdx.x+1)+(m*cuda.blockDim.x)]-flux[(cuda.threadIdx.x-1)+(m*cuda.blockDim.x)])
		u31 = 1.0/rtmp[cuda.threadIdx.x*5+0]
		u21j[cuda.threadIdx.x] = u31*rtmp[cuda.threadIdx.x*5+1]
		u31j[cuda.threadIdx.x] = u31*rtmp[cuda.threadIdx.x*5+2]
		u41j[cuda.threadIdx.x] = u31*rtmp[cuda.threadIdx.x*5+3]
		u51j[cuda.threadIdx.x] = u31*rtmp[cuda.threadIdx.x*5+4]
		cuda.syncthreads()
		if cuda.threadIdx.x>=1:
			flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = ty3*(u21j[cuda.threadIdx.x]-u21j[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = (4.0/3.0)*ty3*(u31j[cuda.threadIdx.x]-u31j[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = ty3*(u41j[cuda.threadIdx.x]-u41j[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = ( 0.5*(1.0-C1*C5)*ty3*((u21j[cuda.threadIdx.x]*u21j[cuda.threadIdx.x]+u31j[cuda.threadIdx.x]*u31j[cuda.threadIdx.x]+u41j[cuda.threadIdx.x]*u41j[cuda.threadIdx.x])-(u21j[cuda.threadIdx.x-1]*u21j[cuda.threadIdx.x-1]+u31j[cuda.threadIdx.x-1]*u31j[cuda.threadIdx.x-1]+u41j[cuda.threadIdx.x-1]*u41j[cuda.threadIdx.x-1]))+(1.0/6.0)*ty3*(u31j[cuda.threadIdx.x]*u31j[cuda.threadIdx.x]-u31j[cuda.threadIdx.x-1]*u31j[cuda.threadIdx.x-1])+C1*C5*ty3*(u51j[cuda.threadIdx.x]-u51j[cuda.threadIdx.x-1]) )

		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (j<(ny-1)):
			utmp[0] += dy1*ty1*(rtmp[cuda.threadIdx.x*5-5]-2.0*rtmp[cuda.threadIdx.x*5+0]+rtmp[cuda.threadIdx.x*5+5])
			utmp[1] += ( ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(1*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(1*cuda.blockDim.x)])+dy2*ty1*(rtmp[cuda.threadIdx.x*5-4]-2.0*rtmp[cuda.threadIdx.x*5+1]+rtmp[cuda.threadIdx.x*5+6]) )
			utmp[2] += ( ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(2*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(2*cuda.blockDim.x)])+dy3*ty1*(rtmp[cuda.threadIdx.x*5-3]-2.0*rtmp[cuda.threadIdx.x*5+2]+rtmp[cuda.threadIdx.x*5+7]) )
			utmp[3] += ( ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(3*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(3*cuda.blockDim.x)])+dy4*ty1*(rtmp[cuda.threadIdx.x*5-2]-2.0*rtmp[cuda.threadIdx.x*5+3]+rtmp[cuda.threadIdx.x*5+8]) )
			utmp[4] += ( ty3*C3*C4*(flux[(cuda.threadIdx.x+1)+(4*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(4*cuda.blockDim.x)])+dy5*ty1*(rtmp[cuda.threadIdx.x*5-1]-2.0*rtmp[cuda.threadIdx.x*5+4]+rtmp[cuda.threadIdx.x*5+9]) )
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			if j==1:
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( 1)+ny*( k)))] = utmp[m]-dssp*(+5.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( i)+nx*(( 3)+ny*( k)))])
			if j==2:
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( 2)+ny*( k)))] = ( utmp[m]-dssp*(-4.0
									*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( i)+nx*(( 4)+ny*( k)))]) )
			if (j>=3) and (j<(ny-3)):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( k)))] = ( utmp[m]-dssp*(rsd[(m)+5*(( i)+nx*(( j-2)+ny*( k)))]
									-4.0*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( i)+nx*(( j+2)+ny*( k)))]) )
			if j==(ny-3):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( ny-3)+ny*( k)))] = ( utmp[m]-dssp*(rsd[(m)+5*(( i)+nx*(( ny-5)+ny*( k)))]
									-4.0*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]) )
			if j==(ny-2):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( ny-2)+ny*( k)))] = utmp[m]-dssp*(rsd[(m)+5*(( i)+nx*(( ny-4)+ny*( k)))]-4.0*rtmp[cuda.threadIdx.x*5+m-5]+5.0*rtmp[cuda.threadIdx.x*5+m])

		j += cuda.blockDim.x-2
	#END while j < ny:
#END erhs_gpu_kernel_3()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def erhs_gpu_kernel_2(frct,
					rsd,
					nx,
					ny,
					nz,
					tx1, tx2, tx3,
					dx1, dx2, dx3, dx4, dx5,
					dssp):
	#double* flux = (double*)extern_share_data;
	flux = cuda.shared.array(shape=0, dtype=numba.float64)
	rtmp = flux[(cuda.blockDim.x*5):] 
	u21i = rtmp[(cuda.blockDim.x*5):]
	u31i = u21i[(cuda.blockDim.x):]
	u41i = u31i[(cuda.blockDim.x):]
	u51i = u41i[(cuda.blockDim.x):]

	utmp = cuda.local.array(5, numba.float64)

	k = cuda.blockIdx.x + 1
	j = cuda.blockIdx.y + 1
	i = cuda.threadIdx.x

	while i < nx:
		nthreads = nx-(i-cuda.threadIdx.x)
		if nthreads > cuda.blockDim.x:
			nthreads = cuda.blockDim.x
		m = cuda.threadIdx.x
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		m += nthreads
		rtmp[m] = rsd[(m%5)+5*(( (i-cuda.threadIdx.x)+int(m/5))+nx*(( j)+ny*( k)))]
		cuda.syncthreads()
		# ---------------------------------------------------------------------
		# xi-direction flux differences
		# ---------------------------------------------------------------------
		flux[cuda.threadIdx.x+(0*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+1]
		u21 = rtmp[cuda.threadIdx.x*5+1]/rtmp[cuda.threadIdx.x*5+0]
		q = ( 0.5*(rtmp[cuda.threadIdx.x*5+1]*rtmp[cuda.threadIdx.x*5+1]+rtmp[cuda.threadIdx.x*5+2]*rtmp[cuda.threadIdx.x*5+2]+rtmp[cuda.threadIdx.x*5+3]*rtmp[cuda.threadIdx.x*5+3])/rtmp[cuda.threadIdx.x*5+0] )
		flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+1]*u21+C2*(rtmp[cuda.threadIdx.x*5+4]-q)
		flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+2]*u21
		flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = rtmp[cuda.threadIdx.x*5+3]*u21
		flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = (C1*rtmp[cuda.threadIdx.x*5+4]-C2*q)*u21
		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (i<(nx-1)):
			for m in range(5):
				utmp[m] = frct[(m)+5*(( i)+nx*(( j)+ny*( k)))]-tx2*(flux[(cuda.threadIdx.x+1)+(m*cuda.blockDim.x)]-flux[(cuda.threadIdx.x-1)+(m*cuda.blockDim.x)])
		u21 = 1.0/rtmp[cuda.threadIdx.x*5+0]
		u21i[cuda.threadIdx.x] = u21*rtmp[cuda.threadIdx.x*5+1]
		u31i[cuda.threadIdx.x] = u21*rtmp[cuda.threadIdx.x*5+2]
		u41i[cuda.threadIdx.x] = u21*rtmp[cuda.threadIdx.x*5+3]
		u51i[cuda.threadIdx.x] = u21*rtmp[cuda.threadIdx.x*5+4]
		cuda.syncthreads()
		if cuda.threadIdx.x>=1:
			flux[cuda.threadIdx.x+(1*cuda.blockDim.x)] = (4.0/3.0)*tx3*(u21i[cuda.threadIdx.x]-u21i[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(2*cuda.blockDim.x)] = tx3*(u31i[cuda.threadIdx.x]-u31i[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(3*cuda.blockDim.x)] = tx3*(u41i[cuda.threadIdx.x]-u41i[cuda.threadIdx.x-1])
			flux[cuda.threadIdx.x+(4*cuda.blockDim.x)] = ( 0.5*(1.0-C1*C5)*tx3*((u21i[cuda.threadIdx.x]*u21i[cuda.threadIdx.x]+u31i[cuda.threadIdx.x]*u31i[cuda.threadIdx.x]+u41i[cuda.threadIdx.x]*u41i[cuda.threadIdx.x])-(u21i[cuda.threadIdx.x-1]*u21i[cuda.threadIdx.x-1]+u31i[cuda.threadIdx.x-1]*u31i[cuda.threadIdx.x-1]+u41i[cuda.threadIdx.x-1]*u41i[cuda.threadIdx.x-1]))+(1.0/6.0)*tx3*(u21i[cuda.threadIdx.x]*u21i[cuda.threadIdx.x]-u21i[cuda.threadIdx.x-1]*u21i[cuda.threadIdx.x-1])+C1*C5*tx3*(u51i[cuda.threadIdx.x]-u51i[cuda.threadIdx.x-1]) )

		cuda.syncthreads()
		if (cuda.threadIdx.x>=1) and (cuda.threadIdx.x<(cuda.blockDim.x-1)) and (i<nx-1):
			utmp[0] += dx1*tx1*(rtmp[cuda.threadIdx.x*5-5]-2.0*rtmp[cuda.threadIdx.x*5+0]+rtmp[cuda.threadIdx.x*5+5])
			utmp[1] += ( tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(1*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(1*cuda.blockDim.x)])+dx2*tx1*(rtmp[cuda.threadIdx.x*5-4]-2.0*rtmp[cuda.threadIdx.x*5+1]+rtmp[cuda.threadIdx.x*5+6]) )
			utmp[2] += ( tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(2*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(2*cuda.blockDim.x)])+dx3*tx1*(rtmp[cuda.threadIdx.x*5-3]-2.0*rtmp[cuda.threadIdx.x*5+2]+rtmp[cuda.threadIdx.x*5+7]) )
			utmp[3] += ( tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(3*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(3*cuda.blockDim.x)])+dx4*tx1*(rtmp[cuda.threadIdx.x*5-2]-2.0*rtmp[cuda.threadIdx.x*5+3]+rtmp[cuda.threadIdx.x*5+8]) )
			utmp[4] += ( tx3*C3*C4*(flux[(cuda.threadIdx.x+1)+(4*cuda.blockDim.x)]-flux[cuda.threadIdx.x+(4*cuda.blockDim.x)])+dx5*tx1*(rtmp[cuda.threadIdx.x*5-1]-2.0*rtmp[cuda.threadIdx.x*5+4]+rtmp[cuda.threadIdx.x*5+9]) )
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			if i==1:
				for m in range(5):
					frct[(m)+5*(( 1)+nx*(( j)+ny*( k)))] = ( utmp[m]-dssp*(+5.0
									*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( 3)+nx*(( j)+ny*( k)))]) )
			if i==2:
				for m in range(5):
					frct[(m)+5*(( 2)+nx*(( j)+ny*( k)))] = ( utmp[m]-dssp*(-4.0
									*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*(( 4)+nx*(( j)+ny*( k)))]) )
			if (i>=3) and (i<(nx-3)):
				for m in range(5):
					frct[(m)+5*(( i)+nx*(( j)+ny*( k)))] = ( utmp[m]-dssp*(rsd[(m)+5*(( i-2)+nx*(( j)+ny*( k)))]
									-4.0*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]+rsd[(m)+5*((i+2)+nx*((j)+ny*(k)))]) )
			if i==(nx-3):
				for m in range(5):
					frct[(m)+5*(( nx-3)+nx*(( j)+ny*( k)))] = ( utmp[m]-dssp*(rsd[(m)+5*(( nx-5)+nx*(( j)+ny*( k)))]
									-4.0*rtmp[cuda.threadIdx.x*5+m-5]+6.0*rtmp[cuda.threadIdx.x*5+m]-4.0*rtmp[cuda.threadIdx.x*5+m+5]) )
			if i==(nx-2):
				for m in range(5):
					frct[(m)+5*(( nx-2)+nx*(( j)+ny*( k)))] = utmp[m]-dssp*(rsd[(m)+5*(( nx-4)+nx*(( j)+ny*( k)))]-4.0*rtmp[cuda.threadIdx.x*5+m-5]+5.0*rtmp[cuda.threadIdx.x*5+m]) 

		i += cuda.blockDim.x-2
	#END while i < nx:
#END erhs_gpu_kernel_2()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64[:, :])')
def erhs_gpu_kernel_1(frct,
					rsd,
					nx,
					ny,
					nz,
					ce):
	i_j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j_k % nx
	j = int(i_j_k / nx) % ny
	k = int(i_j_k / (nx * ny))

	if i_j_k >= (nx*ny*nz):
		return

	for m in range(5):
		frct[(m)+5*(( i)+nx*(( j)+ny*( k)))] = 0.0
	zeta = k / (nz-1)
	eta = j / (ny-1)
	xi = i / (nx-1)
	for m in range(5):
		rsd[(m)+5*(( i)+nx*(( j)+ny*( k)))] = ( ce[0][m]+
		(ce[1][m]+
		 (ce[4][m]+
		  (ce[7][m]+
		   ce[10][m]*xi)*xi)*xi)*xi+
			(ce[2][m]+
			 (ce[5][m]+
			  (ce[8][m]+
			   ce[11][m]*eta)*eta)*eta)*eta+
			(ce[3][m]+
			 (ce[6][m]+
			  (ce[9][m]+
			   ce[12][m]*zeta)*zeta)*zeta)*zeta )
#END erhs_gpu_kernel_1()


# ---------------------------------------------------------------------
# compute the right hand side based on exact solution
# ---------------------------------------------------------------------
def erhs_gpu(frct_device,
			 rsd_device,
			 ce_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERHS_1)
	# #KERNEL ERHS 1
	erhs_1_workload = nx * ny * nz
	erhs_1_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_1
	erhs_1_blocks_per_grid = math.ceil(erhs_1_workload / erhs_1_threads_per_block)

	erhs_gpu_kernel_1[erhs_1_blocks_per_grid, 
					erhs_1_threads_per_block](frct_device, 
											rsd_device, 
											nx, 
											ny, 
											nz,
											ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERHS_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERHS_2)
	# #KERNEL ERHS 2
	erhs_2_blocks_per_grid = (nz-2, ny-2)
	erhs_2_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_2
	if THREADS_PER_BLOCK_ON_ERHS_2 != device_prop.WARP_SIZE:
		erhs_2_threads_per_block = device_prop.WARP_SIZE
	size_shared_data = rsd_device.dtype.itemsize * ( (2*(min(nx,erhs_2_threads_per_block))*5) + (4*(min(nx,erhs_2_threads_per_block))) )
	#print("threadSize=[%d]" % (erhs_2_threads_per_block))
	#print("blockSize=[%d, %d]" % (erhs_2_blocks_per_grid[0], erhs_2_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	erhs_gpu_kernel_2[erhs_2_blocks_per_grid, 
					min(nx,erhs_2_threads_per_block),
					stream,
					size_shared_data](frct_device, 
									rsd_device, 
									nx, 
									ny, 
									nz,
									tx1, tx2, tx3,
									dx1, dx2, dx3, dx4, dx5,
									dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERHS_2)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERHS_3)
	# #KERNEL ERHS 3
	erhs_3_blocks_per_grid = (nz-2, nx-2)
	erhs_3_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_3
	if THREADS_PER_BLOCK_ON_ERHS_3 != device_prop.WARP_SIZE:
		erhs_3_threads_per_block = device_prop.WARP_SIZE
	size_shared_data = rsd_device.dtype.itemsize * ( (2*(min(ny,erhs_3_threads_per_block))*5) + (4*(min(ny,erhs_3_threads_per_block))) )
	#print("threadSize=[%d]" % (erhs_3_threads_per_block))
	#print("blockSize=[%d, %d]" % (erhs_3_blocks_per_grid[0], erhs_3_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))
	
	erhs_gpu_kernel_3[erhs_3_blocks_per_grid, 
					min(ny, erhs_3_threads_per_block),
					stream,
					size_shared_data](frct_device, 
									rsd_device, 
									nx, 
									ny, 
									nz,
									ty1, ty2, ty3,
									dy1, dy2, dy3, dy4, dy5,
									dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERHS_3)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERHS_4)
	# #KERNEL ERHS 4
	erhs_4_blocks_per_grid = (ny-2, nx-2)
	erhs_4_threads_per_block = THREADS_PER_BLOCK_ON_ERHS_4
	if THREADS_PER_BLOCK_ON_ERHS_4 != device_prop.WARP_SIZE:
		erhs_4_threads_per_block = device_prop.WARP_SIZE
	size_shared_data = rsd_device.dtype.itemsize * ( (2*(min(nz,erhs_4_threads_per_block))*5) + (4*(min(nz,erhs_4_threads_per_block))) )
	#print("threadSize=[%d]" % (erhs_4_threads_per_block))
	#print("blockSize=[%d, %d]" % (erhs_4_blocks_per_grid[0], erhs_4_blocks_per_grid[1]))
	#print("sharedMemory=%d" % (size_shared_data))

	erhs_gpu_kernel_4[erhs_4_blocks_per_grid, 
					min(nz,erhs_4_threads_per_block),
					stream,
					size_shared_data](frct_device, 
									rsd_device, 
									nx, 
									ny, 
									nz,
									tz1, tz2, tz3,
									dz1, dz2, dz3, dz4, dz5,
									dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERHS_4)
##END erhs_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:, :])')
def setiv_gpu_kernel(u,
					nx,
					ny,
					nz,
					ce):
	ue_1jk = cuda.local.array(5, numba.float64)
	ue_nx0jk = cuda.local.array(5, numba.float64)
	ue_i1k = cuda.local.array(5, numba.float64)
	ue_iny0k = cuda.local.array(5, numba.float64)
	ue_ij1 = cuda.local.array(5, numba.float64)
	ue_ijnz = cuda.local.array(5, numba.float64)

	k = cuda.blockIdx.x + 1
	j = cuda.blockIdx.y + 1
	i = cuda.threadIdx.x + 1

	zeta = k / (nz-1)
	eta = j / (ny-1)
	xi = i / (nx-1)
	exact_gpu_device(0, j, k, ue_1jk, nx, ny, nz, ce)
	exact_gpu_device(nx-1, j, k, ue_nx0jk, nx, ny, nz, ce)
	exact_gpu_device(i, 0, k, ue_i1k, nx, ny, nz, ce)
	exact_gpu_device(i, ny-1, k, ue_iny0k, nx, ny, nz, ce)
	exact_gpu_device(i, j, 0, ue_ij1, nx, ny, nz, ce)
	exact_gpu_device(i, j, nz-1, ue_ijnz, nx, ny, nz, ce)
	for m in range(5):
		pxi = (1.0-xi)*ue_1jk[m]+xi*ue_nx0jk[m]
		peta = (1.0-eta)*ue_i1k[m]+eta*ue_iny0k[m]
		pzeta = (1.0-zeta)*ue_ij1[m]+zeta*ue_ijnz[m]
		u[(m)+5*(( i)+nx*(( j)+ny*( k)))] = pxi+peta+pzeta-pxi*peta-peta*pzeta-pzeta*pxi+pxi*peta*pzeta
#END setiv_gpu_kernel()


# ---------------------------------------------------------------------
# set the initial values of independent variables based on tri-linear
# interpolation of boundary values in the computational space.
# ---------------------------------------------------------------------
def setiv_gpu(u_device, 
			ce_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_SETIV)
	# #KERNEL SETIV
	setiv_blocks_per_grid = (nz-2, ny-2)
	setiv_threads_per_block = THREADS_PER_BLOCK_ON_SETIV
	if THREADS_PER_BLOCK_ON_SETIV != nx-2:
		setiv_threads_per_block = nx-2

	setiv_gpu_kernel[setiv_blocks_per_grid, 
					setiv_threads_per_block](u_device, 
											nx, 
											ny, 
											nz,
											ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_SETIV)
#END setiv_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:, :])')
def setbv_gpu_kernel_1(u,
					nx,
					ny,
					nz,
					ce):
	temp1 = cuda.local.array(5, numba.float64)
	temp2 = cuda.local.array(5, numba.float64)

	j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	j = j_k % ny
	k = int(j_k / ny) % nz

	if j_k >= (nz*ny):
		return

	# ---------------------------------------------------------------------
	# set the dependent variable values along east and west faces
	# ---------------------------------------------------------------------
	exact_gpu_device(0, 
			j, 
			k, 
			temp1, 
			nx, 
			ny, 
			nz,
			ce)
	exact_gpu_device(nx-1, 
			j, 
			k, 
			temp2, 
			nx, 
			ny, 
			nz,
			ce)
	for m in range(5):
		u[(m)+5*(( 0)+nx*(( j)+ny*( k)))] = temp1[m]
		u[(m)+5*(( nx-1)+nx*(( j)+ny*( k)))] = temp2[m]
#END setbv_gpu_kernel_1()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:, :])')
def setbv_gpu_kernel_2(u,
					nx,
					ny,
					nz,
					ce):
	temp1 = cuda.local.array(5, numba.float64)
	temp2 = cuda.local.array(5, numba.float64)

	i_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_k % nx
	k = int(i_k / nx) % nz

	if i_k >= (nx*nz):
		return

	# ---------------------------------------------------------------------
	# set the dependent variable values along north and south faces
	# ---------------------------------------------------------------------
	exact_gpu_device(i, 
			0, 
			k, 
			temp1, 
			nx, 
			ny, 
			nz,
			ce)
	exact_gpu_device(i, 
			ny-1, 
			k, 
			temp2, 
			nx, 
			ny, 
			nz,
			ce)
	for m in range(5):
		u[(m)+5*(( i)+nx*(( 0)+ny*( k)))]    = temp1[m]
		u[(m)+5*(( i)+nx*(( ny-1)+ny*( k)))] = temp2[m]
#END setbv_gpu_kernel_2()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:, :])')
def setbv_gpu_kernel_3(u,
					nx,
					ny,
					nz,
					ce):
	temp1 = cuda.local.array(5, numba.float64)
	temp2 = cuda.local.array(5, numba.float64)

	i_j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j % nx
	j = int(i_j / nx) % ny

	if i_j >= (nx*ny):
		return

	# ---------------------------------------------------------------------
	# set the dependent variable values along the top and bottom faces
	# ---------------------------------------------------------------------
	exact_gpu_device(i, 
			j, 
			0, 
			temp1, 
			nx, 
			ny, 
			nz,
			ce)
	exact_gpu_device(i, 
			j, 
			nz-1, 
			temp2, 
			nx, 
			ny, 
			nz,
			ce)
	for m in range(5):
		u[(m)+5*(( i)+nx*(( j)+ny*( 0)))]    = temp1[m]
		u[(m)+5*(( i)+nx*(( j)+ny*( nz-1)))] = temp2[m]
#END setbv_gpu_kernel_3()



# ---------------------------------------------------------------------
# set the boundary values of dependent variables
# ---------------------------------------------------------------------
def setbv_gpu(u_device,
			  ce_device):
	m_ceil = math.ceil
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_SETBV_3)
	# #KERNEL SETBV 3
	setbv_3_workload = nx * ny
	setbv_3_threads_per_block = THREADS_PER_BLOCK_ON_SETBV_3
	setbv_3_blocks_per_grid = m_ceil(setbv_3_workload/setbv_3_threads_per_block)

	setbv_gpu_kernel_3[setbv_3_blocks_per_grid, 
					setbv_3_threads_per_block](u_device, 
											nx, 
											ny, 
											nz,
											ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_SETBV_3)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_SETBV_2)
	# #KERNEL SETBV 2
	setbv_2_workload = nx * nz
	setbv_2_threads_per_block = THREADS_PER_BLOCK_ON_SETBV_2
	setbv_2_blocks_per_grid = m_ceil(setbv_2_workload/setbv_2_threads_per_block)

	setbv_gpu_kernel_2[setbv_2_blocks_per_grid, 
					setbv_2_threads_per_block](u_device, 
											nx, 
											ny, 
											nz,
											ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_SETBV_2)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_SETBV_1)
	# #KERNEL SETBV 1
	setbv_1_workload = ny * nz
	setbv_1_threads_per_block = THREADS_PER_BLOCK_ON_SETBV_1
	setbv_1_blocks_per_grid = m_ceil(setbv_1_workload/setbv_1_threads_per_block)

	setbv_gpu_kernel_1[setbv_1_blocks_per_grid, 
					setbv_1_threads_per_block](u_device, 
											nx, 
											ny, 
											nz,
											ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_SETBV_1)
#END setbv_gpu()


#*****************************************************************
#************************* CPU FUNCTIONS *************************
#*****************************************************************

# ---------------------------------------------------------------------
# verification routine                         
# ---------------------------------------------------------------------
def verify_gpu(xcr, 
			xce,
			xci):
	# ---------------------------------------------------------------------
	# tolerance level
	# ---------------------------------------------------------------------
	epsilon = 1.0e-08
	dtref = 0.0
	verified = True
	xcrref = numpy.repeat(1.0, 5)
	xceref = numpy.repeat(1.0, 5)
	xciref = 1.0
	
	if npbparams.CLASS == 'S':
		dtref = 5.0e-1
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (12X12X12) grid,
		# after 50 time steps, with DT = 5.0d-01
		# ---------------------------------------------------------------------
		xcrref[0] = 1.6196343210976702e-02
		xcrref[1] = 2.1976745164821318e-03
		xcrref[2] = 1.5179927653399185e-03
		xcrref[3] = 1.5029584435994323e-03
		xcrref[4] = 3.4264073155896461e-02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (12X12X12) grid,
		# after 50 time steps, with DT = 5.0d-01
		# ---------------------------------------------------------------------
		xceref[0] = 6.4223319957960924e-04
		xceref[1] = 8.4144342047347926e-05
		xceref[2] = 5.8588269616485186e-05
		xceref[3] = 5.8474222595157350e-05
		xceref[4] = 1.3103347914111294e-03
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (12X12X12) grid,
		# after 50 time steps, with DT = 5.0d-01
		# ---------------------------------------------------------------------
		xciref = 7.8418928865937083e+00
	elif npbparams.CLASS == 'W':
		dtref = 1.5e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (33x33x33) grid,
		# after 300 time steps, with DT = 1.5d-3
		# ---------------------------------------------------------------------
		xcrref[0] = 0.1236511638192e+02
		xcrref[1] = 0.1317228477799e+01
		xcrref[2] = 0.2550120713095e+01
		xcrref[3] = 0.2326187750252e+01
		xcrref[4] = 0.2826799444189e+02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (33X33X33) grid,
		# ---------------------------------------------------------------------
		xceref[0] = 0.4867877144216e+00
		xceref[1] = 0.5064652880982e-01
		xceref[2] = 0.9281818101960e-01
		xceref[3] = 0.8570126542733e-01
		xceref[4] = 0.1084277417792e+01
		# ---------------------------------------------------------------------
		# rReference value of surface integral, for the (33X33X33) grid,
		# after 300 time steps, with DT = 1.5d-3
		# ---------------------------------------------------------------------
		xciref = 0.1161399311023e+02
	elif npbparams.CLASS == 'A':
		dtref = 2.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (64X64X64) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 7.7902107606689367e+02
		xcrref[1] = 6.3402765259692870e+01
		xcrref[2] = 1.9499249727292479e+02
		xcrref[3] = 1.7845301160418537e+02
		xcrref[4] = 1.8384760349464247e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (64X64X64) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 2.9964085685471943e+01
		xceref[1] = 2.8194576365003349e+00
		xceref[2] = 7.3473412698774742e+00
		xceref[3] = 6.7139225687777051e+00
		xceref[4] = 7.0715315688392578e+01
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (64X64X64) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xciref = 2.6030925604886277e+01
	elif npbparams.CLASS == 'B':
		dtref = 2.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (102X102X102) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 3.5532672969982736e+03
		xcrref[1] = 2.6214750795310692e+02
		xcrref[2] = 8.8333721850952190e+02
		xcrref[3] = 7.7812774739425265e+02
		xcrref[4] = 7.3087969592545314e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (102X102X102) 
		# grid, after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 1.1401176380212709e+02
		xceref[1] = 8.1098963655421574e+00
		xceref[2] = 2.8480597317698308e+01
		xceref[3] = 2.5905394567832939e+01
		xceref[4] = 2.6054907504857413e+02
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (102X102X102) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xciref = 4.7887162703308227e+01
	elif npbparams.CLASS == 'C':
		dtref = 2.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (162X162X162) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 1.03766980323537846e+04
		xcrref[1] = 8.92212458801008552e+02
		xcrref[2] = 2.56238814582660871e+03
		xcrref[3] = 2.19194343857831427e+03
		xcrref[4] = 1.78078057261061185e+04
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (162X162X162) 
		# grid, after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 2.15986399716949279e+02
		xceref[1] = 1.55789559239863600e+01
		xceref[2] = 5.41318863077207766e+01
		xceref[3] = 4.82262643154045421e+01
		xceref[4] = 4.55902910043250358e+02
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (162X162X162) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xciref = 6.66404553572181300e+01
	elif npbparams.CLASS == 'D':
		dtref = 1.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (408X408X408) grid,
		# after 300 time steps, with DT = 1.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 0.4868417937025e+05
		xcrref[1] = 0.4696371050071e+04
		xcrref[2] = 0.1218114549776e+05
		xcrref[3] = 0.1033801493461e+05
		xcrref[4] = 0.7142398413817e+05
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (408X408X408) 
		# grid, after 300 time steps, with DT = 1.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 0.3752393004482e+03
		xceref[1] = 0.3084128893659e+02
		xceref[2] = 0.9434276905469e+02
		xceref[3] = 0.8230686681928e+02
		xceref[4] = 0.7002620636210e+03
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (408X408X408) grid,
		# after 300 time steps, with DT = 1.0d+00
		# ---------------------------------------------------------------------
		xciref = 0.8334101392503e+02
	elif npbparams.CLASS == 'E':
		dtref = 0.5e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (1020X1020X1020) grid,
		# after 300 time steps, with DT = 0.5d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 0.2099641687874e+06
		xcrref[1] = 0.2130403143165e+05
		xcrref[2] = 0.5319228789371e+05
		xcrref[3] = 0.4509761639833e+05
		xcrref[4] = 0.2932360006590e+06
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (1020X1020X1020) 
		# grid, after 300 time steps, with DT = 0.5d+00
		# ---------------------------------------------------------------------
		xceref[0] = 0.4800572578333e+03
		xceref[1] = 0.4221993400184e+02
		xceref[2] = 0.1210851906824e+03
		xceref[3] = 0.1047888986770e+03
		xceref[4] = 0.8363028257389e+03
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (1020X1020X1020) grid,
		# after 300 time steps, with DT = 0.5d+00
		# ---------------------------------------------------------------------
		xciref = 0.9512163272273e+02
	else:
		verified = False
		
	# ---------------------------------------------------------------------
	# verification test for residuals if gridsize is one of 
	# the defined grid sizes above (class .ne. 'U')
	# ---------------------------------------------------------------------
	# compute the difference of solution values and the known reference values.
	# ---------------------------------------------------------------------
	xcrdif = numpy.empty(5, dtype=numpy.float64)
	xcedif = numpy.empty(5, dtype=numpy.float64)
	for m in range(5):
		xcrdif[m] = abs((xcr[m]-xcrref[m]) / xcrref[m])
		xcedif[m] = abs((xce[m]-xceref[m]) / xceref[m])

	xcidif = abs((xci-xciref)/xciref)
	# ---------------------------------------------------------------------
	# output the comparison of computed results to known cases.
	# ---------------------------------------------------------------------
	print("\n Verification being performed for class_npb %c" % (npbparams.CLASS))
	print(" accuracy setting for epsilon = %20.13E" % (epsilon))
	verified = abs(dt_host-dtref) <= epsilon
	if not verified:
		print(" DT does not match the reference value of %15.8E" % (dtref))
	
	print(" Comparison of RMS-norms of residual")
	for m in range(5):
		if xcrdif[m] <= epsilon:
			print("          %2d  %20.13E%20.13E%20.13E" % (m+1, xcr[m], xcrref[m], xcrdif[m]))
		else:
			verified = False
			print(" FAILURE: %2d  %20.13E%20.13E%20.13E" % (m+1, xcr[m], xcrref[m], xcrdif[m]))
	
	print(" Comparison of RMS-norms of solution error")
	for m in range(5):
		if xcedif[m] <= epsilon:
			print("          %2d  %20.13E%20.13E%20.13E" % (m+1, xce[m], xceref[m], xcedif[m]))
		else:
			verified = False
			print(" FAILURE: %2d  %20.13E%20.13E%20.13E" % (m+1, xce[m], xceref[m], xcedif[m]))
	
	print(" Comparison of surface integral")
	if xcidif <= epsilon:
		print("              %20.13E%20.13E%20.13E" % (xci, xciref, xcidif))
	else:
		verified = False
		print(" FAILURE:     %20.13E%20.13E%20.13E" % (xci, xciref, xcidif))

	if verified:
		print(" Verification Successful")
	else:
		print(" Verification failed")
	
	return verified
#END verify_gpu()


def setcoeff_gpu():
	global dxi, deta, dzeta
	global tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3
	global dx1, dx2, dx3, dx4, dx5
	global dy1, dy2, dy3, dy4, dy5
	global dz1, dz2, dz3, dz4, dz5
	global dssp
	global ce, ce_device
	global const_jac_device
	
	dt = dt_host
	omega = omega_host
	
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	# set up coefficients
	# ---------------------------------------------------------------------
	dxi = 1.0/(nx-1)
	deta = 1.0/(ny-1)
	dzeta = 1.0/(nz-1)
	tx1 = 1.0/(dxi*dxi)
	tx2 = 1.0/(2.0*dxi)
	tx3 = 1.0/dxi
	ty1 = 1.0/(deta*deta)
	ty2 = 1.0/(2.0*deta)
	ty3 = 1.0/deta
	tz1 = 1.0/(dzeta*dzeta)
	tz2 = 1.0/(2.0*dzeta)
	tz3 = 1.0/dzeta
	# ---------------------------------------------------------------------
	# diffusion coefficients
	# ---------------------------------------------------------------------
	dx1 = 0.75
	dx2 = dx1
	dx3 = dx1
	dx4 = dx1
	dx5 = dx1
	dy1 = 0.75
	dy2 = dy1
	dy3 = dy1
	dy4 = dy1
	dy5 = dy1
	dz1 = 1.00
	dz2 = dz1
	dz3 = dz1
	dz4 = dz1
	dz5 = dz1
	# ---------------------------------------------------------------------
	# fourth difference dissipation
	# ---------------------------------------------------------------------
	dssp = (max(max(dx1,dy1),dz1))/4.0
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the first pde
	# ---------------------------------------------------------------------
	ce[0][0] = 2.0
	ce[1][0] = 0.0
	ce[2][0] = 0.0
	ce[3][0] = 4.0
	ce[4][0] = 5.0
	ce[5][0] = 3.0
	ce[6][0] = 0.5
	ce[7][0] = 0.02
	ce[8][0] = 0.01
	ce[9][0] = 0.03
	ce[10][0] = 0.5
	ce[11][0] = 0.4
	ce[12][0] = 0.3
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the second pde
	# ---------------------------------------------------------------------
	ce[0][1] = 1.0
	ce[1][1] = 0.0
	ce[2][1] = 0.0
	ce[3][1] = 0.0
	ce[4][1] = 1.0
	ce[5][1] = 2.0
	ce[6][1] = 3.0
	ce[7][1] = 0.01
	ce[8][1] = 0.03
	ce[9][1] = 0.02
	ce[10][1] = 0.4
	ce[11][1] = 0.3
	ce[12][1] = 0.5
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the third pde
	# ---------------------------------------------------------------------
	ce[0][2] = 2.0
	ce[1][2] = 2.0
	ce[2][2] = 0.0
	ce[3][2] = 0.0
	ce[4][2] = 0.0
	ce[5][2] = 2.0
	ce[6][2] = 3.0
	ce[7][2] = 0.04
	ce[8][2] = 0.03
	ce[9][2] = 0.05
	ce[10][2] = 0.3
	ce[11][2] = 0.5
	ce[12][2] = 0.4
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the fourth pde
	# ---------------------------------------------------------------------
	ce[0][3] = 2.0
	ce[1][3] = 2.0
	ce[2][3] = 0.0
	ce[3][3] = 0.0
	ce[4][3] = 0.0
	ce[5][3] = 2.0
	ce[6][3] = 3.0
	ce[7][3] = 0.03
	ce[8][3] = 0.05
	ce[9][3] = 0.04
	ce[10][3] = 0.2
	ce[11][3] = 0.1
	ce[12][3] = 0.3
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the fifth pde
	# ---------------------------------------------------------------------
	ce[0][4] = 5.0
	ce[1][4] = 4.0
	ce[2][4] = 3.0
	ce[3][4] = 2.0
	ce[4][4] = 0.1
	ce[5][4] = 0.4
	ce[6][4] = 0.3
	ce[7][4] = 0.05
	ce[8][4] = 0.04
	ce[9][4] = 0.03
	ce[10][4] = 0.1
	ce[11][4] = 0.3
	ce[12][4] = 0.2
	
	#Constant arrays to GPU memory
	ce_device = cuda.to_device(ce)
	
	const_jac = numpy.array([dt, omega, tx1, tx2, ty1, ty2, tz1, tz2, dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, dy5, dz1, dz2, dz3, dz4, dz5], numpy.float64)
	const_jac_device = cuda.to_device(const_jac)
	
	#Another constant values are going to be passed to kernels as parameters
#END setcoeff()


def read_input():
	# ---------------------------------------------------------------------
	# if input file does not exist, it uses defaults
	# ipr = 1 for detailed progress output
	# inorm = how often the norm is printed (once every inorm iterations)
	# itmax = number of pseudo time steps
	# dt = time step
	# omega 1 over-relaxation factor for SSOR
	# tolrsd = steady state residual tolerance levels
	# nx, ny, nz = number of grid points in x, y, z directions
	# ---------------------------------------------------------------------
	global ipr, inorm, itmax
	global dt_host, omega_host
	global tolrsd
	global nx, ny, nz
	
	fp = os.path.isfile("inputlu.data")
	if fp:
		print(" Reading from input file inputlu.data") 
		print(" ERROR - Not implemented") 
		sys.exit()
	else:
		ipr = IPR_DEFAULT
		inorm = npbparams.INORM_DEFAULT
		itmax = npbparams.ITMAX_DEFAULT
		dt_host = npbparams.DT_DEFAULT
		omega_host = OMEGA_DEFAULT
		tolrsd[0] = TOLRSD1_DEF
		tolrsd[1] = TOLRSD2_DEF
		tolrsd[2] = TOLRSD3_DEF
		tolrsd[3] = TOLRSD4_DEF
		tolrsd[4] = TOLRSD5_DEF
		nx = npbparams.ISIZ1
		ny = npbparams.ISIZ2
		nz = npbparams.ISIZ3
		
	# ---------------------------------------------------------------------
	# check problem size
	# ---------------------------------------------------------------------
	if (nx<4) or (ny<4) or (nz<4):
		print("     PROBLEM SIZE IS TOO SMALL - \n"
				"     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5")
		sys.exit()
	
	if (nx>npbparams.ISIZ1) or (ny>npbparams.ISIZ2) or (nz>npbparams.ISIZ3):
		print("     PROBLEM SIZE IS TOO LARGE - \n"
				"     NX, NY AND NZ SHOULD BE EQUAL TO \n"
				"     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY")
		sys.exit()
		
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - LU Benchmark\n")
	print(" Size: %4dx%4dx%4d" % (nx, ny, nz))
	print(" Iterations: %4d" % (itmax))
	print()
#END read_input()


def main():
	global u_device, rsd_device, frct_device
	global rho_i_device, qs_device
	global ce_device
	global norm_buffer_device
	global rsdnm, errnm
	global const_jac_device
	global frc
	
	# ---------------------------------------------------------------------
	# read input data
	# ---------------------------------------------------------------------
	read_input()
	# ---------------------------------------------------------------------
	# set up coefficients
	# ---------------------------------------------------------------------
	setup_gpu()
	setcoeff_gpu()
	# ---------------------------------------------------------------------
	# set the boundary values for dependent variables
	# ---------------------------------------------------------------------
	setbv_gpu(u_device, ce_device)
	# ---------------------------------------------------------------------
	# set the initial values for dependent variables
	# ---------------------------------------------------------------------
	setiv_gpu(u_device, ce_device)
	# ---------------------------------------------------------------------
	# compute the forcing term based on prescribed exact solution
	# ---------------------------------------------------------------------
	erhs_gpu(frct_device, rsd_device, ce_device)
	# ---------------------------------------------------------------------
	# perform one SSOR iteration to touch all pages
	# ---------------------------------------------------------------------
	ssor_gpu(1, 
			u_device, rsd_device, frct_device,
			rho_i_device, qs_device,
			norm_buffer_device, rsdnm,
			const_jac_device)
	# ---------------------------------------------------------------------
	# reset the boundary and initial values
	# ---------------------------------------------------------------------
	setbv_gpu(u_device, ce_device)
	setiv_gpu(u_device, ce_device)
	# ---------------------------------------------------------------------
	# perform the SSOR iterations
	# ---------------------------------------------------------------------
	ssor_gpu(itmax,
			u_device, rsd_device, frct_device,
			rho_i_device, qs_device,
			norm_buffer_device, rsdnm,
			const_jac_device)
	# ---------------------------------------------------------------------
	# compute the solution error
	# ---------------------------------------------------------------------
	error_gpu(u_device, norm_buffer_device, errnm, ce_device)
	# ---------------------------------------------------------------------
	# compute the surface integral
	# ---------------------------------------------------------------------
	frc = pintgr_gpu(u_device, norm_buffer_device)
	# ---------------------------------------------------------------------
	# verification test
	# ---------------------------------------------------------------------
	verified = verify_gpu(rsdnm, errnm, frc)
	mflops = ( itmax*(1984.77 * nx
			* ny
			* nz
			- 10923.3 * pow(((nx+ny+nz)/3.0),2.0) 
			+ 27770.9 * (nx+ny+nz)/3.0
			- 144010.0) / (maxtime*1000000.0) )
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_ERHS_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-erhs-1", THREADS_PER_BLOCK_ON_ERHS_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERHS_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-erhs-2", THREADS_PER_BLOCK_ON_ERHS_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERHS_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-erhs-3", THREADS_PER_BLOCK_ON_ERHS_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERHS_4)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-erhs-4", THREADS_PER_BLOCK_ON_ERHS_4, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERROR)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-error", THREADS_PER_BLOCK_ON_ERROR, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_NORM)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-norm", THREADS_PER_BLOCK_ON_NORM, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_JACLD_BLTS)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-jacld-blts", THREADS_PER_BLOCK_ON_JACLD_BLTS, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_JACU_BUTS)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-jacu-buts", THREADS_PER_BLOCK_ON_JACU_BUTS, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_L2NORM)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-l2norm", THREADS_PER_BLOCK_ON_L2NORM, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_PINTGR_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-pintgr-1", THREADS_PER_BLOCK_ON_PINTGR_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_PINTGR_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-pintgr-2", THREADS_PER_BLOCK_ON_PINTGR_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_PINTGR_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-pintgr-3", THREADS_PER_BLOCK_ON_PINTGR_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_PINTGR_4)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-pintgr-4", THREADS_PER_BLOCK_ON_PINTGR_4, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-rhs-1", THREADS_PER_BLOCK_ON_RHS_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-rhs-2", THREADS_PER_BLOCK_ON_RHS_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-rhs-3", THREADS_PER_BLOCK_ON_RHS_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_4)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-rhs-4", THREADS_PER_BLOCK_ON_RHS_4, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_SETBV_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-setbv-1", THREADS_PER_BLOCK_ON_SETBV_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_SETBV_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-setbv-2", THREADS_PER_BLOCK_ON_SETBV_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_SETBV_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-setbv-3", THREADS_PER_BLOCK_ON_SETBV_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_SETIV)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-setiv", THREADS_PER_BLOCK_ON_SETIV, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_SSOR_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-ssor-1", THREADS_PER_BLOCK_ON_SSOR_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_SSOR_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" lu-ssor-2", THREADS_PER_BLOCK_ON_SSOR_2, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" lu-erhs-1", THREADS_PER_BLOCK_ON_ERHS_1)
		gpu_config_string += "%29s\t%25d\n" % (" lu-erhs-2", THREADS_PER_BLOCK_ON_ERHS_2)
		gpu_config_string += "%29s\t%25d\n" % (" lu-erhs-3", THREADS_PER_BLOCK_ON_ERHS_3)
		gpu_config_string += "%29s\t%25d\n" % (" lu-erhs-4", THREADS_PER_BLOCK_ON_ERHS_4)
		gpu_config_string += "%29s\t%25d\n" % (" lu-error", THREADS_PER_BLOCK_ON_ERROR)
		gpu_config_string += "%29s\t%25d\n" % (" lu-norm", THREADS_PER_BLOCK_ON_NORM)
		gpu_config_string += "%29s\t%25d\n" % (" lu-jacld-blts", THREADS_PER_BLOCK_ON_JACLD_BLTS)
		gpu_config_string += "%29s\t%25d\n" % (" lu-jacu-buts", THREADS_PER_BLOCK_ON_JACU_BUTS)
		gpu_config_string += "%29s\t%25d\n" % (" lu-l2norm", THREADS_PER_BLOCK_ON_L2NORM)
		gpu_config_string += "%29s\t%25d\n" % (" lu-pintgr-1", THREADS_PER_BLOCK_ON_PINTGR_1)
		gpu_config_string += "%29s\t%25d\n" % (" lu-pintgr-2", THREADS_PER_BLOCK_ON_PINTGR_2)
		gpu_config_string += "%29s\t%25d\n" % (" lu-pintgr-3", THREADS_PER_BLOCK_ON_PINTGR_3)
		gpu_config_string += "%29s\t%25d\n" % (" lu-pintgr-4", THREADS_PER_BLOCK_ON_PINTGR_4)
		gpu_config_string += "%29s\t%25d\n" % (" lu-rhs-1", THREADS_PER_BLOCK_ON_RHS_1)
		gpu_config_string += "%29s\t%25d\n" % (" lu-rhs-2", THREADS_PER_BLOCK_ON_RHS_2)
		gpu_config_string += "%29s\t%25d\n" % (" lu-rhs-3", THREADS_PER_BLOCK_ON_RHS_3)
		gpu_config_string += "%29s\t%25d\n" % (" lu-rhs-4", THREADS_PER_BLOCK_ON_RHS_4)
		gpu_config_string += "%29s\t%25d\n" % (" lu-setbv-1", THREADS_PER_BLOCK_ON_SETBV_1)
		gpu_config_string += "%29s\t%25d\n" % (" lu-setbv-2", THREADS_PER_BLOCK_ON_SETBV_2)
		gpu_config_string += "%29s\t%25d\n" % (" lu-setbv-3", THREADS_PER_BLOCK_ON_SETBV_3)
		gpu_config_string += "%29s\t%25d\n" % (" lu-setiv", THREADS_PER_BLOCK_ON_SETIV)
		gpu_config_string += "%29s\t%25d\n" % (" lu-ssor-1", THREADS_PER_BLOCK_ON_SSOR_1)
		gpu_config_string += "%29s\t%25d\n" % (" lu-ssor-2", THREADS_PER_BLOCK_ON_SSOR_2)
	
	c_print_results.c_print_results("LU",
			npbparams.CLASS,
			nx, 
			ny,
			nz,
			itmax,
			maxtime,
			mflops,
			"          floating point",
			verified,
			device_prop.name,
			gpu_config_string)
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA LU')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_lu_info(args.CLASS)
	
	main()
