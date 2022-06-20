# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
# 	R. Van der Wijngaart 
# 	W. Saphir 
# 	H. Jin
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


# Global variables
IMAX = 0
JMAX = 0
KMAX = 0
IMAXP = 0
JMAXP = 0

PROFILING_TOTAL_TIME = 0

PROFILING_ADD = 1
PROFILING_COMPUTE_RHS_1 = 2
PROFILING_COMPUTE_RHS_2 = 3
PROFILING_ERROR_NORM_1 = 4
PROFILING_ERROR_NORM_2 = 5
PROFILING_EXACT_RHS_1 = 6
PROFILING_EXACT_RHS_2 = 7
PROFILING_EXACT_RHS_3 = 8
PROFILING_EXACT_RHS_4 = 9
PROFILING_INITIALIZE = 10
PROFILING_RHS_NORM_1 = 11
PROFILING_RHS_NORM_2 = 12
PROFILING_TXINVR = 13
PROFILING_X_SOLVE = 14
PROFILING_Y_SOLVE = 15
PROFILING_Z_SOLVE = 16

#u_host = None
#us_host = None
#vs_host = None
#ws_host = None
#qs_host = None
#rho_i_host = None
#speed_host = None
#square_host = None
#rhs_host = None
#forcing_host = None
#cv_host = None
#rhon_host = None
#rhos_host = None
#rhoq_host = None
#cuf_host = None
#q_host = None
#ue_host = None
#buf_host = None
#lhs_host = None
#lhsp_host = None
#lhsm_host = None
ce = numpy.empty((13, 5), dtype=numpy.float64())

grid_points = numpy.empty(3, dtype=numpy.int32)
nx, ny, nz = 0, 0, 0
dt_host = 0.0

tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
dx1, dx2, dx3, dx4, dx5 = 0.0, 0.0, 0.0, 0.0, 0.0
dy1, dy2, dy3, dy4, dy5 = 0.0, 0.0, 0.0, 0.0, 0.0
dz1, dz2, dz3, dz4, dz5 = 0.0, 0.0, 0.0, 0.0, 0.0
dssp = 0.0
dt = 0.0
dxmax, dymax, dzmax = 0.0, 0.0, 0.0
xxcon1, xxcon2, xxcon3, xxcon4, xxcon5 = 0.0, 0.0, 0.0, 0.0, 0.0
dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1 = 0.0, 0.0, 0.0, 0.0, 0.0
yycon1, yycon2, yycon3, yycon4, yycon5 = 0.0, 0.0, 0.0, 0.0, 0.0
dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1 = 0.0, 0.0, 0.0, 0.0, 0.0
zzcon1, zzcon2, zzcon3, zzcon4, zzcon5 = 0.0, 0.0, 0.0, 0.0, 0.0
dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1 = 0.0, 0.0, 0.0, 0.0, 0.0
dnxm1, dnym1, dnzm1 = 0.0, 0.0, 0.0
c1c2, c1c5, c3c4, c1345, conz1 = 0.0, 0.0, 0.0, 0.0, 0.0
c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0
c4dssp, c5dssp, dtdssp = 0.0, 0.0, 0.0
dttx1, dttx2, dtty1, dtty2, dttz1, dttz2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
bt = 0.0
c2dttx1, c2dtty1, c2dttz1 = 0.0, 0.0, 0.0
comz1, comz4, comz5, comz6 = 0.0, 0.0, 0.0, 0.0
c3c4tx3, c3c4ty3, c3c4tz3 = 0.0, 0.0, 0.0
c2iv, con43, con16 = 0.0, 0.0, 0.0

# GPU variables
u_device = None 
forcing_device = None  
rhs_device = None 
rho_i_device = None  
us_device = None
vs_device = None 
ws_device = None 
qs_device = None 
speed_device = None 
square_device = None 
lhs_device = None 
rhs_buffer_device = None 
rms_buffer_device = None

const_exact_rhs_2_device = None
const_exact_rhs_3_device = None
const_exact_rhs_4_device = None
const_compute_rhs_gpu_kernel_2_device = None
const_x_solve_gpu_kernel_device = None
const_y_solve_gpu_kernel_device = None
const_z_solve_gpu_kernel_device = None
ce_device = None

THREADS_PER_BLOCK_ON_ADD = 0
THREADS_PER_BLOCK_ON_COMPUTE_RHS_1 = 0
THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 = 0
THREADS_PER_BLOCK_ON_ERROR_NORM_1 = 0
THREADS_PER_BLOCK_ON_ERROR_NORM_2 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_1 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_2 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_3 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_4 = 0
THREADS_PER_BLOCK_ON_INITIALIZE = 0
THREADS_PER_BLOCK_ON_RHS_NORM_1 = 0
THREADS_PER_BLOCK_ON_RHS_NORM_2 = 0
THREADS_PER_BLOCK_ON_TXINVR = 0
THREADS_PER_BLOCK_ON_X_SOLVE = 0
THREADS_PER_BLOCK_ON_Y_SOLVE = 0
THREADS_PER_BLOCK_ON_Z_SOLVE = 0

stream = 0

gpu_device_id = 0
total_devices = 0
device_prop = None


def set_global_variables():
	global IMAX, JMAX, KMAX, IMAXP, JMAXP
	#global u_host
	#global us_host, vs_host, ws_host, qs_host, rho_i_host, speed_host, square_host
	#global rhs_host, forcing_host
	#global cv_host, rhon_host, rhos_host, rhoq_host, cuf_host, q_host
	#global ue_host, buf_host
	#global lhs_host, lhsp_host, lhsm_host

	IMAX = npbparams.PROBLEM_SIZE
	JMAX = npbparams.PROBLEM_SIZE
	KMAX = npbparams.PROBLEM_SIZE
	IMAXP = int(IMAX/2*2)
	JMAXP = int(JMAX/2*2)
	
	#u_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	
	#us_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	#vs_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	#ws_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	#qs_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	#rho_i_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	#speed_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	#square_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	
	#rhs_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	#forcing_host = numpy.zeros((KMAX, JMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	
	#cv_host = numpy.empty(npbparams.PROBLEM_SIZE, dtype=numpy.float64())
	#rhon_host = numpy.empty(npbparams.PROBLEM_SIZE, dtype=numpy.float64())
	#rhos_host = numpy.empty(npbparams.PROBLEM_SIZE, dtype=numpy.float64())
	#rhoq_host = numpy.empty(npbparams.PROBLEM_SIZE, dtype=numpy.float64())
	#cuf_host = numpy.empty(npbparams.PROBLEM_SIZE, dtype=numpy.float64())
	#q_host = numpy.empty(npbparams.PROBLEM_SIZE, dtype=numpy.float64())
	
	#ue_host = numpy.empty((5, npbparams.PROBLEM_SIZE), dtype=numpy.float64())
	#buf_host = numpy.empty((5, npbparams.PROBLEM_SIZE), dtype=numpy.float64())
	
	#lhs_host = numpy.empty((IMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	#lhsp_host = numpy.empty((IMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	#lhsm_host = numpy.empty((IMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
#END set_global_variables()


def set_constants(dt_host):
	global ce
	global bt
	global c1, c2, c3, c4, c5
	global dnxm1, dnym1, dnzm1
	global c1c2, c1c5, c3c4, c1345, conz1
	global tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3
	global dx1, dx2, dx3, dx4, dx5
	global dy1, dy2, dy3, dy4, dy5
	global dz1, dz2, dz3, dz4, dz5
	global dxmax, dymax, dzmax
	global dssp
	global c4dssp, c5dssp
	global dttx1, dttx2, dtty1, dtty2, dttz1, dttz2
	global c2dttx1, c2dtty1, c2dttz1
	global dtdssp
	global comz1, comz4, comz5, comz6
	global c3c4tx3, c3c4ty3, c3c4tz3
	global dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1
	global dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1
	global dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1
	global c2iv, con43, con16
	global xxcon1, xxcon2, xxcon3, xxcon4, xxcon5
	global yycon1, yycon2, yycon3, yycon4, yycon5
	global zzcon1, zzcon2, zzcon3, zzcon4, zzcon5
	
	global const_exact_rhs_2_device, const_exact_rhs_3_device, const_exact_rhs_4_device
	global const_compute_rhs_gpu_kernel_2_device
	global const_x_solve_gpu_kernel_device, const_y_solve_gpu_kernel_device, const_z_solve_gpu_kernel_device
	global ce_device
	
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
	# -------
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
	# -------
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
	# -------
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
	# -------
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
	# -------
	bt = math.sqrt(0.5)
	dt = dt_host
	c1 = 1.4
	c2 = 0.4
	c3 = 0.1
	c4 = 1.0
	c5 = 1.4
	# -------
	dnxm1 = 1.0 / (grid_points[0]-1)
	dnym1 = 1.0 / (grid_points[1]-1)
	dnzm1 = 1.0 / (grid_points[2]-1)
	# -------
	c1c2 = c1 * c2
	c1c5 = c1 * c5
	c3c4 = c3 * c4
	c1345 = c1c5 * c3c4
	# -------
	conz1 = (1.0-c1c5)
	# -------
	tx1 = 1.0 / (dnxm1*dnxm1)
	tx2 = 1.0 / (2.0*dnxm1)
	tx3 = 1.0 / dnxm1
	# -------
	ty1 = 1.0 / (dnym1*dnym1)
	ty2 = 1.0 / (2.0*dnym1)
	ty3 = 1.0 / dnym1
	# -------
	tz1 = 1.0 / (dnzm1*dnzm1)
	tz2 = 1.0 / (2.0*dnzm1)
	tz3 = 1.0 / dnzm1
	# -------
	dx1 = 0.75
	dx2 = 0.75
	dx3 = 0.75
	dx4 = 0.75
	dx5 = 0.75
	# -------
	dy1 = 0.75
	dy2 = 0.75
	dy3 = 0.75
	dy4 = 0.75
	dy5 = 0.75
	# -------
	dz1 = 1.0 
	dz2 = 1.0 
	dz3 = 1.0 
	dz4 = 1.0 
	dz5 = 1.0
	# -------
	dxmax = max(dx3, dx4)
	dymax = max(dy2, dy4)
	dzmax = max(dz2, dz3)
	# -------
	dssp = 0.25 * max(dx1, max(dy1, dz1))
	# -------
	c4dssp = 4.0 * dssp
	c5dssp = 5.0 * dssp
	# -------
	dttx1 = dt * tx1
	dttx2 = dt * tx2
	dtty1 = dt * ty1
	dtty2 = dt * ty2
	dttz1 = dt * tz1
	dttz2 = dt * tz2
	# -------
	c2dttx1 = 2.0 * dttx1
	c2dtty1 = 2.0 * dtty1
	c2dttz1 = 2.0 * dttz1
	# -------
	dtdssp = dt * dssp
	# -------
	comz1 = dtdssp
	comz4 = 4.0 * dtdssp
	comz5 = 5.0 * dtdssp
	comz6 = 6.0 * dtdssp
	# -------
	c3c4tx3 = c3c4 * tx3
	c3c4ty3 = c3c4 * ty3
	c3c4tz3 = c3c4 * tz3
	# -------
	dx1tx1 = dx1 * tx1
	dx2tx1 = dx2 * tx1
	dx3tx1 = dx3 * tx1
	dx4tx1 = dx4 * tx1
	dx5tx1 = dx5 * tx1
	# -------
	dy1ty1 = dy1 * ty1
	dy2ty1 = dy2 * ty1
	dy3ty1 = dy3 * ty1
	dy4ty1 = dy4 * ty1
	dy5ty1 = dy5 * ty1
	# -------
	dz1tz1 = dz1 * tz1
	dz2tz1 = dz2 * tz1
	dz3tz1 = dz3 * tz1
	dz4tz1 = dz4 * tz1
	dz5tz1 = dz5 * tz1
	# -------
	c2iv = 2.5
	con43 = 4.0 / 3.0
	con16 = 1.0 / 6.0
	# -------
	xxcon1 = c3c4tx3 * con43 * tx3
	xxcon2 = c3c4tx3 * tx3
	xxcon3 = c3c4tx3 * conz1 * tx3
	xxcon4 = c3c4tx3 * con16 * tx3
	xxcon5 = c3c4tx3 * c1c5 * tx3
	# -------
	yycon1 = c3c4ty3 * con43 * ty3
	yycon2 = c3c4ty3 * ty3
	yycon3 = c3c4ty3 * conz1 * ty3
	yycon4 = c3c4ty3 * con16 * ty3
	yycon5 = c3c4ty3 * c1c5 * ty3
	# -------
	zzcon1 = c3c4tz3 * con43 * tz3
	zzcon2 = c3c4tz3 * tz3
	zzcon3 = c3c4tz3 * conz1 * tz3
	zzcon4 = c3c4tz3 * con16 * tz3
	zzcon5 = c3c4tz3 * c1c5 * tz3
	
	#Constant arrays to GPU memory
	const_exact_rhs_2 = numpy.array([dnzm1, dnym1, dnxm1, tx2, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1, 
									c1, c2, xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dssp], numpy.float64)
	const_exact_rhs_2_device = cuda.to_device(const_exact_rhs_2)
	# -------
	const_exact_rhs_3 = numpy.array([dnzm1, dnym1, dnxm1, ty2, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1, 
									c1, c2, yycon1, yycon2, yycon3, yycon4, yycon5, dssp], numpy.float64)
	const_exact_rhs_3_device = cuda.to_device(const_exact_rhs_3)
	# -------
	const_exact_rhs_4 = numpy.array([dnzm1, dnym1, dnxm1, tz2, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1, 
									c1, c2, zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dssp], numpy.float64)
	const_exact_rhs_4_device = cuda.to_device(const_exact_rhs_4)
	# -------
	const_compute_rhs_gpu_kernel_2 = numpy.array([dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1,
									dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
									dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1,
									tx2, ty2, tz2,
									xxcon2, xxcon3, xxcon4, xxcon5,
									yycon2, yycon3, yycon4, yycon5,
									zzcon2, zzcon3, zzcon4, zzcon5,
									c1, c2, con43, dssp, dt], numpy.float64)
	const_compute_rhs_gpu_kernel_2_device = cuda.to_device(const_compute_rhs_gpu_kernel_2)
	# -------
	const_x_solve_gpu_kernel = numpy.array([dx1, dx2, dx5, dttx1, dttx2, dxmax, c2dttx1, 
										bt, c1c5, c3c4, con43, comz1, comz4, comz5, comz6], numpy.float64)
	const_x_solve_gpu_kernel_device = cuda.to_device(const_x_solve_gpu_kernel)
	# -------
	const_y_solve_gpu_kernel = numpy.array([dy1, dy3, dy5, dtty1, dtty2, dymax, c2dtty1, 
										bt, c1c5, c3c4, con43, comz1, comz4, comz5, comz6], numpy.float64)
	const_y_solve_gpu_kernel_device = cuda.to_device(const_y_solve_gpu_kernel)
	# -------
	const_z_solve_gpu_kernel = numpy.array([dz1, dz4, dz5, dttz1, dttz2, dzmax, c2dttz1, 
										bt, c1c5, c3c4, con43, comz1, comz4, comz5, comz6, c2iv], numpy.float64)
	const_z_solve_gpu_kernel_device = cuda.to_device(const_z_solve_gpu_kernel)
	# -------
	ce_device = cuda.to_device(ce)
	
	#Another constant values are going to be passed to kernels as parameters
#END set_constants()


def setup_gpu():
	global gpu_device_id, total_devices
	global device_prop
	
	global THREADS_PER_BLOCK_ON_ADD
	global THREADS_PER_BLOCK_ON_COMPUTE_RHS_1, THREADS_PER_BLOCK_ON_COMPUTE_RHS_2
	global THREADS_PER_BLOCK_ON_ERROR_NORM_1, THREADS_PER_BLOCK_ON_ERROR_NORM_2
	global THREADS_PER_BLOCK_ON_EXACT_RHS_1, THREADS_PER_BLOCK_ON_EXACT_RHS_2
	global THREADS_PER_BLOCK_ON_EXACT_RHS_3, THREADS_PER_BLOCK_ON_EXACT_RHS_4
	global THREADS_PER_BLOCK_ON_INITIALIZE
	global THREADS_PER_BLOCK_ON_RHS_NORM_1, THREADS_PER_BLOCK_ON_RHS_NORM_2
	global THREADS_PER_BLOCK_ON_TXINVR
	global THREADS_PER_BLOCK_ON_X_SOLVE, THREADS_PER_BLOCK_ON_Y_SOLVE, THREADS_PER_BLOCK_ON_Z_SOLVE
	
	global u_device, forcing_device
	global rhs_device
	global rho_i_device, us_device, vs_device, ws_device, qs_device, speed_device, square_device
	global lhs_device, rhs_buffer_device, rms_buffer_device
	
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
	# -- new
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_ADD
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ADD = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ADD = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_COMPUTE_RHS_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_ERROR_NORM_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERROR_NORM_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERROR_NORM_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_ERROR_NORM_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERROR_NORM_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERROR_NORM_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_EXACT_RHS_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_EXACT_RHS_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_EXACT_RHS_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_EXACT_RHS_4
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_4 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_4 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_INITIALIZE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_INITIALIZE = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_INITIALIZE = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_RHS_NORM_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_NORM_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_NORM_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_RHS_NORM_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_NORM_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_NORM_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_TXINVR
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_TXINVR = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_TXINVR = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_X_SOLVE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_X_SOLVE = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_X_SOLVE = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_Y_SOLVE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Y_SOLVE = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Y_SOLVE = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.SP_THREADS_PER_BLOCK_ON_Z_SOLVE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Z_SOLVE = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Z_SOLVE = device_prop.WARP_SIZE
		
	n_float64 = numpy.float64
	u_device = cuda.device_array(int(5 * nx * ny * nz), n_float64)
	forcing_device = cuda.device_array(int(5 * nx * ny * nz), n_float64)
	rhs_device = cuda.device_array(int(5 * nx * ny * nz), n_float64)
	
	rho_i_device = cuda.device_array(int(nx * ny * nz), n_float64)
	us_device = cuda.device_array(int(nx * ny * nz), n_float64)
	vs_device = cuda.device_array(int(nx * ny * nz), n_float64)
	ws_device = cuda.device_array(int(nx * ny * nz), n_float64)
	qs_device = cuda.device_array(int(nx * ny * nz), n_float64)
	speed_device = cuda.device_array(int(nx * ny * nz), n_float64)
	square_device = cuda.device_array(int(nx * ny * nz), n_float64)
	
	lhs_device = cuda.device_array(int(9 * nx * ny * nz), n_float64) #lhs[0..1], lhsm[2..3], lhsp[4..8]
	rhs_buffer_device = cuda.device_array(int(5 * nx * ny * nz), n_float64)
	
	facesize = int(max(max(nx*ny, nx*nz), ny*nz) * 5)
	rms_buffer_device = cuda.device_array(facesize, n_float64)
#END setup_gpu()


#*****************************************************************
#************************* GPU FUNCTIONS *************************
#*****************************************************************

# ---------------------------------------------------------------------
# this function returns the exact solution at point xi, eta, zeta  
# ---------------------------------------------------------------------
@cuda.jit('void(float64, float64, float64, float64[:], float64[:, :])', device=True)
def exact_solution_gpu_device(xi,
							eta,
							zeta,
							dtemp,
							ce):
	for m in range(5):
		dtemp[m] = ( ce[0][m]+xi*
			(ce[1][m]+xi*
			 (ce[4][m]+xi*
			  (ce[7][m]+xi*
			   ce[10][m])))+eta*
			(ce[2][m]+eta*
			 (ce[5][m]+eta*
			  (ce[8][m]+eta*
			   ce[11][m])))+zeta*
			(ce[3][m]+zeta*
			 (ce[6][m]+zeta*
			  (ce[9][m]+zeta*
			   ce[12][m]))) )
#END exact_solution_gpu_device()


@cuda.jit('void(float64[:], int32, int32, int32)')
def rhs_norm_gpu_kernel_2(rms,
						nx,
						ny,
						nz):
	buff = cuda.shared.array(shape=0, dtype=numba.float64)

	i = cuda.threadIdx.x

	for m in range(5):
		buff[i+(m*cuda.blockDim.x)] = 0.0
	while i < (nx*ny):
		for m in range(5):
			buff[cuda.threadIdx.x+(m*cuda.blockDim.x)] += rms[i+nx*ny*m]
		i += cuda.blockDim.x

	maxpos = cuda.blockDim.x
	dist = int((maxpos+1)/2)
	i = cuda.threadIdx.x
	cuda.syncthreads()
	while maxpos>1:
		if i<dist and i+dist<maxpos:
			for m in range(5):
				buff[i+(m*cuda.blockDim.x)] += buff[(i+dist)+(m*cuda.blockDim.x)]

		maxpos = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	m = cuda.threadIdx.x
	if m<5: 
		rms[m] = math.sqrt(buff[0+(m*cuda.blockDim.x)] / ((nz-2)*(ny-2)*(nx-2)))
#END rhs_norm_gpu_kernel_2()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32)')
def rhs_norm_gpu_kernel_1(rms,
						rhs,
						nx,
						ny,
						nz):
	rms_loc = cuda.local.array(5, numba.float64)

	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

	if j>=ny or i>=nx:
		return

	for m in range(5): 
		rms_loc[m] = 0.0
	if i>=1 and i<nx-1 and j>=1 and j<ny-1:
		for k in range(1, nz-1):
			for m in range(5): 
				add = rhs[(m)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
				rms_loc[m] += add*add

	for m in range(5): 
		rms[i+nx*(j+ny*m)] = rms_loc[m]
#END rhs_norm_gpu_kernel_1()


def rhs_norm_gpu(rms,
				 rms_buffer_device, 
				 rhs_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_NORM_1)
	# #KERNEL RHS NORM 1
	rhs_norm_1_threads_per_block = THREADS_PER_BLOCK_ON_RHS_NORM_1
	rhs_norm_1_blocks_per_grid = (ny, nx)

	rhs_norm_gpu_kernel_1[rhs_norm_1_blocks_per_grid, 
		rhs_norm_1_threads_per_block](rms_buffer_device, 
									rhs_device, 
									nx, 
									ny, 
									nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_NORM_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_NORM_2)
	# #KERNEL RHS NORM 2
	rhs_norm_2_threads_per_block = THREADS_PER_BLOCK_ON_RHS_NORM_2
	rhs_norm_2_blocks_per_grid = 1
	size_shared_data = rms_buffer_device.dtype.itemsize * rhs_norm_2_threads_per_block * 5

	rhs_norm_gpu_kernel_2[rhs_norm_2_blocks_per_grid,
		rhs_norm_2_threads_per_block,
		stream,
		size_shared_data](rms_buffer_device, 
						nx, 
						ny, 
						nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_NORM_2)

	#cudaMemcpy(rms, rms_buffer_device, 5*sizeof(double), cudaMemcpyDeviceToHost);
	rms_buffer = rms_buffer_device.copy_to_host() #Numba requires to copy the full array
	for i in range(5):
		rms[i] = rms_buffer[i]
#END rhs_norm_gpu(0


@cuda.jit('void(float64[:], int32, int32, int32)')
def error_norm_gpu_kernel_2(rms,
							nx,
							ny,
							nz):
	buff = cuda.shared.array(shape=0, dtype=numba.float64)

	i = cuda.threadIdx.x

	for m in range(5):
		buff[i+(m*cuda.blockDim.x)] = 0.0
	while i < (nx*ny):
		for m in range(5):
			buff[cuda.threadIdx.x+(m*cuda.blockDim.x)] += rms[i+nx*ny*m]
		i += cuda.blockDim.x

	maxpos = cuda.blockDim.x
	dist = int((maxpos+1)/2)
	i = cuda.threadIdx.x
	cuda.syncthreads()
	while maxpos>1:
		if i<dist and i+dist<maxpos:
			for m in range(5):
				buff[i+(m*cuda.blockDim.x)] += buff[(i+dist)+(m*cuda.blockDim.x)]

		maxpos = dist
		dist = int((dist+1)/2)
		cuda.syncthreads()

	m = cuda.threadIdx.x
	if m<5: 
		rms[m] = math.sqrt(buff[0+(m*cuda.blockDim.x)] / ((nz-2)*(ny-2)*(nx-2)))
#END error_norm_gpu_kernel_2()


@cuda.jit('void(float64[:], float64[:], int32, int32, int32, float64, float64, float64, float64[:, :])')
def error_norm_gpu_kernel_1(rms,
							u,
							nx,
							ny,
							nz,
							dnxm1, dnym1, dnzm1,
							ce_device):
	u_exact = cuda.local.array(5, numba.float64)
	rms_loc = cuda.local.array(5, numba.float64)
	
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

	if j>=ny or i>=nx:
		return
	
	for m in range(5):
		rms_loc[m] = 0.0
	xi = i*dnxm1
	eta = j*dnym1
	for k in range(nz):
		zeta = k * dnzm1
		exact_solution_gpu_device(xi, eta, zeta, u_exact, ce_device)
		for m in range(5):
			add = u[(i)+nx*((j)+ny*((k)+nz*(m)))] - u_exact[m]
			rms_loc[m] += add*add

	for m in range(5):
		rms[i+nx*(j+ny*m)] = rms_loc[m]
#END error_norm_gpu_kernel_1()


# ---------------------------------------------------------------------
# this function computes the norm of the difference between the
# computed solution and the exact solution
# ---------------------------------------------------------------------
def error_norm_gpu(rms,
				   rms_buffer_device,
				   u_device):
	global ce_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERROR_NORM_1)
	# #KERNEL ERROR NORM 1 
	error_norm_1_threads_per_block = THREADS_PER_BLOCK_ON_ERROR_NORM_1
	error_norm_1_blocks_per_grid = (ny, nx)

	error_norm_gpu_kernel_1[error_norm_1_blocks_per_grid, 
		error_norm_1_threads_per_block](rms_buffer_device, 
										u_device, 
										nx, 
										ny, 
										nz,
										dnxm1, dnym1, dnzm1,
										ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERROR_NORM_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ERROR_NORM_2)
	# #KERNEL ERROR NORM 2
	error_norm_2_threads_per_block = THREADS_PER_BLOCK_ON_ERROR_NORM_2
	error_norm_2_blocks_per_grid = 1
	size_shared_data = rms_buffer_device.dtype.itemsize * error_norm_2_threads_per_block * 5

	error_norm_gpu_kernel_2[error_norm_2_blocks_per_grid,
		error_norm_2_threads_per_block,
		stream,
		size_shared_data](rms_buffer_device, 
						nx, 
						ny, 
						nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ERROR_NORM_2)

	#cudaMemcpy(rms, rms_buffer_device, 5*sizeof(double), cudaMemcpyDeviceToHost);
	rms_buffer = rms_buffer_device.copy_to_host() #Numba requires to copy the full array
	for i in range(5):
		rms[i] = rms_buffer[i]
#END error_norm_gpu()


# ---------------------------------------------------------------------
# verification routine                         
# ---------------------------------------------------------------------
#
def verify_gpu(u_device, rhs_device, forcing_device):
	global rho_i_device, us_device, vs_device, ws_device, speed_device, qs_device, square_device
	global rms_buffer_device
	
	dt = dt_host
	xce = numpy.empty(5, dtype=numpy.float64)
	xcr = numpy.empty(5, dtype=numpy.float64)
	# ---------------------------------------------------------------------
	# tolerance level
	# ---------------------------------------------------------------------
	epsilon=1.0e-08
	# ---------------------------------------------------------------------
	# compute the error norm and the residual norm, and exit if not printing
	# ---------------------------------------------------------------------
	error_norm_gpu(xce, 
				rms_buffer_device, u_device)
	
	compute_rhs_gpu(rho_i_device, us_device, vs_device, ws_device, 
					speed_device, qs_device, square_device, 
					u_device, rhs_device, forcing_device)
	
	rhs_norm_gpu(xcr,
				rms_buffer_device, rhs_device)
	
	for m in range(5):
		xcr[m] = xcr[m] / dt
	
	verified = True
	dtref = 0.0
	xcrref = numpy.repeat(1.0, 5)
	xceref = numpy.repeat(1.0, 5)
	
	# ---------------------------------------------------------------------
	# reference data for 12X12X12 grids after 100 time steps, with DT = 1.50d-02
	# ---------------------------------------------------------------------
	if npbparams.CLASS == 'S':
		dtref = 1.5e-2
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 2.7470315451339479e-02
		xcrref[1] = 1.0360746705285417e-02
		xcrref[2] = 1.6235745065095532e-02
		xcrref[3] = 1.5840557224455615e-02
		xcrref[4] = 3.4849040609362460e-02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 2.7289258557377227e-05
		xceref[1] = 1.0364446640837285e-05
		xceref[2] = 1.6154798287166471e-05
		xceref[3] = 1.5750704994480102e-05
		xceref[4] = 3.4177666183390531e-05
	# ---------------------------------------------------------------------
	# reference data for 36X36X36 grids after 400 time steps, with DT = 1.5d-03
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'W':
		dtref = 1.5e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.1893253733584e-02
		xcrref[1] = 0.1717075447775e-03
		xcrref[2] = 0.2778153350936e-03
		xcrref[3] = 0.2887475409984e-03
		xcrref[4] = 0.3143611161242e-02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.7542088599534e-04
		xceref[1] = 0.6512852253086e-05
		xceref[2] = 0.1049092285688e-04
		xceref[3] = 0.1128838671535e-04
		xceref[4] = 0.1212845639773e-03
	# ---------------------------------------------------------------------
	# reference data for 64X64X64 grids after 400 time steps, with DT = 1.5d-03
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'A':
		dtref = 1.5e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 2.4799822399300195
		xcrref[1] = 1.1276337964368832
		xcrref[2] = 1.5028977888770491
		xcrref[3] = 1.4217816211695179
		xcrref[4] = 2.1292113035138280
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 1.0900140297820550e-04
		xceref[1] = 3.7343951769282091e-05
		xceref[2] = 5.0092785406541633e-05
		xceref[3] = 4.7671093939528255e-05
		xceref[4] = 1.3621613399213001e-04
	# ---------------------------------------------------------------------
	# reference data for 102X102X102 grids after 400 time steps,
	# with DT = 1.0d-03
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'B':
		dtref = 1.0e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.6903293579998e+02
		xcrref[1] = 0.3095134488084e+02
		xcrref[2] = 0.4103336647017e+02
		xcrref[3] = 0.3864769009604e+02
		xcrref[4] = 0.5643482272596e+02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.9810006190188e-02
		xceref[1] = 0.1022827905670e-02
		xceref[2] = 0.1720597911692e-02
		xceref[3] = 0.1694479428231e-02
		xceref[4] = 0.1847456263981e-01
	# ---------------------------------------------------------------------
	# reference data for 162X162X162 grids after 400 time steps,
	# with DT = 0.67d-03
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'C':
		dtref = 0.67e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.5881691581829e+03
		xcrref[1] = 0.2454417603569e+03
		xcrref[2] = 0.3293829191851e+03
		xcrref[3] = 0.3081924971891e+03
		xcrref[4] = 0.4597223799176e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.2598120500183e+00
		xceref[1] = 0.2590888922315e-01
		xceref[2] = 0.5132886416320e-01
		xceref[3] = 0.4806073419454e-01
		xceref[4] = 0.5483377491301e+00
	# ---------------------------------------------------------------------
	# reference data for 408x408x408 grids after 500 time steps,
	# with DT = 0.3d-03
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'D':
		dtref = 0.30e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.1044696216887e+05
		xcrref[1] = 0.3204427762578e+04
		xcrref[2] = 0.4648680733032e+04
		xcrref[3] = 0.4238923283697e+04
		xcrref[4] = 0.7588412036136e+04
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.5089471423669e+01
		xceref[1] = 0.5323514855894e+00
		xceref[2] = 0.1187051008971e+01
		xceref[3] = 0.1083734951938e+01
		xceref[4] = 0.1164108338568e+02
	# ---------------------------------------------------------------------
	# reference data for 1020x1020x1020 grids after 500 time steps,
	# with DT = 0.1d-03
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'E':
		dtref = 0.10e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.6255387422609e+05
		xcrref[1] = 0.1495317020012e+05
		xcrref[2] = 0.2347595750586e+05
		xcrref[3] = 0.2091099783534e+05
		xcrref[4] = 0.4770412841218e+05
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.6742735164909e+02
		xceref[1] = 0.5390656036938e+01
		xceref[2] = 0.1680647196477e+02
		xceref[3] = 0.1536963126457e+02
		xceref[4] = 0.1575330146156e+03
	else:
		verified = False

	# ---------------------------------------------------------------------
	# verification test for residuals if gridsize is one of 
	# the defined grid sizes above (*class_npb != 'U')
	# ---------------------------------------------------------------------
	# compute the difference of solution values and the known reference values.
	# ---------------------------------------------------------------------
	xcrdif = numpy.empty(5, dtype=numpy.float64)
	xcedif = numpy.empty(5, dtype=numpy.float64)
	for m in range(5):
		xcrdif[m] = abs((xcr[m]-xcrref[m]) / xcrref[m])
		xcedif[m] = abs((xce[m]-xceref[m]) / xceref[m])

	# ---------------------------------------------------------------------
	# output the comparison of computed results to known cases.
	# ---------------------------------------------------------------------
	print(" Verification being performed for class %c" % (npbparams.CLASS))
	print(" accuracy setting for epsilon = %20.13E" % (epsilon))
	
	verified = abs(dt-dtref) <= epsilon
	if not verified:
		print(" DT does not match the reference value of %15.8E" % (dtref))

	print(" Comparison of RMS-norms of residual")
	for m in range(5):
		if xcrdif[m] <= epsilon:
			print("          %2d%20.13E%20.13E%20.13E" % (m+1, xcr[m], xcrref[m], xcrdif[m]))
		else:
			verified = False
			print(" FAILURE: %2d%20.13E%20.13E%20.13E" % (m+1, xcr[m], xcrref[m], xcrdif[m]))

	print(" Comparison of RMS-norms of solution error")
	for m in range(5):
		if xcedif[m] <= epsilon:
			print("          %2d%20.13E%20.13E%20.13E" % (m+1, xce[m], xceref[m], xcedif[m]))
		else:
			verified = False
			print(" FAILURE: %2d%20.13E%20.13E%20.13E" % (m+1, xce[m], xceref[m], xcedif[m]))

	if verified:
		print(" Verification Successful")
	else:
		print(" Verification failed")
	
	return verified
#END verify()


# ---------------------------------------------------------------------
# addition of update to the vector u
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:], float64[:], int32, int32, int32)')
def add_gpu_kernel(u,
				rhs,
				nx,
				ny,
				nz):
	i_j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j_k % nx
	j = int(i_j_k / nx) % ny
	k = int(i_j_k / (nx * ny))

	if i_j_k >= (nx*ny*nz):
		return

	# array(m,i,j,k)
	u[(i)+nx*((j)+ny*((k)+nz*(0)))] += rhs[(0)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	u[(i)+nx*((j)+ny*((k)+nz*(1)))] += rhs[(1)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	u[(i)+nx*((j)+ny*((k)+nz*(2)))] += rhs[(2)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	u[(i)+nx*((j)+ny*((k)+nz*(3)))] += rhs[(3)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	u[(i)+nx*((j)+ny*((k)+nz*(4)))] += rhs[(4)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
#END add_gpu_kernel()


# ---------------------------------------------------------------------
# addition of update to the vector u
# ---------------------------------------------------------------------
def add_gpu(u_device, 
			rhs_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ADD)
	# #KERNEL ADD
	add_workload = nx * ny * nz
	add_threads_per_block = THREADS_PER_BLOCK_ON_ADD
	add_blocks_per_grid = math.ceil(add_workload / add_threads_per_block)

	add_gpu_kernel[add_blocks_per_grid, 
		add_threads_per_block](u_device, 
							rhs_device, 
							nx, 
							ny, 
							nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ADD)
#END add_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],  float64[:], float64[:], float64[:], int32, int32, int32, float64[:])')
def z_solve_gpu_kernel(rho_i,
					us,
					vs,
					ws,
					speed,
					qs,
					u,
					rhs,
					lhs,
					rtmp, #rhstmp
					nx,
					ny,
					nz, 
					const_arr):
	rhos = cuda.local.array(3, numba.float64)
	cv = cuda.local.array(3, numba.float64)
	_lhs = cuda.local.array((3, 5), numba.float64)
	_lhsp = cuda.local.array((3, 5), numba.float64)
	_rhs = cuda.local.array((3, 5), numba.float64)

	# coalesced
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1

	if (j>=(ny-1)) or (i>=(nx-1)):
		return

	#Constants
	dz1, dz4, dz5 = const_arr[0], const_arr[1], const_arr[2]
	dttz1, dttz2 = const_arr[3], const_arr[4]
	dzmax, c2dttz1 = const_arr[5], const_arr[6]
	bt, c1c5, c3c4, con43 = const_arr[7], const_arr[8], const_arr[9], const_arr[10]
	comz1, comz4, comz5, comz6 = const_arr[11], const_arr[12], const_arr[13], const_arr[14]
	c2iv = const_arr[15]

	# ---------------------------------------------------------------------
	# computes the left hand side for the three z-factors   
	# ---------------------------------------------------------------------
	# first fill the lhs for the u-eigenvalue                          
	# ---------------------------------------------------------------------
	_lhs[0][0] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((0)+nz*(4)))] = 0.0
	_lhs[0][1] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((0)+nz*(5)))] = 0.0
	_lhs[0][2] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((0)+nz*(6)))] = 1.0
	_lhs[0][3] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((0)+nz*(7)))] = 0.0
	_lhs[0][4] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((0)+nz*(8)))] = 0.0
	for k in range(3):
		fac1 = c3c4*rho_i[(i)+(j)*nx+(k)*nx*ny]
		rhos[k] = max(max(max(dz4+con43*fac1, dz5+c1c5*fac1), dzmax+fac1), dz1)
		cv[k] = ws[(i)+(j)*nx+(k)*nx*ny]

	_lhs[1][0] = 0.0
	_lhs[1][1] = -dttz2*cv[0]-dttz1*rhos[0]
	_lhs[1][2] = 1.0+c2dttz1*rhos[1]
	_lhs[1][3] = dttz2*cv[2]-dttz1*rhos[2]
	_lhs[1][4] = 0.0
	_lhs[1][2] += comz5
	_lhs[1][3] -= comz4
	_lhs[1][4] += comz1
	for m in range(5): 
		lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((1)+nz*(m+4)))] = _lhs[1][m]
	rhos[0] = rhos[1]
	rhos[1] = rhos[2]
	cv[0] = cv[1]
	cv[1] = cv[2]
	for m in range(3):
		_rhs[0][m] = rhs[(m)+(i)*5+(j)*5*nx+(0)*5*nx*ny]
		_rhs[1][m] = rhs[(m)+(i)*5+(j)*5*nx+(1)*5*nx*ny]

	# ---------------------------------------------------------------------
	# FORWARD ELIMINATION  
	# ---------------------------------------------------------------------
	for k in range(nz-2): 
		# ---------------------------------------------------------------------
		# first fill the lhs for the u-eigenvalue                   
		# ---------------------------------------------------------------------
		if (k+2)==(nz-1):
			_lhs[2][0] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(4)))] = 0.0
			_lhs[2][1] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(5)))] = 0.0
			_lhs[2][2] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(6)))] = 1.0
			_lhs[2][3] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(7)))] = 0.0
			_lhs[2][4] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(8)))] = 0.0
		else:
			fac1 = c3c4*rho_i[(i)+(j)*nx+(k+3)*nx*ny]
			rhos[2] = max(max(max(dz4+con43*fac1, dz5+c1c5*fac1), dzmax+fac1), dz1)
			cv[2] = ws[(i)+(j)*nx+(k+3)*nx*ny]
			_lhs[2][0] = 0.0
			_lhs[2][1] = -dttz2*cv[0]-dttz1*rhos[0]
			_lhs[2][2] = 1.0+c2dttz1*rhos[1]
			_lhs[2][3] = dttz2*cv[2]-dttz1*rhos[2]
			_lhs[2][4] = 0.0
			# ---------------------------------------------------------------------
			# add fourth order dissipation                             
			# ---------------------------------------------------------------------
			if (k+2)==(2):
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
				_lhs[2][4] += comz1
			elif ((k+2)>=(3)) and ((k+2)<(nz-3)):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
				_lhs[2][4] += comz1
			elif (k+2)==(nz-3):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
			elif (k+2)==(nz-2):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz5
				
			# ---------------------------------------------------------------------
			# store computed lhs for later reuse
			# ---------------------------------------------------------------------
			for m in range(5): 
				lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(m+4)))] = _lhs[2][m]
			rhos[0] = rhos[1]
			rhos[1] = rhos[2]
			cv[0] = cv[1]
			cv[1] = cv[2]

		# ---------------------------------------------------------------------
		# load rhs values for current iteration
		# ---------------------------------------------------------------------
		for m in range(3): 
			_rhs[2][m] = rhs[(m)+(i)*5+(j)*5*nx+(k+2)*5*nx*ny]
		# ---------------------------------------------------------------------
		# perform current iteration
		# ---------------------------------------------------------------------
		fac1 = 1.0/_lhs[0][2]
		_lhs[0][3] *= fac1
		_lhs[0][4] *= fac1
		for m in range(3):
			_rhs[0][m] *= fac1
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
		for m in range(3):
			_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3]
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4]
		for m in range(3):
			_rhs[2][m] -= _lhs[2][0]*_rhs[0][m]
		# ---------------------------------------------------------------------
		# store computed lhs and prepare data for next iteration 
		# rhs is stored in a temp array such that write accesses are coalesced 
		# ---------------------------------------------------------------------
		lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(0)))] = _lhs[0][3]
		lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(1)))] = _lhs[0][4]
		for m in range(5):
			_lhs[0][m] = _lhs[1][m]
			_lhs[1][m] = _lhs[2][m]

		for m in range(3):
			rtmp[(i)+nx*((j)+ny*((k)+nz*(m)))] = _rhs[0][m]
			_rhs[0][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[2][m]

	#END for k in range(nz-2)
	# ---------------------------------------------------------------------
	# the last two rows in this zone are a bit different,  
	# since they do not have two more rows available for the
	# elimination of off-diagonal entries    
	# ---------------------------------------------------------------------
	k = nz-2
	fac1 = 1.0/_lhs[0][2]
	_lhs[0][3] *= fac1
	_lhs[0][4] *= fac1
	for m in range(3):
		_rhs[0][m] *= fac1
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
	for m in range(3):
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
	# ---------------------------------------------------------------------
	# scale the last row immediately 
	# ---------------------------------------------------------------------
	fac1 = 1.0/_lhs[1][2]
	for m in range(3):
		_rhs[1][m] *= fac1
	lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(0)))] = _lhs[0][3]
	lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(1)))] = _lhs[0][4]
	# ---------------------------------------------------------------------
	# subsequently, fill the other factors (u+c), (u-c)
	# ---------------------------------------------------------------------
	for k in range(3):
		cv[k] = speed[(i)+(j)*nx+(k)*nx*ny]
	for m in range(5):
		_lhsp[0][m] = _lhs[0][m] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((0)+nz*(m+4)))]
		_lhsp[1][m] = _lhs[1][m] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((1)+nz*(m+4)))]

	_lhsp[1][1] -= dttz2*cv[0]
	_lhsp[1][3] += dttz2*cv[2]
	_lhs[1][1] += dttz2*cv[0]
	_lhs[1][3] -= dttz2*cv[2]
	cv[0] = cv[1]
	cv[1] = cv[2]
	_rhs[0][3] = rhs[(3)+(i)*5+(j)*5*nx+(0)*5*nx*ny]
	_rhs[0][4] = rhs[(4)+(i)*5+(j)*5*nx+(0)*5*nx*ny]
	_rhs[1][3] = rhs[(3)+(i)*5+(j)*5*nx+(1)*5*nx*ny]
	_rhs[1][4] = rhs[(4)+(i)*5+(j)*5*nx+(1)*5*nx*ny]
	# ---------------------------------------------------------------------
	# do the u+c and the u-c factors                 
	# ---------------------------------------------------------------------
	for k in range(nz-2):
		#---------------------------------------------------------------------
		#first, fill the other factors (u+c), (u-c) 
		#---------------------------------------------------------------------
		for m in range(5):
			_lhsp[2][m] = _lhs[2][m] = lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k+2)+nz*(m+4)))]

		_rhs[2][3] = rhs[(3)+(i)*5+(j)*5*nx+(k+2)*5*nx*ny]
		_rhs[2][4] = rhs[(4)+(i)*5+(j)*5*nx+(k+2)*5*nx*ny]
		if (k+2)<(nz-1):
			cv[2] = speed[(i)+(j)*nx+(k+3)*nx*ny]
			_lhsp[2][1] -= dttz2*cv[0]
			_lhsp[2][3] += dttz2*cv[2]
			_lhs[2][1] += dttz2*cv[0]
			_lhs[2][3] -= dttz2*cv[2]
			cv[0] = cv[1]
			cv[1] = cv[2]

		m = 3
		fac1 = 1.0/_lhsp[0][2]
		_lhsp[0][3] *= fac1
		_lhsp[0][4] *= fac1
		_rhs[0][m] *= fac1
		_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3]
		_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4]
		_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m]
		_lhsp[2][1] -= _lhsp[2][0]*_lhsp[0][3]
		_lhsp[2][2] -= _lhsp[2][0]*_lhsp[0][4]
		_rhs[2][m] -= _lhsp[2][0]*_rhs[0][m]
		m = 4
		fac1 = 1.0/_lhs[0][2]
		_lhs[0][3] *=  fac1
		_lhs[0][4] *=  fac1
		_rhs[0][m] *=  fac1
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3]
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4]
		_rhs[2][m] -= _lhs[2][0]*_rhs[0][m]
		# ---------------------------------------------------------------------
		# store computed lhs and prepare data for next iteration 
		# rhs is stored in a temp array such that write accesses are coalesced  
		# ---------------------------------------------------------------------
		for m in range(3, 5): 
			lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(m+4)))] = _lhsp[0][m]
			lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(m-1)))] = _lhs[0][m]
			rtmp[(i)+nx*((j)+ny*((k)+nz*(m)))] = _rhs[0][m]
			_rhs[0][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[2][m]

		for m in range(5):
			_lhsp[0][m] = _lhsp[1][m]
			_lhsp[1][m] = _lhsp[2][m]
			_lhs[0][m] = _lhs[1][m]
			_lhs[1][m] = _lhs[2][m]
	#END for k in range(nz-2)
	# ---------------------------------------------------------------------
	# and again the last two rows separately 
	# ---------------------------------------------------------------------
	k = nz-2
	m = 3
	fac1 = 1.0/_lhsp[0][2]
	_lhsp[0][3] *= fac1
	_lhsp[0][4] *= fac1
	_rhs[0][m] *= fac1
	_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3]
	_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4]
	_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m]
	m = 4
	fac1 = 1.0/_lhs[0][2]
	_lhs[0][3] *= fac1
	_lhs[0][4] *= fac1
	_rhs[0][m] *= fac1
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
	_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
	# ---------------------------------------------------------------------
	# scale the last row immediately 
	# ---------------------------------------------------------------------
	_rhs[1][3] /= _lhsp[1][2]
	_rhs[1][4] /= _lhs[1][2]
	# ---------------------------------------------------------------------
	# BACKSUBSTITUTION 
	# ---------------------------------------------------------------------
	for m in range(3):
		_rhs[0][m] -= lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((nz-2)+nz*(0)))]*_rhs[1][m]
	_rhs[0][3] -= _lhsp[0][3]*_rhs[1][3]
	_rhs[0][4] -= _lhs[0][3]*_rhs[1][4]
	for m in range(5):
		_rhs[2][m] = _rhs[1][m]
		_rhs[1][m] = _rhs[0][m]

	for k in range(nz-3, -1, -1):
		# ---------------------------------------------------------------------
		# the first three factors
		# ---------------------------------------------------------------------
		for m in range(3):
			_rhs[0][m] = rtmp[(i)+nx*((j)+ny*((k)+nz*(m)))]-lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(0)))]*_rhs[1][m]-lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(1)))]*_rhs[2][m]
		# ---------------------------------------------------------------------
		# and the remaining two
		# ---------------------------------------------------------------------
		_rhs[0][3] = rtmp[(i)+nx*((j)+ny*((k)+nz*(3)))]-lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(7)))]*_rhs[1][3]-lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(8)))]*_rhs[2][3]
		_rhs[0][4] = rtmp[(i)+nx*((j)+ny*((k)+nz*(4)))]-lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(2)))]*_rhs[1][4]-lhs[((i)-1)+(nx-2)*(((j)-1)+(ny-2)*((k)+nz*(3)))]*_rhs[2][4]
		if (k+2)<(nz-1):
			# ---------------------------------------------------------------------
			# do the block-diagonal inversion          
			# ---------------------------------------------------------------------
			xvel = us[(i)+(j)*nx+(k+2)*nx*ny]
			yvel = vs[(i)+(j)*nx+(k+2)*nx*ny]
			zvel = ws[(i)+(j)*nx+(k+2)*nx*ny]
			ac = speed[(i)+(j)*nx+(k+2)*nx*ny]
			uzik1 = u[(i)+nx*((j)+ny*((k+2)+nz*(0)))]
			t1 = (bt*uzik1)/ac*(_rhs[2][3]+_rhs[2][4])
			t2 = _rhs[2][2]+t1
			t3 = bt*uzik1*(_rhs[2][3]-_rhs[2][4])
			_rhs[2][4] = uzik1*(-xvel*_rhs[2][1]+yvel*_rhs[2][0])+qs[(i)+(j)*nx+(k+2)*nx*ny]*t2+c2iv*(ac*ac)*t1+zvel*t3
			_rhs[2][3] = zvel*t2+t3
			_rhs[2][2] = uzik1*_rhs[2][0]+yvel*t2
			_rhs[2][1] = -uzik1*_rhs[2][1]+xvel*t2
			_rhs[2][0] = t2

		for m in range(5):
			rhs[(m)+(i)*5+(j)*5*nx+(k+2)*5*nx*ny] = _rhs[2][m]
			_rhs[2][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[0][m]

	# ---------------------------------------------------------------------
	# do the block-diagonal inversion          
	# ---------------------------------------------------------------------
	xvel = us[(i)+(j)*nx+(1)*nx*ny]
	yvel = vs[(i)+(j)*nx+(1)*nx*ny]
	zvel = ws[(i)+(j)*nx+(1)*nx*ny]
	ac = speed[(i)+(j)*nx+(1)*nx*ny]
	uzik1 = u[(i)+nx*((j)+ny*((1)+nz*(0)))]
	t1 = (bt*uzik1)/ac*(_rhs[2][3]+_rhs[2][4])
	t2 = _rhs[2][2]+t1
	t3 = bt*uzik1*(_rhs[2][3]-_rhs[2][4])
	rhs[(4)+(i)*5+(j)*5*nx+(1)*5*nx*ny] = uzik1*(-xvel*_rhs[2][1]+yvel*_rhs[2][0])+qs[(i)+(j)*nx+(1)*nx*ny]*t2+c2iv*(ac*ac)*t1+zvel*t3
	rhs[(3)+(i)*5+(j)*5*nx+(1)*5*nx*ny] = zvel*t2+t3
	rhs[(2)+(i)*5+(j)*5*nx+(1)*5*nx*ny] = uzik1*_rhs[2][0]+yvel*t2
	rhs[(1)+(i)*5+(j)*5*nx+(1)*5*nx*ny] = -uzik1*_rhs[2][1]+xvel*t2
	rhs[(0)+(i)*5+(j)*5*nx+(1)*5*nx*ny] = t2
	for m in range(5):
		rhs[(m)+(i)*5+(j)*5*nx+(0)*5*nx*ny] = _rhs[1][m]
#END z_solve_gpu_kernel()


# ---------------------------------------------------------------------
# this function performs the solution of the approximate factorization
# step in the z-direction for all five matrix components
# simultaneously. The Thomas algorithm is employed to solve the
# systems for the z-lines. Boundary conditions are non-periodic
# ---------------------------------------------------------------------
def z_solve_gpu(rho_i_device,
				us_device,
				vs_device,
				ws_device,
				speed_device,
				qs_device,
				u_device,
				rhs_device,
				lhs_device,
				rhs_buffer_device):
	global const_z_solve_gpu_kernel_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Z_SOLVE)
	# #KERNEL Z SOLVE
	z_solve_blocks_per_grid = (1, ny)
	z_solve_threads_per_block = THREADS_PER_BLOCK_ON_Z_SOLVE
	if THREADS_PER_BLOCK_ON_Z_SOLVE != nx:
		z_solve_threads_per_block = nx

	z_solve_gpu_kernel[z_solve_blocks_per_grid, 
		z_solve_threads_per_block](rho_i_device,
								us_device,
								vs_device,
								ws_device,
								speed_device,
								qs_device,
								u_device,
								rhs_device,
								lhs_device,
								rhs_buffer_device,
								nx,
								ny,
								nz,
								const_z_solve_gpu_kernel_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Z_SOLVE)
#END z_solve_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64[:])')
def y_solve_gpu_kernel(rho_i, 
					vs, 
					speed, 
					rhs, 
					lhs, 
					rtmp, #rhstmp 
					nx, 
					ny, 
					nz,
					const_arr):
	rhoq = cuda.local.array(3, numba.float64)
	cv = cuda.local.array(3, numba.float64)
	_lhs = cuda.local.array((3, 5), numba.float64)
	_lhsp = cuda.local.array((3, 5), numba.float64)
	_rhs = cuda.local.array((3, 5), numba.float64)

	# coalesced
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
	k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1

	if (k>=(nz-1)) or (i>=(nx-1)):
		return
	
	#Constants
	dy1, dy3, dy5 = const_arr[0], const_arr[1], const_arr[2]
	dtty1, dtty2 = const_arr[3], const_arr[4]
	dymax, c2dtty1 = const_arr[5], const_arr[6]
	bt, c1c5, c3c4, con43 = const_arr[7], const_arr[8], const_arr[9], const_arr[10]
	comz1, comz4, comz5, comz6 = const_arr[11], const_arr[12], const_arr[13], const_arr[14]

	# ---------------------------------------------------------------------
	# computes the left hand side for the three y-factors   
	# ---------------------------------------------------------------------
	# first fill the lhs for the u-eigenvalue         
	# ---------------------------------------------------------------------
	_lhs[0][0] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((0)+ny*(4)))] = 0.0
	_lhs[0][1] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((0)+ny*(5)))] = 0.0
	_lhs[0][2] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((0)+ny*(6)))] = 1.0
	_lhs[0][3] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((0)+ny*(7)))] = 0.0
	_lhs[0][4] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((0)+ny*(8)))] = 0.0
	for j in range(3):
		fac1 = c3c4*rho_i[(i)+(j)*nx+(k)*nx*ny]
		rhoq[j] = max(max(max(dy3+con43*fac1, dy5+c1c5*fac1), dymax+fac1), dy1)
		cv[j] = vs[(i)+(j)*nx+(k)*nx*ny]

	_lhs[1][0] = 0.0
	_lhs[1][1] = -dtty2*cv[0]-dtty1*rhoq[0]
	_lhs[1][2] = 1.0+c2dtty1*rhoq[1]
	_lhs[1][3] = dtty2*cv[2]-dtty1*rhoq[2]
	_lhs[1][4] = 0.0
	_lhs[1][2] += comz5
	_lhs[1][3] -= comz4
	_lhs[1][4] += comz1
	for m in range(5): 
		lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((1)+ny*(m+4)))] = _lhs[1][m]
	rhoq[0] = rhoq[1]
	rhoq[1] = rhoq[2]
	cv[0] = cv[1]
	cv[1] = cv[2]
	for m in range(3): 
		_rhs[0][m] = rhs[(m)+(i)*5+(0)*5*nx+(k)*5*nx*ny]
		_rhs[1][m] = rhs[(m)+(i)*5+(1)*5*nx+(k)*5*nx*ny]

	# ---------------------------------------------------------------------
	# FORWARD ELIMINATION  
	# ---------------------------------------------------------------------
	for j in range(ny-2): 
		# ---------------------------------------------------------------------
		# first fill the lhs for the u-eigenvalue         
		# ---------------------------------------------------------------------
		if (j+2)==(ny-1):
			_lhs[2][0] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(4)))] = 0.0
			_lhs[2][1] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(5)))] = 0.0
			_lhs[2][2] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(6)))] = 1.0
			_lhs[2][3] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(7)))] = 0.0
			_lhs[2][4] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(8)))] = 0.0
		else:
			fac1 = c3c4*rho_i[(i)+(j+3)*nx+(k)*nx*ny]
			rhoq[2] = max(max(max(dy3+con43*fac1, dy5+c1c5*fac1), dymax+fac1), dy1)
			cv[2] = vs[(i)+(j+3)*nx+(k)*nx*ny]
			_lhs[2][0] = 0.0
			_lhs[2][1] = -dtty2*cv[0]-dtty1*rhoq[0]
			_lhs[2][2] = 1.0+c2dtty1*rhoq[1]
			_lhs[2][3] = dtty2*cv[2]-dtty1*rhoq[2]
			_lhs[2][4] = 0.0
			# ---------------------------------------------------------------------
			# add fourth order dissipation                             
			# ---------------------------------------------------------------------
			if (j+2)==(2):
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
				_lhs[2][4] += comz1
			elif ((j+2)>=(3)) and ((j+2)<(ny-3)):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
				_lhs[2][4] += comz1
			elif (j+2)==(ny-3):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
			elif (j+2)==(ny-2):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz5

			# ---------------------------------------------------------------------
			# store computed lhs for later reuse                           
			# ---------------------------------------------------------------------
			for m in range(5):
				lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(m+4)))] = _lhs[2][m]
			rhoq[0] = rhoq[1]
			rhoq[1] = rhoq[2]
			cv[0] = cv[1]
			cv[1] = cv[2]

		# ---------------------------------------------------------------------
		# load rhs values for current iteration                          
		# ---------------------------------------------------------------------
		for m in range(3):
			_rhs[2][m] = rhs[(m)+(i)*5+(j+2)*5*nx+(k)*5*nx*ny]
		# ---------------------------------------------------------------------
		# perform current iteration                         
		# ---------------------------------------------------------------------
		fac1 = 1.0/_lhs[0][2]
		_lhs[0][3] *= fac1
		_lhs[0][4] *= fac1
		for m in range(3):
			_rhs[0][m] *= fac1
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
		for m in range(3):
			_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3]
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4]
		for m in range(3):
			_rhs[2][m] -= _lhs[2][0]*_rhs[0][m]
		# ---------------------------------------------------------------------
		# store computed lhs and prepare data for next iteration
		# rhs is stored in a temp array such that write accesses are coalesced                  
		# ---------------------------------------------------------------------
		lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(0)))] = _lhs[0][3]
		lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(1)))] = _lhs[0][4]
		for m in range(5):
			_lhs[0][m] = _lhs[1][m]
			_lhs[1][m] = _lhs[2][m]

		for m in range(3):
			rtmp[(i)+nx*((k)+nz*((j)+ny*(m)))] = _rhs[0][m]
			_rhs[0][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[2][m]
	#END for j in range(ny-2):
	# ---------------------------------------------------------------------
	# the last two rows in this zone are a bit different, 
	# since they do not have two more rows available for the  
	# elimination of off-diagonal entries              
	# ---------------------------------------------------------------------
	j = ny-2
	fac1 = 1.0/_lhs[0][2]
	_lhs[0][3] *= fac1
	_lhs[0][4] *= fac1
	for m in range(3):
		_rhs[0][m] *= fac1
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
	for m in range(3):
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
	# ---------------------------------------------------------------------
	# scale the last row immediately 
	# ---------------------------------------------------------------------
	fac1 = 1.0/_lhs[1][2]
	for m in range(3):
		_rhs[1][m] *= fac1
	lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((ny-2)+ny*(0)))] = _lhs[0][3]
	lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((ny-2)+ny*(1)))] = _lhs[0][4]
	# ---------------------------------------------------------------------
	# do the u+c and the u-c factors                 
	# ---------------------------------------------------------------------
	for j in range(3):
		cv[j] = speed[(i)+(j)*nx+(k)*nx*ny]
	for m in range(5):
		_lhsp[0][m] = _lhs[0][m] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((0)+ny*(m+4)))]
		_lhsp[1][m] = _lhs[1][m] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((1)+ny*(m+4)))]

	_lhsp[1][1] -= dtty2*cv[0]
	_lhsp[1][3] += dtty2*cv[2]
	_lhs[1][1] += dtty2*cv[0]
	_lhs[1][3] -= dtty2*cv[2]
	cv[0] = cv[1]
	cv[1] = cv[2]
	_rhs[0][3] = rhs[(3)+(i)*5+(0)*5*nx+(k)*5*nx*ny]
	_rhs[0][4] = rhs[(4)+(i)*5+(0)*5*nx+(k)*5*nx*ny]
	_rhs[1][3] = rhs[(3)+(i)*5+(1)*5*nx+(k)*5*nx*ny]
	_rhs[1][4] = rhs[(4)+(i)*5+(1)*5*nx+(k)*5*nx*ny]
	for j in range(ny-2):
		for m in range(5):
			_lhsp[2][m] = _lhs[2][m] = lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j+2)+ny*(m+4)))]

		_rhs[2][3] = rhs[(3)+(i)*5+(j+2)*5*nx+(k)*5*nx*ny]
		_rhs[2][4] = rhs[(4)+(i)*5+(j+2)*5*nx+(k)*5*nx*ny]
		if (j+2)<(ny-1):
			cv[2] = speed[(i)+(j+3)*nx+(k)*nx*ny]
			_lhsp[2][1] -= dtty2*cv[0]
			_lhsp[2][3] += dtty2*cv[2]
			_lhs[2][1] += dtty2*cv[0]
			_lhs[2][3] -= dtty2*cv[2]
			cv[0] = cv[1]
			cv[1] = cv[2]

		fac1 = 1.0/_lhsp[0][2]
		m = 3
		_lhsp[0][3] *= fac1
		_lhsp[0][4] *= fac1
		_rhs[0][m] *= fac1
		_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3]
		_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4]
		_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m]
		_lhsp[2][1] -= _lhsp[2][0]*_lhsp[0][3]
		_lhsp[2][2] -= _lhsp[2][0]*_lhsp[0][4]
		_rhs[2][m] -= _lhsp[2][0]*_rhs[0][m]
		m = 4
		fac1 = 1.0/_lhs[0][2]
		_lhs[0][3] *= fac1
		_lhs[0][4] *= fac1
		_rhs[0][m] *= fac1
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3]
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4]
		_rhs[2][m] -= _lhs[2][0]*_rhs[0][m]
		# ---------------------------------------------------------------------
		# store computed lhs and prepare data for next iteration 
		# rhs is stored in a temp array such that write accesses are coalesced  
		# ---------------------------------------------------------------------
		for m in range(3, 5):
			lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(m+4)))] = _lhsp[0][m]
			lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(m-1)))] = _lhs[0][m]
			rtmp[(i)+nx*((k)+nz*((j)+ny*(m)))] = _rhs[0][m]
			_rhs[0][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[2][m]

		for m in range(5):
			_lhsp[0][m] = _lhsp[1][m]
			_lhsp[1][m] = _lhsp[2][m]
			_lhs[0][m] = _lhs[1][m]
			_lhs[1][m] = _lhs[2][m]
	#END for j in range(ny-2):
	# ---------------------------------------------------------------------
	# and again the last two rows separately 
	# ---------------------------------------------------------------------
	j = ny-2
	m = 3
	fac1 = 1.0/_lhsp[0][2]
	_lhsp[0][3] *= fac1
	_lhsp[0][4] *= fac1
	_rhs[0][m] *= fac1
	_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3]
	_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4]
	_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m]
	m = 4
	fac1 = 1.0/_lhs[0][2]
	_lhs[0][3] *= fac1
	_lhs[0][4] *= fac1
	_rhs[0][m] *= fac1
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
	_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
	# ---------------------------------------------------------------------
	# scale the last row immediately 
	# ---------------------------------------------------------------------
	_rhs[1][3] /= _lhsp[1][2]
	_rhs[1][4] /= _lhs[1][2]
	# ---------------------------------------------------------------------
	# BACKSUBSTITUTION 
	# ---------------------------------------------------------------------
	for m in range(3):
		_rhs[0][m] -= lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((ny-2)+ny*(0)))]*_rhs[1][m]
	_rhs[0][3] -= _lhsp[0][3]*_rhs[1][3]
	_rhs[0][4] -= _lhs[0][3]*_rhs[1][4]
	for m in range(5):
		_rhs[2][m] = _rhs[1][m]
		_rhs[1][m] = _rhs[0][m]

	for j in range(ny-3, -1, -1):
		# ---------------------------------------------------------------------
		# the first three factors
		# ---------------------------------------------------------------------
		for m in range(3):
			_rhs[0][m] = rtmp[(i)+nx*((k)+nz*((j)+ny*(m)))]-lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(0)))]*_rhs[1][m]-lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(1)))]*_rhs[2][m]
		# ---------------------------------------------------------------------
		# and the remaining two
		# ---------------------------------------------------------------------
		_rhs[0][3] = rtmp[(i)+nx*((k)+nz*((j)+ny*(3)))]-lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(7)))]*_rhs[1][3]-lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(8)))]*_rhs[2][3]
		_rhs[0][4] = rtmp[(i)+nx*((k)+nz*((j)+ny*(4)))]-lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(2)))]*_rhs[1][4]-lhs[((i)-1)+(nx-2)*(((k)-1)+(nz-2)*((j)+ny*(3)))]*_rhs[2][4]
		if (j+2)<(ny-1):
			# ---------------------------------------------------------------------
			# do the block-diagonal inversion          
			# ---------------------------------------------------------------------
			r1 = _rhs[2][0]
			r2 = _rhs[2][1]
			r3 = _rhs[2][2]
			r4 = _rhs[2][3]
			r5 = _rhs[2][4]
			t1 = bt*r1
			t2 = 0.5*(r4+r5)
			_rhs[2][0] = bt*(r4-r5)
			_rhs[2][1] = -r3
			_rhs[2][2] = r2
			_rhs[2][3] = -t1+t2
			_rhs[2][4] = t1+t2

		for m in range(5):
			rhs[(m)+(i)*5+(j+2)*5*nx+(k)*5*nx*ny] = _rhs[2][m]
			_rhs[2][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[0][m]

	# ---------------------------------------------------------------------
	# do the block-diagonal inversion          
	# ---------------------------------------------------------------------
	t1 = bt*_rhs[2][0]
	t2 = 0.5*(_rhs[2][3]+_rhs[2][4])
	rhs[(0)+(i)*5+(1)*5*nx+(k)*5*nx*ny] = bt*(_rhs[2][3]-_rhs[2][4])
	rhs[(1)+(i)*5+(1)*5*nx+(k)*5*nx*ny] = -_rhs[2][2]
	rhs[(2)+(i)*5+(1)*5*nx+(k)*5*nx*ny] = _rhs[2][1]
	rhs[(3)+(i)*5+(1)*5*nx+(k)*5*nx*ny] = -t1+t2
	rhs[(4)+(i)*5+(1)*5*nx+(k)*5*nx*ny] = t1+t2
	for m in range(5):
		rhs[(m)+(i)*5+(0)*5*nx+(k)*5*nx*ny] = _rhs[1][m]
#END y_solve_gpu_kernel()


# ---------------------------------------------------------------------
# this function performs the solution of the approximate factorization
# step in the y-direction for all five matrix components
# simultaneously. the thomas algorithm is employed to solve the
# systems for the y-lines. boundary conditions are non-periodic
# ---------------------------------------------------------------------
def y_solve_gpu(rho_i_device, 
				vs_device, 
				speed_device, 
				rhs_device, 
				lhs_device, 
				rhs_buffer_device):
	global const_y_solve_gpu_kernel_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Y_SOLVE)
	# #KERNEL Y SOLVE
	y_solve_blocks_per_grid = (1, nz)
	y_solve_threads_per_block = THREADS_PER_BLOCK_ON_Y_SOLVE
	if THREADS_PER_BLOCK_ON_Y_SOLVE != nx:
		y_solve_threads_per_block = nx

	y_solve_gpu_kernel[y_solve_blocks_per_grid,
		y_solve_threads_per_block](rho_i_device, 
								vs_device, 
								speed_device, 
								rhs_device, 
								lhs_device, 
								rhs_buffer_device, 
								nx, 
								ny, 
								nz,
								const_y_solve_gpu_kernel_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Y_SOLVE)
#END y_solve_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64[:])')
def x_solve_gpu_kernel(rho_i, 
					us, 
					speed, 
					rhs, 
					lhs, 
					rtmp, #rhstmp
					nx, 
					ny, 
					nz,
					const_arr):
	rhon = cuda.local.array(3, numba.float64)
	cv = cuda.local.array(3, numba.float64)
	_lhs = cuda.local.array((3, 5), numba.float64)
	_lhsp = cuda.local.array((3, 5), numba.float64)
	_rhs = cuda.local.array((3, 5), numba.float64)
	
	# coalesced
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
	k = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1

	if (k>=nz-1) or (j>=ny-1):
		return
	
	#Constants
	dx1, dx2, dx5 = const_arr[0], const_arr[1], const_arr[2]
	dttx1, dttx2 = const_arr[3], const_arr[4]
	dxmax, c2dttx1 = const_arr[5], const_arr[6]
	bt, c1c5, c3c4, con43 = const_arr[7], const_arr[8], const_arr[9], const_arr[10]
	comz1, comz4, comz5, comz6 = const_arr[11], const_arr[12], const_arr[13], const_arr[14]

	# ---------------------------------------------------------------------
	# computes the left hand side for the three x-factors  
	# ---------------------------------------------------------------------
	# first fill the lhs for the u-eigenvalue                   
	# ---------------------------------------------------------------------
	_lhs[0][0] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((0)+nx*(4)))] = 0.0
	_lhs[0][1] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((0)+nx*(5)))] = 0.0
	_lhs[0][2] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((0)+nx*(6)))] = 1.0
	_lhs[0][3] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((0)+nx*(7)))] = 0.0
	_lhs[0][4] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((0)+nx*(8)))] = 0.0
	for i in range(3):
		fac1 = c3c4*rho_i[(i)+(j)*nx+(k)*nx*ny]
		rhon[i] = max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), dx1)
		cv[i] = us[(i)+(j)*nx+(k)*nx*ny]

	_lhs[1][0] = 0.0
	_lhs[1][1] = -dttx2*cv[0]-dttx1*rhon[0]
	_lhs[1][2] = 1.0+c2dttx1*rhon[1]
	_lhs[1][3] = dttx2*cv[2]-dttx1*rhon[2]
	_lhs[1][4] = 0.0
	_lhs[1][2] += comz5
	_lhs[1][3] -= comz4
	_lhs[1][4] += comz1
	for m in range(5):
		lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((1)+nx*(m+4)))] = _lhs[1][m]
	rhon[0] = rhon[1]
	rhon[1] = rhon[2]
	cv[0] = cv[1]
	cv[1] = cv[2]
	for m in range(3):
		_rhs[0][m] = rhs[(m)+(0)*5+(j)*5*nx+(k)*5*nx*ny]
		_rhs[1][m] = rhs[(m)+(1)*5+(j)*5*nx+(k)*5*nx*ny]

	# ---------------------------------------------------------------------
	# FORWARD ELIMINATION  
	# ---------------------------------------------------------------------
	# perform the thomas algorithm first, FORWARD ELIMINATION     
	# ---------------------------------------------------------------------
	for i in range(nx-2):
		# ---------------------------------------------------------------------
		# first fill the lhs for the u-eigenvalue                   
		# ---------------------------------------------------------------------
		if (i+2)==(nx-1):
			_lhs[2][0] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(4)))] = 0.0
			_lhs[2][1] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(5)))] = 0.0
			_lhs[2][2] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(6)))] = 1.0
			_lhs[2][3] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(7)))] = 0.0
			_lhs[2][4] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(8)))] = 0.0
		else:
			fac1 = c3c4*rho_i[(i+3)+(j)*nx+(k)*nx*ny]
			rhon[2] = max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), dx1)
			cv[2] = us[(i+3)+(j)*nx+(k)*nx*ny]
			_lhs[2][0] = 0.0
			_lhs[2][1] = -dttx2*cv[0]-dttx1*rhon[0]
			_lhs[2][2] = 1.0+c2dttx1*rhon[1]
			_lhs[2][3] = dttx2*cv[2]-dttx1*rhon[2]
			_lhs[2][4] = 0.0
			# ---------------------------------------------------------------------
			# add fourth order dissipation                             
			# ---------------------------------------------------------------------
			if (i+2)==(2):
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
				_lhs[2][4] += comz1
			elif (i+2>=3) and (i+2<nx-3):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
				_lhs[2][4] += comz1
			elif (i+2)==(nx-3):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz6
				_lhs[2][3] -= comz4
			elif (i+2)==(nx-2):
				_lhs[2][0] += comz1
				_lhs[2][1] -= comz4
				_lhs[2][2] += comz5

			# ---------------------------------------------------------------------
			# store computed lhs for later reuse
			# ---------------------------------------------------------------------
			for m in range(5):
				lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(m+4)))] = _lhs[2][m]
			rhon[0] = rhon[1]
			rhon[1] = rhon[2]
			cv[0] = cv[1]
			cv[1] = cv[2]

		# ---------------------------------------------------------------------
		# load rhs values for current iteration
		# ---------------------------------------------------------------------
		for m in range(3): 
			_rhs[2][m] = rhs[(m)+(i+2)*5+(j)*5*nx+(k)*5*nx*ny]
		# ---------------------------------------------------------------------
		# perform current iteration
		# ---------------------------------------------------------------------
		fac1 = 1.0/_lhs[0][2]
		_lhs[0][3] *= fac1
		_lhs[0][4] *= fac1
		for m in range(3):
			_rhs[0][m] *= fac1
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
		for m in range(3):
			_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3]
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4]
		for m in range(3):
			_rhs[2][m] -= _lhs[2][0]*_rhs[0][m]
		# ---------------------------------------------------------------------
		# store computed lhs and prepare data for next iteration 
		# rhs is stored in a temp array such that write accesses are coalesced 
		# ---------------------------------------------------------------------
		lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(0)))] = _lhs[0][3]
		lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(1)))] = _lhs[0][4]
		for m in range(5):
			_lhs[0][m] = _lhs[1][m]
			_lhs[1][m] = _lhs[2][m]

		for m in range(3):
			rtmp[(j)+ny*((k)+nz*((i)+nx*(m)))] = _rhs[0][m]
			_rhs[0][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[2][m]
	#END for i in range(nx-2)
	
	# ---------------------------------------------------------------------
	# the last two rows in this zone are a bit different,  
	# since they do not have two more rows available for the
	# elimination of off-diagonal entries    
	# ---------------------------------------------------------------------
	i = nx-2
	fac1 = 1.0/_lhs[0][2]
	_lhs[0][3] *= fac1
	_lhs[0][4] *= fac1
	for m in range(3):
		_rhs[0][m] *= fac1
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
	for m in range(3):
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
	# ---------------------------------------------------------------------
	# scale the last row immediately 
	# ---------------------------------------------------------------------
	fac1 = 1.0/_lhs[1][2]
	for m in range(3):
		_rhs[1][m] *= fac1
	lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((nx-2)+nx*(0)))] = _lhs[0][3]
	lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((nx-2)+nx*(1)))] = _lhs[0][4]
	# ---------------------------------------------------------------------
	# subsequently, fill the other factors (u+c), (u-c)
	# ---------------------------------------------------------------------
	for i in range(3):
		cv[i] = speed[(i)+(j)*nx+(k)*nx*ny]
	for m in range(5):
		_lhsp[0][m] = _lhs[0][m] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((0)+nx*(m+4)))]
		_lhsp[1][m] = _lhs[1][m] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((1)+nx*(m+4)))]

	_lhsp[1][1] -= dttx2*cv[0]
	_lhsp[1][3] += dttx2*cv[2]
	_lhs[1][1] += dttx2*cv[0]
	_lhs[1][3] -= dttx2*cv[2]
	cv[0] = cv[1]
	cv[1] = cv[2]
	_rhs[0][3] = rhs[(3)+(0)*5+(j)*5*nx+(k)*5*nx*ny]
	_rhs[0][4] = rhs[(4)+(0)*5+(j)*5*nx+(k)*5*nx*ny]
	_rhs[1][3] = rhs[(3)+(1)*5+(j)*5*nx+(k)*5*nx*ny]
	_rhs[1][4] = rhs[(4)+(1)*5+(j)*5*nx+(k)*5*nx*ny]
	# ---------------------------------------------------------------------
	# do the u+c and the u-c factors                 
	# ---------------------------------------------------------------------
	for i in range(nx-2):
		# first, fill the other factors (u+c), (u-c) 
		# ---------------------------------------------------------------------
		for m in range(5):
			_lhsp[2][m] = _lhs[2][m] = lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i+2)+nx*(m+4)))]

		_rhs[2][3] = rhs[(3)+(i+2)*5+(j)*5*nx+(k)*5*nx*ny]
		_rhs[2][4] = rhs[(4)+(i+2)*5+(j)*5*nx+(k)*5*nx*ny]
		if (i+2)<(nx-1):
			cv[2] = speed[(i+3)+(j)*nx+(k)*nx*ny]
			_lhsp[2][1] -= dttx2*cv[0]
			_lhsp[2][3] += dttx2*cv[2]
			_lhs[2][1] += dttx2*cv[0]
			_lhs[2][3] -= dttx2*cv[2]
			cv[0] = cv[1]
			cv[1] = cv[2]

		m = 3
		fac1 = 1.0/_lhsp[0][2]
		_lhsp[0][3] *= fac1
		_lhsp[0][4] *= fac1
		_rhs[0][m] *= fac1
		_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3]
		_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4]
		_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m]
		_lhsp[2][1] -= _lhsp[2][0]*_lhsp[0][3]
		_lhsp[2][2] -= _lhsp[2][0]*_lhsp[0][4]
		_rhs[2][m] -= _lhsp[2][0]*_rhs[0][m]
		m = 4
		fac1 = 1.0/_lhs[0][2]
		_lhs[0][3] *= fac1
		_lhs[0][4] *= fac1
		_rhs[0][m] *= fac1
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3]
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4]
		_rhs[2][m] -= _lhs[2][0]*_rhs[0][m]
		# ---------------------------------------------------------------------
		# store computed lhs and prepare data for next iteration 
		# rhs is stored in a temp array such that write accesses are coalesced  
		# ---------------------------------------------------------------------
		for m in range(3, 5):
			lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(m+4)))] = _lhsp[0][m]
			lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(m-1)))] = _lhs[0][m]
			rtmp[(j)+ny*((k)+nz*((i)+nx*(m)))] = _rhs[0][m]
			_rhs[0][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[2][m]

		for m in range(5):
			_lhsp[0][m] = _lhsp[1][m]
			_lhsp[1][m] = _lhsp[2][m]
			_lhs[0][m] = _lhs[1][m]
			_lhs[1][m] = _lhs[2][m]
	#END for i in range(nx-2)
	# ---------------------------------------------------------------------
	# and again the last two rows separately 
	# ---------------------------------------------------------------------
	i = nx-2
	m = 3
	fac1 = 1.0/_lhsp[0][2]
	_lhsp[0][3] *= fac1
	_lhsp[0][4] *= fac1
	_rhs[0][m] *= fac1
	_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3]
	_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4]
	_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m]
	m = 4
	fac1 = 1.0/_lhs[0][2]
	_lhs[0][3] *= fac1
	_lhs[0][4] *= fac1
	_rhs[0][m] *= fac1
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3]
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4]
	_rhs[1][m] -= _lhs[1][1]*_rhs[0][m]
	# ---------------------------------------------------------------------
	# scale the last row immediately 
	# ---------------------------------------------------------------------
	_rhs[1][3] /= _lhsp[1][2]
	_rhs[1][4] /= _lhs[1][2]
	# ---------------------------------------------------------------------
	# BACKSUBSTITUTION 
	# ---------------------------------------------------------------------
	for m in range(3):
		_rhs[0][m] -= lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((nx-2)+nx*(0)))]*_rhs[1][m]
	_rhs[0][3] -= _lhsp[0][3]*_rhs[1][3]
	_rhs[0][4] -= _lhs[0][3]*_rhs[1][4]
	for m in range(5):
		_rhs[2][m] = _rhs[1][m]
		_rhs[1][m] = _rhs[0][m]

	for i in range(nx-3, -1, -1):
		# ---------------------------------------------------------------------
		# the first three factors
		# ---------------------------------------------------------------------
		for m in range(3):
			_rhs[0][m] = rtmp[(j)+ny*((k)+nz*((i)+nx*(m)))]-lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(0)))]*_rhs[1][m]-lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(1)))]*_rhs[2][m]
		# ---------------------------------------------------------------------
		# and the remaining two
		# ---------------------------------------------------------------------
		_rhs[0][3] = rtmp[(j)+ny*((k)+nz*((i)+nx*(3)))]-lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(7)))]*_rhs[1][3]-lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(8)))]*_rhs[2][3]
		_rhs[0][4] = rtmp[(j)+ny*((k)+nz*((i)+nx*(4)))]-lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(2)))]*_rhs[1][4]-lhs[((j)-1)+(ny-2)*(((k)-1)+(nz-2)*((i)+nx*(3)))]*_rhs[2][4]
		if i+2<nx-1:
			# ---------------------------------------------------------------------
			# do the block-diagonal inversion          
			# ---------------------------------------------------------------------
			r1 = _rhs[2][0]
			r2 = _rhs[2][1]
			r3 = _rhs[2][2]
			r4 = _rhs[2][3]
			r5 = _rhs[2][4]
			t1 = bt*r3
			t2 = 0.5*(r4+r5)
			_rhs[2][0] = -r2
			_rhs[2][1] = r1
			_rhs[2][2] = bt*(r4-r5)
			_rhs[2][3] = -t1+t2
			_rhs[2][4] = t1+t2

		for m in range(5):
			rhs[(m)+(i+2)*5+(j)*5*nx+(k)*5*nx*ny] = _rhs[2][m]
			_rhs[2][m] = _rhs[1][m]
			_rhs[1][m] = _rhs[0][m]

	# ---------------------------------------------------------------------
	# do the block-diagonal inversion          
	# ---------------------------------------------------------------------
	t1 = bt*_rhs[2][2]
	t2 = 0.5*(_rhs[2][3]+_rhs[2][4])
	rhs[(0)+(1)*5+(j)*5*nx+(k)*5*nx*ny] = -_rhs[2][1]
	rhs[(1)+(1)*5+(j)*5*nx+(k)*5*nx*ny] = _rhs[2][0]
	rhs[(2)+(1)*5+(j)*5*nx+(k)*5*nx*ny] = bt*(_rhs[2][3]-_rhs[2][4])
	rhs[(3)+(1)*5+(j)*5*nx+(k)*5*nx*ny] = -t1+t2
	rhs[(4)+(1)*5+(j)*5*nx+(k)*5*nx*ny] = t1+t2
	for m in range(5):
		rhs[(m)+(0)*5+(j)*5*nx+(k)*5*nx*ny] = _rhs[1][m]
#END x_solve_gpu_kernel()


# ---------------------------------------------------------------------
# this function performs the solution of the approximate factorization
# step in the x-direction for all five matrix components
# simultaneously. the thomas algorithm is employed to solve the
# systems for the x-lines. boundary conditions are non-periodic
# ---------------------------------------------------------------------
def x_solve_gpu(rho_i_device, 
				us_device,
				speed_device,
				rhs_device,
				lhs_device,
				rhs_buffer_device):
	global const_x_solve_gpu_kernel_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_X_SOLVE)
	# #KERNEL X SOLVE
	x_solve_blocks_per_grid = (1, nz)
	x_solve_threads_per_block = THREADS_PER_BLOCK_ON_X_SOLVE
	if THREADS_PER_BLOCK_ON_X_SOLVE != ny:
		x_solve_threads_per_block = ny

	x_solve_gpu_kernel[x_solve_blocks_per_grid,
		x_solve_threads_per_block](rho_i_device, 
								us_device,
								speed_device,
								rhs_device,
								lhs_device,
								rhs_buffer_device,
								nx, 
								ny, 
								nz,
								const_x_solve_gpu_kernel_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_X_SOLVE)
#END x_solve_gpu()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64, float64)')
def txinvr_gpu_kernel(rho_i, 
					us, 
					vs, 
					ws, 
					speed, 
					qs, 
					rhs, 
					nx, 
					ny, 
					nz,
					c2, bt):
	i_j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j_k % nx
	j = int(i_j_k / nx) % ny
	k = int(i_j_k / (nx * ny))

	if i_j_k >= (nx*ny*nz):
		return

	ru1 = rho_i[(i)+(j)*nx+(k)*nx*ny]
	uu = us[(i)+(j)*nx+(k)*nx*ny]
	vv = vs[(i)+(j)*nx+(k)*nx*ny]
	ww = ws[(i)+(j)*nx+(k)*nx*ny]
	ac = speed[(i)+(j)*nx+(k)*nx*ny]
	ac2inv = 1.0 / (ac*ac)
	r1 = rhs[(0)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	r2 = rhs[(1)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	r3 = rhs[(2)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	r4 = rhs[(3)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	r5 = rhs[(4)+(i)*5+(j)*5*nx+(k)*5*nx*ny]
	t1 = c2*ac2inv*(qs[(i)+(j)*nx+(k)*nx*ny]*r1-uu*r2-vv*r3-ww*r4+r5)
	t2 = bt*ru1*(uu*r1-r2)
	t3 = (bt*ru1*ac)*t1
	rhs[(0)+(i)*5+(j)*5*nx+(k)*5*nx*ny] = r1-t1
	rhs[(1)+(i)*5+(j)*5*nx+(k)*5*nx*ny] = -ru1*(ww*r1-r4)
	rhs[(2)+(i)*5+(j)*5*nx+(k)*5*nx*ny] = ru1*(vv*r1-r3)
	rhs[(3)+(i)*5+(j)*5*nx+(k)*5*nx*ny] = -t2+t3
	rhs[(4)+(i)*5+(j)*5*nx+(k)*5*nx*ny] = t2+t3
#END txinvr_gpu_kernel()


# ---------------------------------------------------------------------
# block-diagonal matrix-vector multiplication                  
# ---------------------------------------------------------------------
def txinvr_gpu(rho_i_device, 
			us_device, 
			vs_device, 
			ws_device, 
			speed_device, 
			qs_device, 
			rhs_device):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_TXINVR)
	# #KERNEL TXINVR
	txinvr_workload = nx * ny * nz
	txinvr_threads_per_block = THREADS_PER_BLOCK_ON_TXINVR
	txinvr_blocks_per_grid = math.ceil(txinvr_workload / txinvr_threads_per_block)

	txinvr_gpu_kernel[txinvr_blocks_per_grid, 
		txinvr_threads_per_block](rho_i_device, 
								us_device, 
								vs_device, 
								ws_device, 
								speed_device, 
								qs_device, 
								rhs_device, 
								nx, 
								ny, 
								nz,
								c2, bt)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_TXINVR)
#END txinvr_gpu


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64[:])')
def compute_rhs_gpu_kernel_2(rho_i, 
							us, 
							vs, 
							ws, 
							qs, 
							square, 
							rhs, 
							forcing, 
							u, 
							nx, 
							ny, 
							nz,
							const_arr):
	k = cuda.blockIdx.y
	j = cuda.blockIdx.x
	i = cuda.threadIdx.x

	rtmp = cuda.local.array(5, numba.float64)
	# ---------------------------------------------------------------------
	# copy the exact forcing term to the right hand side;  because 
	# this forcing term is known, we can store it on the whole grid
	# including the boundary                   
	# ---------------------------------------------------------------------
	for m in range(5):
		rtmp[m] = forcing[(i)+nx*((j)+ny*((k)+nz*(m)))]
	# ---------------------------------------------------------------------
	# compute xi-direction fluxes 
	# ---------------------------------------------------------------------
	if k>=1 and k<nz-1 and j>=1 and j<ny-1 and i>=1 and i<nx-1:
		#Constants
		dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1 = const_arr[0], const_arr[1], const_arr[2], const_arr[3], const_arr[4]
		dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1 = const_arr[5], const_arr[6], const_arr[7], const_arr[8], const_arr[9]
		dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1 = const_arr[10], const_arr[11], const_arr[12], const_arr[13], const_arr[14]
		tx2, ty2, tz2 = const_arr[15], const_arr[16], const_arr[17]
		xxcon2, xxcon3, xxcon4, xxcon5 = const_arr[18], const_arr[19], const_arr[20], const_arr[21]
		yycon2, yycon3, yycon4, yycon5 = const_arr[22], const_arr[23], const_arr[24], const_arr[25]
		zzcon2, zzcon3, zzcon4, zzcon5 = const_arr[26], const_arr[27], const_arr[28], const_arr[29]
		c1, c2, con43, dssp, dt = const_arr[30], const_arr[31], const_arr[32], const_arr[33], const_arr[34]
		
		uijk = us[(i)+(j)*nx+(k)*nx*ny]
		up1 = us[(i+1)+(j)*nx+(k)*nx*ny]
		um1 = us[(i-1)+(j)*nx+(k)*nx*ny]
		rtmp[0] = rtmp[0]+dx1tx1*(u[(i+1)+nx*((j)+ny*((k)+nz*(0)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(0)))]+u[(i-1)+nx*((j)+ny*((k)+nz*(0)))])-tx2*(u[(i+1)+nx*((j)+ny*((k)+nz*(1)))]-u[(i-1)+nx*((j)+ny*((k)+nz*(1)))])
		rtmp[1] = rtmp[1]+dx2tx1*(u[(i+1)+nx*((j)+ny*((k)+nz*(1)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(1)))]+u[(i-1)+nx*((j)+ny*((k)+nz*(1)))])+xxcon2*con43*(up1-2.0*uijk+um1)-tx2*(u[(i+1)+nx*((j)+ny*((k)+nz*(1)))]*up1-u[(i-1)+nx*((j)+ny*((k)+nz*(1)))]*um1+(u[(i+1)+nx*((j)+ny*((k)+nz*(4)))]-square[(i+1)+(j)*nx+(k)*nx*ny]-u[(i-1)+nx*((j)+ny*((k)+nz*(4)))]+square[(i-1)+(j)*nx+(k)*nx*ny])*c2)
		rtmp[2] = rtmp[2]+dx3tx1*(u[(i+1)+nx*((j)+ny*((k)+nz*(2)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(2)))]+u[(i-1)+nx*((j)+ny*((k)+nz*(2)))])+xxcon2*(vs[(i+1)+(j)*nx+(k)*nx*ny]-2.0*vs[(i)+(j)*nx+(k)*nx*ny]+vs[(i-1)+(j)*nx+(k)*nx*ny])-tx2*(u[(i+1)+nx*((j)+ny*((k)+nz*(2)))]*up1-u[(i-1)+nx*((j)+ny*((k)+nz*(2)))]*um1)
		rtmp[3] = rtmp[3]+dx4tx1*(u[(i+1)+nx*((j)+ny*((k)+nz*(3)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(3)))]+u[(i-1)+nx*((j)+ny*((k)+nz*(3)))])+xxcon2*(ws[(i+1)+(j)*nx+(k)*nx*ny]-2.0*ws[(i)+(j)*nx+(k)*nx*ny]+ws[(i-1)+(j)*nx+(k)*nx*ny])-tx2*(u[(i+1)+nx*((j)+ny*((k)+nz*(3)))]*up1-u[(i-1)+nx*((j)+ny*((k)+nz*(3)))]*um1)
		rtmp[4] = rtmp[4]+dx5tx1*(u[(i+1)+nx*((j)+ny*((k)+nz*(4)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(4)))]+u[(i-1)+nx*((j)+ny*((k)+nz*(4)))])+xxcon3*(qs[(i+1)+(j)*nx+(k)*nx*ny]-2.0*qs[(i)+(j)*nx+(k)*nx*ny]+qs[(i-1)+(j)*nx+(k)*nx*ny])+ xxcon4*(up1*up1-2.0*uijk*uijk+um1*um1)+xxcon5*(u[(i+1)+nx*((j)+ny*((k)+nz*(4)))]*rho_i[(i+1)+(j)*nx+(k)*nx*ny]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(4)))]*rho_i[(i)+(j)*nx+(k)*nx*ny]+u[(i-1)+nx*((j)+ny*((k)+nz*(4)))]*rho_i[(i-1)+(j)*nx+(k)*nx*ny])-tx2*((c1*u[(i+1)+nx*((j)+ny*((k)+nz*(4)))]-c2*square[(i+1)+(j)*nx+(k)*nx*ny])*up1-(c1*u[(i-1)+nx*((j)+ny*((k)+nz*(4)))]-c2*square[(i-1)+(j)*nx+(k)*nx*ny])*um1)
		# ---------------------------------------------------------------------
		# add fourth order xi-direction dissipation               
		# ---------------------------------------------------------------------
		if i==1:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(5.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i+1)+nx*((j)+ny*((k)+nz*(m)))]+u[(i+2)+nx*((j)+ny*((k)+nz*(m)))])
		elif i==2:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(-4.0*u[(i-1)+nx*((j)+ny*((k)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i+1)+nx*((j)+ny*((k)+nz*(m)))]+u[(i+2)+nx*((j)+ny*((k)+nz*(m)))])
		elif i>=3 and i<nx-3:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i-2)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i-1)+nx*((j)+ny*((k)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i+1)+nx*((j)+ny*((k)+nz*(m)))]+u[(i+2)+nx*((j)+ny*((k)+nz*(m)))])
		elif i==nx-3:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i-2)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i-1)+nx*((j)+ny*((k)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i+1)+nx*((j)+ny*((k)+nz*(m)))])
		elif i==nx-2:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i-2)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i-1)+nx*((j)+ny*((k)+nz*(m)))] + 5.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))])

		# ---------------------------------------------------------------------
		# compute eta-direction fluxes 
		# ---------------------------------------------------------------------
		vijk = vs[(i)+(j)*nx+(k)*nx*ny]
		vp1 = vs[(i)+(j+1)*nx+(k)*nx*ny]
		vm1 = vs[(i)+(j-1)*nx+(k)*nx*ny]
		rtmp[0] = rtmp[0]+dy1ty1*(u[(i)+nx*((j+1)+ny*((k)+nz*(0)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(0)))]+u[(i)+nx*((j-1)+ny*((k)+nz*(0)))])-ty2*(u[(i)+nx*((j+1)+ny*((k)+nz*(2)))]-u[(i)+nx*((j-1)+ny*((k)+nz*(2)))])
		rtmp[1] = rtmp[1]+dy2ty1*(u[(i)+nx*((j+1)+ny*((k)+nz*(1)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(1)))]+u[(i)+nx*((j-1)+ny*((k)+nz*(1)))])+yycon2*(us[(i)+(j+1)*nx+(k)*nx*ny]-2.0*us[(i)+(j)*nx+(k)*nx*ny]+us[(i)+(j-1)*nx+(k)*nx*ny])-ty2*(u[(i)+nx*((j+1)+ny*((k)+nz*(1)))]*vp1-u[(i)+nx*((j-1)+ny*((k)+nz*(1)))]*vm1)
		rtmp[2] = rtmp[2]+dy3ty1*(u[(i)+nx*((j+1)+ny*((k)+nz*(2)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(2)))]+u[(i)+nx*((j-1)+ny*((k)+nz*(2)))])+yycon2*con43*(vp1-2.0*vijk+vm1)-ty2*(u[(i)+nx*((j+1)+ny*((k)+nz*(2)))]*vp1-u[(i)+nx*((j-1)+ny*((k)+nz*(2)))]*vm1+(u[(i)+nx*((j+1)+ny*((k)+nz*(4)))]-square[(i)+(j+1)*nx+(k)*nx*ny]-u[(i)+nx*((j-1)+ny*((k)+nz*(4)))]+square[(i)+(j-1)*nx+(k)*nx*ny])*c2)
		rtmp[3] = rtmp[3]+dy4ty1*(u[(i)+nx*((j+1)+ny*((k)+nz*(3)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(3)))]+u[(i)+nx*((j-1)+ny*((k)+nz*(3)))])+yycon2*(ws[(i)+(j+1)*nx+(k)*nx*ny]-2.0*ws[(i)+(j)*nx+(k)*nx*ny]+ws[(i)+(j-1)*nx+(k)*nx*ny])-ty2*(u[(i)+nx*((j+1)+ny*((k)+nz*(3)))]*vp1-u[(i)+nx*((j-1)+ny*((k)+nz*(3)))]*vm1)
		rtmp[4] = rtmp[4]+dy5ty1*(u[(i)+nx*((j+1)+ny*((k)+nz*(4)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(4)))]+u[(i)+nx*((j-1)+ny*((k)+nz*(4)))])+yycon3*(qs[(i)+(j+1)*nx+(k)*nx*ny]-2.0*qs[(i)+(j)*nx+(k)*nx*ny]+qs[(i)+(j-1)*nx+(k)*nx*ny])+yycon4*(vp1*vp1-2.0*vijk*vijk+vm1*vm1)+yycon5*(u[(i)+nx*((j+1)+ny*((k)+nz*(4)))]*rho_i[(i)+(j+1)*nx+(k)*nx*ny]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(4)))]*rho_i[(i)+(j)*nx+(k)*nx*ny]+u[(i)+nx*((j-1)+ny*((k)+nz*(4)))]*rho_i[(i)+(j-1)*nx+(k)*nx*ny])-ty2*((c1*u[(i)+nx*((j+1)+ny*((k)+nz*(4)))]-c2*square[(i)+(j+1)*nx+(k)*nx*ny])*vp1-(c1*u[(i)+nx*((j-1)+ny*((k)+nz*(4)))]-c2*square[(i)+(j-1)*nx+(k)*nx*ny])*vm1)
		# ---------------------------------------------------------------------
		# add fourth order eta-direction dissipation         
		# ---------------------------------------------------------------------
		if j==1:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(5.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j+1)+ny*((k)+nz*(m)))]+u[(i)+nx*((j+2)+ny*((k)+nz*(m)))])
		elif j==2:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(-4.0*u[(i)+nx*((j-1)+ny*((k)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j+1)+ny*((k)+nz*(m)))]+u[(i)+nx*((j+2)+ny*((k)+nz*(m)))])
		elif j>=3 and j<ny-3:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i)+nx*((j-2)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j-1)+ny*((k)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j+1)+ny*((k)+nz*(m)))]+u[(i)+nx*((j+2)+ny*((k)+nz*(m)))])
		elif j==ny-3:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i)+nx*((j-2)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j-1)+ny*((k)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j+1)+ny*((k)+nz*(m)))])
		elif j==ny-2:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i)+nx*((j-2)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j-1)+ny*((k)+nz*(m)))]+5.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))])

		# ---------------------------------------------------------------------
		# compute zeta-direction fluxes 
		# ---------------------------------------------------------------------
		wijk = ws[(i)+(j)*nx+(k)*nx*ny]
		wp1 = ws[(i)+(j)*nx+(k+1)*nx*ny]
		wm1 = ws[(i)+(j)*nx+(k-1)*nx*ny]
		rtmp[0] = rtmp[0]+dz1tz1*(u[(i)+nx*((j)+ny*((k+1)+nz*(0)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(0)))]+u[(i)+nx*((j)+ny*((k-1)+nz*(0)))])-tz2*(u[(i)+nx*((j)+ny*((k+1)+nz*(3)))]-u[(i)+nx*((j)+ny*((k-1)+nz*(3)))])
		rtmp[1] = rtmp[1]+dz2tz1*(u[(i)+nx*((j)+ny*((k+1)+nz*(1)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(1)))]+u[(i)+nx*((j)+ny*((k-1)+nz*(1)))])+zzcon2*(us[(i)+(j)*nx+(k+1)*nx*ny]-2.0*us[(i)+(j)*nx+(k)*nx*ny]+us[(i)+(j)*nx+(k-1)*nx*ny])-tz2*(u[(i)+nx*((j)+ny*((k+1)+nz*(1)))]*wp1-u[(i)+nx*((j)+ny*((k-1)+nz*(1)))]*wm1)
		rtmp[2] = rtmp[2]+dz3tz1*(u[(i)+nx*((j)+ny*((k+1)+nz*(2)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(2)))]+u[(i)+nx*((j)+ny*((k-1)+nz*(2)))])+zzcon2*(vs[(i)+(j)*nx+(k+1)*nx*ny]-2.0*vs[(i)+(j)*nx+(k)*nx*ny]+vs[(i)+(j)*nx+(k-1)*nx*ny])-tz2*(u[(i)+nx*((j)+ny*((k+1)+nz*(2)))]*wp1-u[(i)+nx*((j)+ny*((k-1)+nz*(2)))]*wm1)
		rtmp[3] = rtmp[3]+dz4tz1*(u[(i)+nx*((j)+ny*((k+1)+nz*(3)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(3)))]+u[(i)+nx*((j)+ny*((k-1)+nz*(3)))])+zzcon2*con43*(wp1-2.0*wijk+wm1)-tz2*(u[(i)+nx*((j)+ny*((k+1)+nz*(3)))]*wp1-u[(i)+nx*((j)+ny*((k-1)+nz*(3)))]*wm1+(u[(i)+nx*((j)+ny*((k+1)+nz*(4)))]-square[(i)+(j)*nx+(k+1)*nx*ny]-u[(i)+nx*((j)+ny*((k-1)+nz*(4)))]+square[(i)+(j)*nx+(k-1)*nx*ny])*c2)
		rtmp[4] = rtmp[4]+dz5tz1*(u[(i)+nx*((j)+ny*((k+1)+nz*(4)))]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(4)))]+u[(i)+nx*((j)+ny*((k-1)+nz*(4)))])+zzcon3*(qs[(i)+(j)*nx+(k+1)*nx*ny]-2.0*qs[(i)+(j)*nx+(k)*nx*ny]+qs[(i)+(j)*nx+(k-1)*nx*ny])+zzcon4*(wp1*wp1-2.0*wijk*wijk+wm1*wm1)+zzcon5*(u[(i)+nx*((j)+ny*((k+1)+nz*(4)))]*rho_i[(i)+(j)*nx+(k+1)*nx*ny]-2.0*u[(i)+nx*((j)+ny*((k)+nz*(4)))]*rho_i[(i)+(j)*nx+(k)*nx*ny]+u[(i)+nx*((j)+ny*((k-1)+nz*(4)))]*rho_i[(i)+(j)*nx+(k-1)*nx*ny])-tz2*((c1*u[(i)+nx*((j)+ny*((k+1)+nz*(4)))]-c2*square[(i)+(j)*nx+(k+1)*nx*ny])*wp1-(c1*u[(i)+nx*((j)+ny*((k-1)+nz*(4)))]-c2*square[(i)+(j)*nx+(k-1)*nx*ny])*wm1)
		# ---------------------------------------------------------------------
		# add fourth order zeta-direction dissipation                
		# ---------------------------------------------------------------------
		if k==1:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(5.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k+1)+nz*(m)))]+u[(i)+nx*((j)+ny*((k+2)+nz*(m)))])
		elif k==2:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(-4.0*u[(i)+nx*((j)+ny*((k-1)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k+1)+nz*(m)))]+u[(i)+nx*((j)+ny*((k+2)+nz*(m)))])
		elif k>=3 and k<nz-3:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i)+nx*((j)+ny*((k-2)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k-1)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k+1)+nz*(m)))]+u[(i)+nx*((j)+ny*((k+2)+nz*(m)))])
		elif k==nz-3:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i)+nx*((j)+ny*((k-2)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k-1)+nz*(m)))]+6.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k+1)+nz*(m)))])
		elif k==nz-2:
			for m in range(5):
				rtmp[m] = rtmp[m]-dssp*(u[(i)+nx*((j)+ny*((k-2)+nz*(m)))]-4.0*u[(i)+nx*((j)+ny*((k-1)+nz*(m)))]+5.0*u[(i)+nx*((j)+ny*((k)+nz*(m)))])

		for m in range(5):
			rtmp[m] *= dt
	#END if k>=1 and k<nz-1 and j>=1 and j<ny-1 and i>=1 and i<nx-1:
	
	for m in range(5):
		rhs[(m)+(i)*5+(j)*5*nx+(k)*5*nx*ny] = rtmp[m]
#END compute_rhs_gpu_kernel_2()


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32, float64)')
def compute_rhs_gpu_kernel_1(rho_i,
							us,
							vs, 
							ws, 
							speed, 
							qs, 
							square, 
							u, 
							nx, 
							ny, 
							nz,
							c1c2):
	i_j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j_k % nx
	j = int(i_j_k / nx) % ny
	k = int(i_j_k / (nx * ny))

	if i_j_k >= (nx*ny*nz):
		return

	# ---------------------------------------------------------------------
	# compute the reciprocal of density, and the kinetic energy, 
	# and the speed of sound. 
	# ---------------------------------------------------------------------
	rho_inv = 1.0 / u[(i)+nx*((j)+ny*((k)+nz*(0)))]
	rho_i[(i)+(j)*nx+(k)*nx*ny] = rho_inv
	us[(i)+(j)*nx+(k)*nx*ny] = u[(i)+nx*((j)+ny*((k)+nz*(1)))] * rho_inv
	vs[(i)+(j)*nx+(k)*nx*ny] = u[(i)+nx*((j)+ny*((k)+nz*(2)))] * rho_inv
	ws[(i)+(j)*nx+(k)*nx*ny] = u[(i)+nx*((j)+ny*((k)+nz*(3)))] * rho_inv
	square[(i)+(j)*nx+(k)*nx*ny] = square_ijk = 0.5*(u[(i)+nx*((j)+ny*((k)+nz*(1)))]*u[(i)+nx*((j)+ny*((k)+nz*(1)))]+u[(i)+nx*((j)+ny*((k)+nz*(2)))]*u[(i)+nx*((j)+ny*((k)+nz*(2)))]+u[(i)+nx*((j)+ny*((k)+nz*(3)))]*u[(i)+nx*((j)+ny*((k)+nz*(3)))])*rho_inv
	qs[(i)+(j)*nx+(k)*nx*ny] = square_ijk*rho_inv
	# ---------------------------------------------------------------------
	# (don't need speed and ainx until the lhs computation)
	# ---------------------------------------------------------------------
	speed[(i)+(j)*nx+(k)*nx*ny] = math.sqrt(c1c2*rho_inv*(u[(i)+nx*((j)+ny*((k)+nz*(4)))]-square_ijk))
#END compute_rhs_gpu_kernel_1()


def compute_rhs_gpu(rho_i_device, us_device, vs_device, ws_device, 
					speed_device, qs_device, square_device, 
					u_device, rhs_device, forcing_device):
	global const_compute_rhs_gpu_kernel_2_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_COMPUTE_RHS_1)
	# #KERNEL COMPUTE RHS 1
	compute_rhs_1_workload = nx * ny * nz
	compute_rhs_1_threads_per_block = THREADS_PER_BLOCK_ON_COMPUTE_RHS_1
	compute_rhs_1_blocks_per_grid = math.ceil(compute_rhs_1_workload / compute_rhs_1_threads_per_block)

	compute_rhs_gpu_kernel_1[compute_rhs_1_blocks_per_grid,
		compute_rhs_1_threads_per_block](rho_i_device, 
										us_device, 
										vs_device, 
										ws_device, 
										speed_device, 
										qs_device, 
										square_device, 
										u_device, 
										nx, 
										ny, 
										nz,
										c1c2)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_COMPUTE_RHS_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_COMPUTE_RHS_2)
	# #KERNEL COMPUTE RHS 2
	compute_rhs_2_blocks_per_grid = (ny, nz)
	compute_rhs_2_threads_per_block = THREADS_PER_BLOCK_ON_COMPUTE_RHS_2
	if THREADS_PER_BLOCK_ON_COMPUTE_RHS_2 != nx:
		compute_rhs_2_threads_per_block = nx

	compute_rhs_gpu_kernel_2[compute_rhs_2_blocks_per_grid, 
		compute_rhs_2_threads_per_block](rho_i_device, 
										us_device, 
										vs_device, 
										ws_device, 
										qs_device, 
										square_device, 
										rhs_device, 
										forcing_device, 
										u_device, 
										nx, 
										ny, 
										nz,
										const_compute_rhs_gpu_kernel_2_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_COMPUTE_RHS_2)
#END compute_rhs_gpu()


def adi_gpu(rho_i_device, us_device, vs_device, ws_device, 
			speed_device, qs_device, square_device, 
			u_device, rhs_device, forcing_device,
			lhs_device, rhs_buffer_device):
	
	compute_rhs_gpu(rho_i_device, us_device, vs_device, ws_device, 
					speed_device, qs_device, square_device, 
					u_device, rhs_device, forcing_device)
	
	txinvr_gpu(rho_i_device, us_device, vs_device, ws_device, 
			speed_device, qs_device, rhs_device)
	
	x_solve_gpu(rho_i_device, us_device, speed_device,
				rhs_device, lhs_device, rhs_buffer_device)
	
	y_solve_gpu(rho_i_device, vs_device, speed_device, 
				rhs_device, lhs_device, rhs_buffer_device)
	
	z_solve_gpu(rho_i_device, us_device, vs_device,
				ws_device, speed_device, qs_device,
				u_device, rhs_device, lhs_device, rhs_buffer_device)
	
	add_gpu(u_device, rhs_device)
#END adi_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:, :], float64, float64, float64)')
def initialize_gpu_kernel(u,
						nx,
						ny,
						nz,
						ce_device,
						dnzm1, dnym1, dnxm1):
	temp = cuda.local.array(5, numba.float64)
	Pface11 = cuda.local.array(5, numba.float64)
	Pface12 = cuda.local.array(5, numba.float64)
	Pface21 = cuda.local.array(5, numba.float64)
	Pface22 = cuda.local.array(5, numba.float64)
	Pface31 = cuda.local.array(5, numba.float64)
	Pface32 = cuda.local.array(5, numba.float64)
	
	k = cuda.blockIdx.x
	j = cuda.blockIdx.y
	i = cuda.threadIdx.x

	# ---------------------------------------------------------------------
	# later (in compute_rhs_gpu) we compute 1/u for every element. a few of 
	# the corner elements are not used, but it convenient (and faster) 
	# to compute the whole thing with a simple loop. make sure those 
	# values are nonzero by initializing the whole thing here. 
	# ---------------------------------------------------------------------
	u[(i)+nx*((j)+ny*((k)+nz*(0)))] = 1.0
	u[(i)+nx*((j)+ny*((k)+nz*(1)))] = 0.0
	u[(i)+nx*((j)+ny*((k)+nz*(2)))] = 0.0
	u[(i)+nx*((j)+ny*((k)+nz*(3)))] = 0.0
	u[(i)+nx*((j)+ny*((k)+nz*(4)))] = 1.0
	# ---------------------------------------------------------------------
	# first store the "interpolated" values everywhere on the grid    
	# ---------------------------------------------------------------------
	zeta = k * dnzm1
	eta = j * dnym1
	xi = i * dnxm1
	exact_solution_gpu_device(0.0, eta, zeta, Pface11, ce_device)
	exact_solution_gpu_device(1.0, eta, zeta, Pface12, ce_device)
	exact_solution_gpu_device(xi, 0.0, zeta, Pface21, ce_device)
	exact_solution_gpu_device(xi, 1.0, zeta, Pface22, ce_device)
	exact_solution_gpu_device(xi, eta, 0.0, Pface31, ce_device)
	exact_solution_gpu_device(xi, eta, 1.0, Pface32, ce_device)
	for m in range(5):
		Pxi = xi*Pface12[m]+(1.0-xi)*Pface11[m]
		Peta = eta*Pface22[m]+(1.0-eta)*Pface21[m]
		Pzeta = zeta*Pface32[m]+(1.0-zeta)*Pface31[m]
		u[(i)+nx*((j)+ny*((k)+nz*(m)))] = Pxi+Peta+Pzeta-Pxi*Peta-Pxi*Pzeta-Peta*Pzeta+Pxi*Peta*Pzeta

	# ---------------------------------------------------------------------
	# now store the exact values on the boundaries        
	# ---------------------------------------------------------------------
	# west face                                                  
	# ---------------------------------------------------------------------
	xi = 0.0
	if i==0:
		zeta = k * dnzm1
		eta = j * dnym1
		exact_solution_gpu_device(xi, eta, zeta, temp, ce_device)
		for m in range(5):
			u[(i)+nx*((j)+ny*((k)+nz*(m)))] = temp[m]

	# ---------------------------------------------------------------------
	# east face                                                      
	# ---------------------------------------------------------------------
	xi = 1.0
	if i==nx-1:
		zeta = k * dnzm1
		eta = j * dnym1
		exact_solution_gpu_device(xi, eta, zeta, temp, ce_device)
		for m in range(5):
			u[(i)+nx*((j)+ny*((k)+nz*(m)))] = temp[m]

	# ---------------------------------------------------------------------
	# south face                                                 
	# ---------------------------------------------------------------------
	eta = 0.0
	if j==0:
		zeta = k * dnzm1
		xi = i * dnxm1
		exact_solution_gpu_device(xi, eta, zeta, temp, ce_device)
		for m in range(5):
			u[(i)+nx*((j)+ny*((k)+nz*(m)))] = temp[m]
	# ---------------------------------------------------------------------
	# north face                                    
	# ---------------------------------------------------------------------
	eta = 1.0
	if j==ny-1:
		zeta = k * dnzm1
		xi = i * dnxm1
		exact_solution_gpu_device(xi, eta, zeta, temp, ce_device)
		for m in range(5):
			u[(i)+nx*((j)+ny*((k)+nz*(m)))] = temp[m]
	# ---------------------------------------------------------------------
	# bottom face                                       
	# ---------------------------------------------------------------------
	zeta = 0.0
	if k==0:
		eta = j * dnym1
		xi = i * dnxm1
		exact_solution_gpu_device(xi, eta, zeta, temp, ce_device)
		for m in range(5):
			u[(i)+nx*((j)+ny*((k)+nz*(m)))] = temp[m]

	# ---------------------------------------------------------------------
	# top face     
	# ---------------------------------------------------------------------
	zeta = 1.0
	if k==nz-1:
		eta = j * dnym1
		xi = i * dnxm1
		exact_solution_gpu_device(xi, eta, zeta, temp, ce_device)
		for m in range(5):
			u[(i)+nx*((j)+ny*((k)+nz*(m)))] = temp[m]
#END initialize_gpu_kernel()


# ---------------------------------------------------------------------
# this subroutine initializes the field variable u using 
# tri-linear transfinite interpolation of the boundary values     
# ---------------------------------------------------------------------
def initialize_gpu(u_device):
	global ce_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_INITIALIZE)
	# #KERNEL INITIALIZE
	initialize_blocks_per_grid = (nz, ny)
	initialize_threads_per_block = THREADS_PER_BLOCK_ON_INITIALIZE
	if THREADS_PER_BLOCK_ON_INITIALIZE != nx:
		initialize_threads_per_block = nx

	initialize_gpu_kernel[initialize_blocks_per_grid, 
		initialize_threads_per_block](u_device, 
									nx, 
									ny, 
									nz,
									ce_device,
									dnzm1, dnym1, dnxm1)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_INITIALIZE)
#END initialize_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:], float64[:, :])')
def exact_rhs_gpu_kernel_4(forcing,
						nx,
						ny,
						nz,
						const_arr,
						ce_device):
	dtemp = cuda.local.array(5, numba.float64)
	ue = cuda.local.array((5, 5), numba.float64)
	buf = cuda.local.array((5, 5), numba.float64)
	cuf = cuda.local.array(3, numba.float64)
	q = cuda.local.array(3, numba.float64)
	
	j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
	i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1

	if j>=ny-1 or i>=nx-1:
		return

	#Constants
	dnzm1, dnym1, dnxm1 = const_arr[0], const_arr[1], const_arr[2]  
	tz2 = const_arr[3]
	dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1 = const_arr[4], const_arr[5], const_arr[6], const_arr[7], const_arr[8]
	c1, c2 = const_arr[9], const_arr[10] 
	zzcon1, zzcon2, zzcon3, zzcon4, zzcon5 = const_arr[11], const_arr[12], const_arr[13], const_arr[14], const_arr[15]
	dssp = const_arr[16]

	eta = j * dnym1
	xi = i * dnxm1
	# ---------------------------------------------------------------------
	# zeta-direction flux differences                      
	# ---------------------------------------------------------------------
	for k in range(3):
		zeta = k * dnzm1
		exact_solution_gpu_device(xi, eta, zeta, dtemp, ce_device)
		for m in range(5):
			ue[k+1][m] = dtemp[m]
		dtpp = 1.0/dtemp[0]
		for m in range(1, 5):
			buf[k][m] = dtpp*dtemp[m]
		cuf[k] = buf[k][3]*buf[k][3]
		buf[k][0] = cuf[k]+buf[k][1]*buf[k][1]+buf[k][2]*buf[k][2]
		q[k] = 0.5*(buf[k][1]*ue[k+1][1]+buf[k][2]*ue[k+1][2]+buf[k][3]*ue[k+1][3])

	for k in range(1, nz-1):
		if k+2<nz:
			zeta = (k+2)*dnzm1
			exact_solution_gpu_device(xi, eta, zeta, dtemp, ce_device)
			for m in range(5):
				ue[4][m] = dtemp[m]

		dtemp[0] = forcing[(i)+nx*((j)+ny*((k)+nz*(0)))]-tz2*(ue[3][3]-ue[1][3])+dz1tz1*(ue[3][0]-2.0*ue[2][0]+ue[1][0])
		dtemp[1] = forcing[(i)+nx*((j)+ny*((k)+nz*(1)))]-tz2*(ue[3][1]*buf[2][3]-ue[1][1]*buf[0][3])+zzcon2*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dz2tz1*(ue[3][1]-2.0*ue[2][1]+ue[1][1])
		dtemp[2] = forcing[(i)+nx*((j)+ny*((k)+nz*(2)))]-tz2*(ue[3][2]*buf[2][3]-ue[1][2]*buf[0][3])+zzcon2*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dz3tz1*(ue[3][2]-2.0*ue[2][2]+ue[1][2])
		dtemp[3] = forcing[(i)+nx*((j)+ny*((k)+nz*(3)))]-tz2*((ue[3][3]*buf[2][3]+c2*(ue[3][4]-q[2]))-(ue[1][3]*buf[0][3]+c2*(ue[1][4]-q[0])))+zzcon1*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dz4tz1*(ue[3][3]-2.0*ue[2][3]+ue[1][3])
		dtemp[4] = forcing[(i)+nx*((j)+ny*((k)+nz*(4)))]-tz2*(buf[2][3]*(c1*ue[3][4]-c2*q[2])-buf[0][3]*(c1*ue[1][4]-c2*q[0]))+0.5*zzcon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+zzcon4*(cuf[2]-2.0*cuf[1]+cuf[0])+zzcon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dz5tz1*(ue[3][4]-2.0*ue[2][4]+ue[1][4])
		# ---------------------------------------------------------------------
		# fourth-order dissipation
		# ---------------------------------------------------------------------
		if k==1:
			for m in range(5):
				dtemp[m] = dtemp[m]-dssp*(5.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif k==2:
			for m in range(5):
				dtemp[m] = dtemp[m]-dssp*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif k>=3 and k<nz-3:
			for m in range(5):
				dtemp[m] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif k==nz-3:
			for m in range(5):
				dtemp[m] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m])
		elif k==nz-2:
			for m in range(5):
				dtemp[m] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m])
		# ---------------------------------------------------------------------
		# now change the sign of the forcing function
		# ---------------------------------------------------------------------
		for m in range(5):
			forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = -1.0*dtemp[m]
		for m in range(5):
			ue[0][m] = ue[1][m] 
			ue[1][m] = ue[2][m]
			ue[2][m] = ue[3][m]
			ue[3][m] = ue[4][m]
			buf[0][m] = buf[1][m]
			buf[1][m] = buf[2][m]

		cuf[0] = cuf[1]
		cuf[1] = cuf[2]
		q[0] = q[1]
		q[1] = q[2]
		if k<nz-2:
			dtpp = 1.0/ue[3][0]
			for m in range(5):
				buf[2][m] = dtpp*ue[3][m]
			cuf[2] = buf[2][3]*buf[2][3]
			buf[2][0] = cuf[2]+buf[2][1]*buf[2][1]+buf[2][2]*buf[2][2]
			q[2] = 0.5*(buf[2][1]*ue[3][1]+buf[2][2]*ue[3][2]+buf[2][3]*ue[3][3])
#END exact_rhs_gpu_kernel_4()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:], float64[:, :])')
def exact_rhs_gpu_kernel_3(forcing,
						nx,
						ny,
						nz,
						const_arr,
						ce_device):
	dtemp = cuda.local.array(5, numba.float64)
	ue = cuda.local.array((5, 5), numba.float64)
	buf = cuda.local.array((5, 5), numba.float64)
	cuf = cuda.local.array(3, numba.float64)
	q = cuda.local.array(3, numba.float64)
	
	k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
	i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	
	if k>=nz-1 or i>=nx-1:
		return
	
	#Constants
	dnzm1, dnym1, dnxm1 = const_arr[0], const_arr[1], const_arr[2]  
	ty2 = const_arr[3]
	dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1 = const_arr[4], const_arr[5], const_arr[6], const_arr[7], const_arr[8]
	c1, c2 = const_arr[9], const_arr[10] 
	yycon1, yycon2, yycon3, yycon4, yycon5 = const_arr[11], const_arr[12], const_arr[13], const_arr[14], const_arr[15]
	dssp = const_arr[16]

	zeta = k * dnzm1
	xi = i * dnxm1
	# ---------------------------------------------------------------------
	# eta-direction flux differences             
	# ---------------------------------------------------------------------
	for j in range(3):
		eta = j * dnym1
		exact_solution_gpu_device(xi, eta, zeta, dtemp, ce_device)
		for m in range(5):
			ue[j+1][m] = dtemp[m]
		dtpp = 1.0/dtemp[0]
		for m in range(1, 5): 
			buf[j][m] = dtpp*dtemp[m]
		cuf[j] = buf[j][2]*buf[j][2]
		buf[j][0] = cuf[j]+buf[j][1]*buf[j][1]+buf[j][3]*buf[j][3]
		q[j] = 0.5*(buf[j][1]*ue[j+1][1]+buf[j][2]*ue[j+1][2]+buf[j][3]*ue[j+1][3])

	for j in range(1, ny-1):
		if j+2<ny:
			eta = (j+2) * dnym1
			exact_solution_gpu_device(xi, eta, zeta, dtemp, ce_device)
			for m in range(5):
				ue[4][m] = dtemp[m]

		dtemp[0] = forcing[(i)+nx*((j)+ny*((k)+nz*(0)))]-ty2*(ue[3][2]-ue[1][2])+dy1ty1*(ue[3][0]-2.0*ue[2][0]+ue[1][0])
		dtemp[1] = forcing[(i)+nx*((j)+ny*((k)+nz*(1)))]-ty2*(ue[3][1]*buf[2][2]-ue[1][1]*buf[0][2])+yycon2*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dy2ty1*(ue[3][1]-2.0*ue[2][1]+ ue[1][1])
		dtemp[2] = forcing[(i)+nx*((j)+ny*((k)+nz*(2)))]-ty2*((ue[3][2]*buf[2][2]+c2*(ue[3][4]-q[2]))-(ue[1][2]*buf[0][2]+c2*(ue[1][4]-q[0])))+yycon1*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dy3ty1*(ue[3][2]-2.0*ue[2][2]+ue[1][2])
		dtemp[3] = forcing[(i)+nx*((j)+ny*((k)+nz*(3)))]-ty2*(ue[3][3]*buf[2][2]-ue[1][3]*buf[0][2])+yycon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dy4ty1*(ue[3][3]-2.0*ue[2][3]+ue[1][3])
		dtemp[4] = forcing[(i)+nx*((j)+ny*((k)+nz*(4)))]-ty2*(buf[2][2]*(c1*ue[3][4]-c2*q[2])-buf[0][2]*(c1*ue[1][4]-c2*q[0]))+0.5*yycon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+yycon4*(cuf[2]-2.0*cuf[1]+cuf[0])+yycon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dy5ty1*(ue[3][4]-2.0*ue[2][4]+ue[1][4])
		# ---------------------------------------------------------------------
		# fourth-order dissipation                      
		# ---------------------------------------------------------------------
		if j==1:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(5.0*ue[2][m]-4.0*ue[3][m] +ue[4][m])
		elif j==2:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif j>=3 and j<ny-3:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif j==ny-3:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m])
		elif j==ny-2:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m])

		for m in range(5):
			ue[0][m] = ue[1][m] 
			ue[1][m] = ue[2][m]
			ue[2][m] = ue[3][m]
			ue[3][m] = ue[4][m]
			buf[0][m] = buf[1][m]
			buf[1][m] = buf[2][m]

		cuf[0] = cuf[1]
		cuf[1] = cuf[2]
		q[0] = q[1]
		q[1] = q[2]
		if j<ny-2:
			dtpp = 1.0/ue[3][0]
			for m in range(1, 5): 
				buf[2][m]=dtpp*ue[3][m]
			cuf[2] = buf[2][2]*buf[2][2]
			buf[2][0] = cuf[2]+buf[2][1]*buf[2][1]+buf[2][3]*buf[2][3]
			q[2] = 0.5*(buf[2][1]*ue[3][1]+buf[2][2]*ue[3][2]+buf[2][3]*ue[3][3])
#END exact_rhs_gpu_kernel_3()


@cuda.jit('void(float64[:], int32, int32, int32, float64[:], float64[:, :])')
def exact_rhs_gpu_kernel_2(forcing,
						nx,
						ny,
						nz,
						const_arr,
						ce_device):
	dtemp = cuda.local.array(5, numba.float64)
	ue = cuda.local.array((5, 5), numba.float64)
	buf = cuda.local.array((5, 5), numba.float64)
	cuf = cuda.local.array(3, numba.float64)
	q = cuda.local.array(3, numba.float64)
	
	k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1

	if k>=(nz-1) or j>=(ny-1):
		return
	
	#Constants
	dnzm1, dnym1, dnxm1 = const_arr[0], const_arr[1], const_arr[2]  
	tx2 = const_arr[3]
	dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1 = const_arr[4], const_arr[5], const_arr[6], const_arr[7], const_arr[8]
	c1, c2 = const_arr[9], const_arr[10] 
	xxcon1, xxcon2, xxcon3, xxcon4, xxcon5 = const_arr[11], const_arr[12], const_arr[13], const_arr[14], const_arr[15]
	dssp = const_arr[16]
	
	zeta = k * dnzm1
	eta = j * dnym1
	# ---------------------------------------------------------------------
	# xi-direction flux differences                      
	# ---------------------------------------------------------------------
	for i in range(3):
		xi = i * dnxm1
		exact_solution_gpu_device(xi, eta, zeta, dtemp, ce_device)
		for m in range(5):
			ue[i+1][m] = dtemp[m]
		dtpp=1.0/dtemp[0]
		for m in range(1, 5):
			buf[i][m] = dtpp * dtemp[m]
		cuf[i] = buf[i][1] * buf[i][1]
		buf[i][0] = cuf[i]+buf[i][2]*buf[i][2]+buf[i][3]*buf[i][3]
		q[i] = 0.5*(buf[i][1]*ue[i+1][1]+buf[i][2]*ue[i+1][2]+buf[i][3]*ue[i+1][3])

	for i in range(1, nx-1):
		if i+2 < nx:
			xi = (i+2) * dnxm1
			exact_solution_gpu_device(xi, eta, zeta, dtemp, ce_device)
			for m in range(5):
				ue[4][m] = dtemp[m]

		dtemp[0] = 0.0-tx2*(ue[3][1]-ue[1][1])+dx1tx1*(ue[3][0]-2.0*ue[2][0]+ue[1][0])
		dtemp[1] = 0.0-tx2*((ue[3][1]*buf[2][1]+c2*(ue[3][4]-q[2]))-(ue[1][1]*buf[0][1]+c2*(ue[1][4]-q[0])))+xxcon1*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dx2tx1*(ue[3][1]-2.0*ue[2][1]+ue[1][1])
		dtemp[2] = 0.0-tx2*(ue[3][2]*buf[2][1]-ue[1][2]*buf[0][1])+xxcon2*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dx3tx1*(ue[3][2]-2.0*ue[2][2]+ue[1][2])
		dtemp[3] = 0.0-tx2*(ue[3][3]*buf[2][1]-ue[1][3]*buf[0][1])+xxcon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dx4tx1*(ue[3][3]-2.0*ue[2][3]+ue[1][3])
		dtemp[4] = 0.0-tx2*(buf[2][1]*(c1*ue[3][4]-c2*q[2])-buf[0][1]*(c1*ue[1][4]-c2*q[0]))+0.5*xxcon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+xxcon4*(cuf[2]-2.0*cuf[1]+cuf[0])+xxcon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dx5tx1*(ue[3][4]-2.0*ue[2][4]+ue[1][4])
		# ---------------------------------------------------------------------
		# fourth-order dissipation                         
		# ---------------------------------------------------------------------
		if i==1:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(5.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif i==2:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif i>=3 and i<nx-3:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m])
		elif i==nx-3:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m])
		elif i==nx-2:
			for m in range(5):
				forcing[(i)+nx*((j)+ny*((k)+nz*(m)))] = dtemp[m]-dssp*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m])

		for m in range(5):
			ue[0][m] = ue[1][m] 
			ue[1][m] = ue[2][m]
			ue[2][m] = ue[3][m]
			ue[3][m] = ue[4][m]
			buf[0][m] = buf[1][m]
			buf[1][m] = buf[2][m]

		cuf[0] = cuf[1]
		cuf[1] = cuf[2]
		q[0] = q[1]
		q[1] = q[2]
		if i<nx-2:
			dtpp = 1.0 / ue[3][0]
			for m in range(1, 5):
				buf[2][m] = dtpp*ue[3][m]
			cuf[2] = buf[2][1]*buf[2][1]
			buf[2][0] = cuf[2]+buf[2][2]*buf[2][2]+buf[2][3]*buf[2][3]
			q[2] = 0.5*(buf[2][1]*ue[3][1]+buf[2][2]*ue[3][2]+buf[2][3]*ue[3][3])
#END exact_rhs_gpu_kernel_2()


@cuda.jit('void(float64[:], int32, int32, int32)')
def exact_rhs_gpu_kernel_1(forcing, 
						nx,
						ny,
						nz):
	i_j_k = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	i = i_j_k % nx
	j = int(i_j_k / nx) % ny
	k = int(i_j_k / (nx * ny))

	if i_j_k >= (nx*ny*nz):
		return

	# ---------------------------------------------------------------------
	# initialize                                  
	# ---------------------------------------------------------------------
	# array(m,i,j,k)
	forcing[(i)+nx*((j)+ny*((k)+nz*(0)))] = 0.0
	forcing[(i)+nx*((j)+ny*((k)+nz*(1)))] = 0.0
	forcing[(i)+nx*((j)+ny*((k)+nz*(2)))] = 0.0
	forcing[(i)+nx*((j)+ny*((k)+nz*(3)))] = 0.0
	forcing[(i)+nx*((j)+ny*((k)+nz*(4)))] = 0.0
#END exact_rhs_gpu_kernel_1()


# ---------------------------------------------------------------------
# compute the right hand side based on exact solution
# ---------------------------------------------------------------------
def exact_rhs_gpu(forcing_device):
	global const_exact_rhs_2_device, const_exact_rhs_3_device, const_exact_rhs_4_device
	global ce_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_EXACT_RHS_1)
	# #KERNEL EXACT RHS 1
	rhs1_workload = nx * ny * nz
	rhs1_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_1
	rhs1_blocks_per_grid = math.ceil(rhs1_workload / rhs1_threads_per_block)

	exact_rhs_gpu_kernel_1[rhs1_blocks_per_grid,
		rhs1_threads_per_block](forcing_device, 
								nx, 
								ny, 
								nz)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_EXACT_RHS_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_EXACT_RHS_2)
	# #KERNEL EXACT RHS 2
	rhs2_blocks_per_grid = (nz, ny)
	rhs2_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_2
	if THREADS_PER_BLOCK_ON_EXACT_RHS_2 > nx:
		rhs2_threads_per_block = nx

	exact_rhs_gpu_kernel_2[rhs2_blocks_per_grid,
		rhs2_threads_per_block](forcing_device, 
								nx, 
								ny, 
								nz,
								const_exact_rhs_2_device,
								ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_EXACT_RHS_2)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_EXACT_RHS_3)
	# #KERNEL EXACT RHS 3
	rhs3_blocks_per_grid = (nz, nx)
	rhs3_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_3
	if THREADS_PER_BLOCK_ON_EXACT_RHS_3 > ny:
		rhs3_threads_per_block = ny

	exact_rhs_gpu_kernel_3[rhs3_blocks_per_grid, 
		rhs3_threads_per_block](forcing_device, 
								nx, 
								ny, 
								nz,
								const_exact_rhs_3_device,
								ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_EXACT_RHS_3)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_EXACT_RHS_4)
	# #KERNEL EXACT RHS 4 
	rhs4_blocks_per_grid = (ny, nx)
	rhs4_threads_per_block = THREADS_PER_BLOCK_ON_EXACT_RHS_4
	if THREADS_PER_BLOCK_ON_EXACT_RHS_4 > nz:
		rhs4_threads_per_block = nz

	exact_rhs_gpu_kernel_4[rhs4_blocks_per_grid, 
		rhs4_threads_per_block](forcing_device, 
								nx, 
								ny, 
								nz, 
								const_exact_rhs_4_device,
								ce_device)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_EXACT_RHS_4)
#END exact_rhs_gpu()


#*****************************************************************
#************************* CPU FUNCTIONS *************************
#*****************************************************************

def main():
	global dt_host
	global grid_points
	global nx, ny, nz
	
	global forcing_device, u_device, rhs_device
	global rho_i_device, us_device, vs_device, ws_device, speed_device, qs_device, square_device
	global lhs_device, rhs_buffer_device
	
	if gpu_config.PROFILING:
		print(" PROFILING mode on")
	
	# ---------------------------------------------------------------------
	# root node reads input file (if it exists) else takes
	# defaults from parameters
	# ---------------------------------------------------------------------
	niter = 0
	dt_host = 0.0
	grid_points[0] = 0
	grid_points[1] = 0
	grid_points[2] = 0
	
	fp = os.path.isfile("inputsp.data")
	if fp:
		print(" Reading from input file inputsp.data") 
		print(" ERROR - Not implemented") 
		sys.exit()
	else:
		print(" No input file inputsp.data. Using compiled defaults")
		niter = npbparams.NITER_DEFAULT
		dt_host = npbparams.DT_DEFAULT
		grid_points[0] = npbparams.PROBLEM_SIZE
		grid_points[1] = npbparams.PROBLEM_SIZE
		grid_points[2] = npbparams.PROBLEM_SIZE
		
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - SP Benchmark\n")
	print(" Size: %4dx%4dx%4d" % (grid_points[0], grid_points[1], grid_points[2]))
	print(" Iterations: %4d    dt: %10.6f" % (niter, dt_host))
	print()
	if (grid_points[0] > IMAX) or (grid_points[1] > JMAX) or (grid_points[2] > KMAX):
		print(" %d, %d, %d" % (grid_points[0], grid_points[1], grid_points[2]))
		print(" Problem size too big for compiled array sizes")
		sys.exit()
	
	nx = grid_points[0]
	ny = grid_points[1]
	nz = grid_points[2]
	
	setup_gpu()
	set_constants(dt_host)
	
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		c_timers.timer_clear(PROFILING_ADD)
		c_timers.timer_clear(PROFILING_COMPUTE_RHS_1)
		c_timers.timer_clear(PROFILING_COMPUTE_RHS_2)
		c_timers.timer_clear(PROFILING_ERROR_NORM_1)
		c_timers.timer_clear(PROFILING_ERROR_NORM_2)
		c_timers.timer_clear(PROFILING_EXACT_RHS_1)
		c_timers.timer_clear(PROFILING_EXACT_RHS_2)
		c_timers.timer_clear(PROFILING_EXACT_RHS_3)
		c_timers.timer_clear(PROFILING_EXACT_RHS_4)
		c_timers.timer_clear(PROFILING_INITIALIZE)
		c_timers.timer_clear(PROFILING_RHS_NORM_1)
		c_timers.timer_clear(PROFILING_RHS_NORM_2)
		c_timers.timer_clear(PROFILING_TXINVR)
		c_timers.timer_clear(PROFILING_X_SOLVE)
		c_timers.timer_clear(PROFILING_Y_SOLVE)
		c_timers.timer_clear(PROFILING_Z_SOLVE)

	exact_rhs_gpu(forcing_device)
	initialize_gpu(u_device)
	# ---------------------------------------------------------------------
	# do one time step to touch all code, and reinitialize
	# ---------------------------------------------------------------------
	# CPU initialize(u)
	adi_gpu(rho_i_device, us_device, vs_device, ws_device, 
			speed_device, qs_device, square_device, 
			u_device, rhs_device, forcing_device,
			lhs_device, rhs_buffer_device)
	initialize_gpu(u_device)
	
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		c_timers.timer_clear(PROFILING_ADD)
		c_timers.timer_clear(PROFILING_COMPUTE_RHS_1)
		c_timers.timer_clear(PROFILING_COMPUTE_RHS_2)
		c_timers.timer_clear(PROFILING_ERROR_NORM_1)
		c_timers.timer_clear(PROFILING_ERROR_NORM_2)
		c_timers.timer_clear(PROFILING_EXACT_RHS_1)
		c_timers.timer_clear(PROFILING_EXACT_RHS_2)
		c_timers.timer_clear(PROFILING_EXACT_RHS_3)
		c_timers.timer_clear(PROFILING_EXACT_RHS_4)
		c_timers.timer_clear(PROFILING_INITIALIZE)
		c_timers.timer_clear(PROFILING_RHS_NORM_1)
		c_timers.timer_clear(PROFILING_RHS_NORM_2)
		c_timers.timer_clear(PROFILING_TXINVR)
		c_timers.timer_clear(PROFILING_X_SOLVE)
		c_timers.timer_clear(PROFILING_Y_SOLVE)
		c_timers.timer_clear(PROFILING_Z_SOLVE)
	
	c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	for step in range(1, niter+1):
		if (step % 20) == 0 or step == 1:
			print(" Time step %4d" % (step))
		adi_gpu(rho_i_device, us_device, vs_device, ws_device, 
				speed_device, qs_device, square_device, 
				u_device, rhs_device, forcing_device,
				lhs_device, rhs_buffer_device)
	
	c_timers.timer_stop(PROFILING_TOTAL_TIME)
	tmax = c_timers.timer_read(PROFILING_TOTAL_TIME)
	
	verified = verify_gpu(u_device, rhs_device, forcing_device)
	n3 = grid_points[0] * grid_points[1] * grid_points[2]
	t = (grid_points[0] + grid_points[1] + grid_points[2]) / 3.0
	mflops = 0.0
	if tmax != 0.0:
		mflops = ( (881.174 * n3-
				4683.91 * (t*t) +
				11484.5 * t-
				19272.4) * niter/(tmax*1000000.0) )
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_ADD)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-add", THREADS_PER_BLOCK_ON_ADD, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_COMPUTE_RHS_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-compute-rhs-1", THREADS_PER_BLOCK_ON_COMPUTE_RHS_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_COMPUTE_RHS_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-compute-rhs-2", THREADS_PER_BLOCK_ON_COMPUTE_RHS_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERROR_NORM_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-error-norm-1", THREADS_PER_BLOCK_ON_ERROR_NORM_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERROR_NORM_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-error-norm-2", THREADS_PER_BLOCK_ON_ERROR_NORM_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-exact-rhs-1", THREADS_PER_BLOCK_ON_EXACT_RHS_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-exact-rhs-2", THREADS_PER_BLOCK_ON_EXACT_RHS_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-exact-rhs-3", THREADS_PER_BLOCK_ON_EXACT_RHS_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_4)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-exact-rhs-4", THREADS_PER_BLOCK_ON_EXACT_RHS_4, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_INITIALIZE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-initialize", THREADS_PER_BLOCK_ON_INITIALIZE, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_NORM_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-rhs-norm-1", THREADS_PER_BLOCK_ON_RHS_NORM_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_NORM_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-rhs-norm-2", THREADS_PER_BLOCK_ON_RHS_NORM_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_TXINVR)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-txinvr", THREADS_PER_BLOCK_ON_TXINVR, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_X_SOLVE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-x-solve", THREADS_PER_BLOCK_ON_X_SOLVE, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Y_SOLVE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-y-solve", THREADS_PER_BLOCK_ON_Y_SOLVE, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Z_SOLVE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" sp-z-solve", THREADS_PER_BLOCK_ON_Z_SOLVE, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" bt-add", THREADS_PER_BLOCK_ON_ADD)
		gpu_config_string += "%29s\t%25d\n" % (" sp-compute-rhs-1", THREADS_PER_BLOCK_ON_COMPUTE_RHS_1)
		gpu_config_string += "%29s\t%25d\n" % (" sp-compute-rhs-2", THREADS_PER_BLOCK_ON_COMPUTE_RHS_2)
		gpu_config_string += "%29s\t%25d\n" % (" sp-error-norm-1", THREADS_PER_BLOCK_ON_ERROR_NORM_1)
		gpu_config_string += "%29s\t%25d\n" % (" sp-error-norm-2", THREADS_PER_BLOCK_ON_ERROR_NORM_2)
		gpu_config_string += "%29s\t%25d\n" % (" sp-exact-rhs-1", THREADS_PER_BLOCK_ON_EXACT_RHS_1)
		gpu_config_string += "%29s\t%25d\n" % (" sp-exact-rhs-2", THREADS_PER_BLOCK_ON_EXACT_RHS_2)
		gpu_config_string += "%29s\t%25d\n" % (" sp-exact-rhs-3", THREADS_PER_BLOCK_ON_EXACT_RHS_3)
		gpu_config_string += "%29s\t%25d\n" % (" sp-exact-rhs-4", THREADS_PER_BLOCK_ON_EXACT_RHS_4)
		gpu_config_string += "%29s\t%25d\n" % (" sp-initialize", THREADS_PER_BLOCK_ON_INITIALIZE)
		gpu_config_string += "%29s\t%25d\n" % (" sp-rhs-norm-1", THREADS_PER_BLOCK_ON_RHS_NORM_1)
		gpu_config_string += "%29s\t%25d\n" % (" sp-rhs-norm-2", THREADS_PER_BLOCK_ON_RHS_NORM_2)
		gpu_config_string += "%29s\t%25d\n" % (" sp-txinvr", THREADS_PER_BLOCK_ON_TXINVR)
		gpu_config_string += "%29s\t%25d\n" % (" sp-x-solve", THREADS_PER_BLOCK_ON_X_SOLVE)
		gpu_config_string += "%29s\t%25d\n" % (" sp-y-solve", THREADS_PER_BLOCK_ON_Y_SOLVE)
		gpu_config_string += "%29s\t%25d\n" % (" sp-z-solve", THREADS_PER_BLOCK_ON_Z_SOLVE)
	
	c_print_results.c_print_results("SP",
			npbparams.CLASS,
			grid_points[0], 
			grid_points[1],
			grid_points[2],
			niter,
			tmax,
			mflops,
			"          floating point",
			verified,
			device_prop.name,
			gpu_config_string)
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA SP')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_sp_info(args.CLASS)
	set_global_variables()
	
	main()
