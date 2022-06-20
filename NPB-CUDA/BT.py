# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	R. Van der Wijngaart
#	T. Harris
#	M. Yarrow
#	H. Jin
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
AA = 0
BB = 1
CC = 2
M_SIZE = 5

PROFILING_TOTAL_TIME = 0
# new
PROFILING_ADD = 1
PROFILING_RHS_1 = 2
PROFILING_RHS_2 = 3
PROFILING_RHS_3 = 4
PROFILING_RHS_4 = 5
PROFILING_RHS_5 = 6
PROFILING_RHS_6 = 7
PROFILING_RHS_7 = 8
PROFILING_RHS_8 = 9
PROFILING_RHS_9 = 10
PROFILING_X_SOLVE_1 = 11
PROFILING_X_SOLVE_2 = 12
PROFILING_X_SOLVE_3 = 13
PROFILING_Y_SOLVE_1 = 14
PROFILING_Y_SOLVE_2 = 15
PROFILING_Y_SOLVE_3 = 16
PROFILING_Z_SOLVE_1 = 17
PROFILING_Z_SOLVE_2 = 18
PROFILING_Z_SOLVE_3 = 19
# old
PROFILING_EXACT_RHS_1 = 20
PROFILING_EXACT_RHS_2 = 21
PROFILING_EXACT_RHS_3 = 22
PROFILING_EXACT_RHS_4 = 23
PROFILING_ERROR_NORM_1 = 24
PROFILING_ERROR_NORM_2 = 25
PROFILING_INITIALIZE = 26
PROFILING_RHS_NORM_1 = 27
PROFILING_RHS_NORM_2 = 28

us = None
vs = None
ws = None
qs = None
rho_i = None
square = None
forcing = None
u = None
rhs = None
cuf = None
q = None
ue = None
buf = None
fjac = None
njac = None
lhs = None
ce = numpy.empty((5, 13), dtype=numpy.float64())

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
dIMAXm1, dJMAXm1, dKMAXm1 = 0.0, 0.0, 0.0
c1c2, c1c5, c3c4, c1345, coKMAX1 = 0.0, 0.0, 0.0, 0.0, 0.0
c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0
c4dssp, c5dssp, dtdssp = 0.0, 0.0, 0.0
dttx1, dttx2, dtty1, dtty2, dttz1, dttz2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
c2dttx1, c2dtty1, c2dttz1 = 0.0, 0.0, 0.0
comz1, comz4, comz5, comz6 = 0.0, 0.0, 0.0, 0.0
c3c4tx3, c3c4ty3, c3c4tz3 = 0.0, 0.0, 0.0
c2iv, con43, con16 = 0.0, 0.0, 0.0
tmp1, tmp2, tmp3 = 0.0, 0.0, 0.0

grid_points = numpy.empty(3, dtype=numpy.int32)


# GPU variables
us_device = None
vs_device = None 
ws_device = None 
qs_device = None 
rho_i_device = None  
square_device = None 
forcing_device = None  
u_device = None 
rhs_device = None 
lhsA_device = None 
lhsB_device = None  
lhsC_device = None 

const_x_solve_device = None
const_y_solve_device = None
const_z_solve_device = None

# -- new
THREADS_PER_BLOCK_ON_ADD = 0
THREADS_PER_BLOCK_ON_RHS_1 = 0
THREADS_PER_BLOCK_ON_RHS_2 = 0
THREADS_PER_BLOCK_ON_RHS_3 = 0
THREADS_PER_BLOCK_ON_RHS_4 = 0
THREADS_PER_BLOCK_ON_RHS_5 = 0
THREADS_PER_BLOCK_ON_RHS_6 = 0
THREADS_PER_BLOCK_ON_RHS_7 = 0
THREADS_PER_BLOCK_ON_RHS_8 = 0
THREADS_PER_BLOCK_ON_RHS_9 = 0
THREADS_PER_BLOCK_ON_X_SOLVE_1 = 0
THREADS_PER_BLOCK_ON_X_SOLVE_2 = 0
THREADS_PER_BLOCK_ON_X_SOLVE_3 = 0
THREADS_PER_BLOCK_ON_Y_SOLVE_1 = 0
THREADS_PER_BLOCK_ON_Y_SOLVE_2 = 0
THREADS_PER_BLOCK_ON_Y_SOLVE_3 = 0
THREADS_PER_BLOCK_ON_Z_SOLVE_1 = 0
THREADS_PER_BLOCK_ON_Z_SOLVE_2 = 0
THREADS_PER_BLOCK_ON_Z_SOLVE_3 = 0
# -- old
THREADS_PER_BLOCK_ON_EXACT_RHS_1 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_2 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_3 = 0
THREADS_PER_BLOCK_ON_EXACT_RHS_4 = 0
THREADS_PER_BLOCK_ON_ERROR_NORM_1 = 0
THREADS_PER_BLOCK_ON_ERROR_NORM_2 = 0
THREADS_PER_BLOCK_ON_INITIALIZE= 0
THREADS_PER_BLOCK_ON_RHS_NORM_1= 0
THREADS_PER_BLOCK_ON_RHS_NORM_2= 0

stream = 0
size_shared_data_empty = 0

gpu_device_id = 0
total_devices = 0
device_prop = None


def set_global_variables():
	global IMAX, JMAX, KMAX, IMAXP, JMAXP
	global us, vs, ws, qs, rho_i, square
	global forcing, u, rhs
	global cuf, q, ue, buf
	global fjac, njac
	global lhs
	
	IMAX = npbparams.PROBLEM_SIZE
	JMAX = npbparams.PROBLEM_SIZE
	KMAX = npbparams.PROBLEM_SIZE
	IMAXP = int(IMAX/2*2)
	JMAXP = int(JMAX/2*2)
	
	us = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	vs = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	ws = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	qs = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	rho_i = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	square = numpy.zeros((KMAX, JMAXP+1, IMAXP+1), dtype=numpy.float64())
	
	forcing = numpy.zeros((KMAX, JMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	u = numpy.zeros((KMAX, JMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	rhs = numpy.zeros((KMAX, JMAXP+1, IMAXP+1, 5), dtype=numpy.float64())
	
	cuf = numpy.empty(npbparams.PROBLEM_SIZE+1, dtype=numpy.float64())
	q = numpy.empty(npbparams.PROBLEM_SIZE+1, dtype=numpy.float64())
	ue = numpy.empty((npbparams.PROBLEM_SIZE+1, 5), dtype=numpy.float64())
	buf = numpy.empty((npbparams.PROBLEM_SIZE+1, 5), dtype=numpy.float64())
	
	fjac = numpy.zeros((npbparams.PROBLEM_SIZE+1, 5, 5), dtype=numpy.float64())
	njac = numpy.zeros((npbparams.PROBLEM_SIZE+1, 5, 5), dtype=numpy.float64())
	
	lhs = numpy.zeros((npbparams.PROBLEM_SIZE+1, 3, 5, 5), dtype=numpy.float64())
#END set_global_variables()


def set_constants():
	global ce
	global c1, c2, c3, c4, c5
	global dIMAXm1, dJMAXm1, dKMAXm1
	global c1c2, c1c5, c3c4, c1345, coKMAX1
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
	
	global const_x_solve_device, const_y_solve_device, const_z_solve_device
	
	ce[0][0] = 2.0
	ce[0][1] = 0.0
	ce[0][2] = 0.0
	ce[0][3] = 4.0
	ce[0][4] = 5.0
	ce[0][5] = 3.0
	ce[0][6] = 0.5
	ce[0][7] = 0.02
	ce[0][8] = 0.01
	ce[0][9] = 0.03
	ce[0][10] = 0.5
	ce[0][11] = 0.4
	ce[0][12] = 0.3
	# -------
	ce[1][0] = 1.0
	ce[1][1] = 0.0
	ce[1][2] = 0.0
	ce[1][3] = 0.0
	ce[1][4] = 1.0
	ce[1][5] = 2.0
	ce[1][6] = 3.0
	ce[1][7] = 0.01
	ce[1][8] = 0.03
	ce[1][9] = 0.02
	ce[1][10] = 0.4
	ce[1][11] = 0.3
	ce[1][12] = 0.5
	# -------
	ce[2][0] = 2.0
	ce[2][1] = 2.0
	ce[2][2] = 0.0
	ce[2][3] = 0.0
	ce[2][4] = 0.0
	ce[2][5] = 2.0
	ce[2][6] = 3.0
	ce[2][7] = 0.04
	ce[2][8] = 0.03
	ce[2][9] = 0.05
	ce[2][10] = 0.3
	ce[2][11] = 0.5
	ce[2][12] = 0.4
	# -------
	ce[3][0] = 2.0
	ce[3][1] = 2.0
	ce[3][2] = 0.0
	ce[3][3] = 0.0
	ce[3][4] = 0.0
	ce[3][5] = 2.0
	ce[3][6] = 3.0
	ce[3][7] = 0.03
	ce[3][8] = 0.05
	ce[3][9] = 0.04
	ce[3][10] = 0.2
	ce[3][11] = 0.1
	ce[3][12] = 0.3
	# -------
	ce[4][0] = 5.0
	ce[4][1] = 4.0
	ce[4][2] = 3.0
	ce[4][3] = 2.0
	ce[4][4] = 0.1
	ce[4][5] = 0.4
	ce[4][6] = 0.3
	ce[4][7] = 0.05
	ce[4][8] = 0.04
	ce[4][9] = 0.03
	ce[4][10] = 0.1
	ce[4][11] = 0.3
	ce[4][12] = 0.2
	# -------
	c1 = 1.4
	c2 = 0.4
	c3 = 0.1
	c4 = 1.0
	c5 = 1.4
	
	dIMAXm1 = 1.0 / (grid_points[0]-1)
	dJMAXm1 = 1.0 / (grid_points[1]-1)
	dKMAXm1 = 1.0 / (grid_points[2]-1)
	
	c1c2 = c1 * c2
	c1c5 = c1 * c5
	c3c4 = c3 * c4
	c1345 = c1c5 * c3c4
	
	coKMAX1 = (1.0-c1c5)
	
	tx1 = 1.0 / (dIMAXm1*dIMAXm1)
	tx2 = 1.0 / (2.0*dIMAXm1)
	tx3 = 1.0 / dIMAXm1
	
	ty1 = 1.0 / (dJMAXm1*dJMAXm1)
	ty2 = 1.0 / (2.0*dJMAXm1)
	ty3 = 1.0 / dJMAXm1
	
	tz1 = 1.0 / (dKMAXm1*dKMAXm1)
	tz2 = 1.0 / (2.0*dKMAXm1)
	tz3 = 1.0 / dKMAXm1
	
	dx1 = 0.75
	dx2 = 0.75
	dx3 = 0.75
	dx4 = 0.75
	dx5 = 0.75
	
	dy1 = 0.75
	dy2 = 0.75
	dy3 = 0.75
	dy4 = 0.75
	dy5 = 0.75
	
	dz1 = 1.0 
	dz2 = 1.0 
	dz3 = 1.0 
	dz4 = 1.0 
	dz5 = 1.0 
	
	dxmax = max(dx3, dx4)
	dymax = max(dy2, dy4)
	dzmax = max(dz2, dz3)
	
	dssp = 0.25 * max(dx1, max(dy1, dz1))
	
	c4dssp = 4.0 * dssp
	c5dssp = 5.0 * dssp
	
	dttx1 = dt * tx1
	dttx2 = dt * tx2
	dtty1 = dt * ty1
	dtty2 = dt * ty2
	dttz1 = dt * tz1
	dttz2 = dt * tz2
	
	c2dttx1 = 2.0 * dttx1
	c2dtty1 = 2.0 * dtty1
	c2dttz1 = 2.0 * dttz1

	dtdssp = dt * dssp
	
	comz1 = dtdssp
	comz4 = 4.0 * dtdssp
	comz5 = 5.0 * dtdssp
	comz6 = 6.0 * dtdssp
	
	c3c4tx3 = c3c4 * tx3
	c3c4ty3 = c3c4 * ty3
	c3c4tz3 = c3c4 * tz3
	
	dx1tx1 = dx1 * tx1
	dx2tx1 = dx2 * tx1
	dx3tx1 = dx3 * tx1
	dx4tx1 = dx4 * tx1
	dx5tx1 = dx5 * tx1
	
	dy1ty1 = dy1 * ty1
	dy2ty1 = dy2 * ty1
	dy3ty1 = dy3 * ty1
	dy4ty1 = dy4 * ty1
	dy5ty1 = dy5 * ty1
	
	dz1tz1 = dz1 * tz1
	dz2tz1 = dz2 * tz1
	dz3tz1 = dz3 * tz1
	dz4tz1 = dz4 * tz1
	dz5tz1 = dz5 * tz1
	
	c2iv = 2.5
	con43 = 4.0 / 3.0
	con16 = 1.0 / 6.0
	
	xxcon1 = c3c4tx3 * con43 * tx3
	xxcon2 = c3c4tx3 * tx3
	xxcon3 = c3c4tx3 * coKMAX1 * tx3
	xxcon4 = c3c4tx3 * con16 * tx3
	xxcon5 = c3c4tx3 * c1c5 * tx3
	
	yycon1 = c3c4ty3 * con43 * ty3
	yycon2 = c3c4ty3 * ty3
	yycon3 = c3c4ty3 * coKMAX1 * ty3
	yycon4 = c3c4ty3 * con16 * ty3
	yycon5 = c3c4ty3 * c1c5 * ty3
	
	zzcon1 = c3c4tz3 * con43 * tz3
	zzcon2 = c3c4tz3 * tz3
	zzcon3 = c3c4tz3 * coKMAX1 * tz3
	zzcon4 = c3c4tz3 * con16 * tz3
	zzcon5 = c3c4tz3 * c1c5 * tz3
	
	#Constant arrays to GPU memory
	const_x_solve = numpy.array([dt, tx1, tx2, dx1, dx2, dx3, dx4, dx5, c1, c2, con43, c3c4, c1345], numpy.float64)
	const_x_solve_device = cuda.to_device(const_x_solve)
	
	const_y_solve = numpy.array([dt, ty1, ty2, dy1, dy2, dy3, dy4, dy5, c1, c2, con43, c3c4, c1345], numpy.float64)
	const_y_solve_device = cuda.to_device(const_y_solve)
	
	const_z_solve = numpy.array([dt, tz1, tz2, dz1, dz2, dz3, dz4, dz5, c1, c2, c3, c4, con43, c3c4, c1345], numpy.float64)
	const_z_solve_device = cuda.to_device(const_z_solve)
	
	#Another constant values are going to be passed to kernels as parameters
#END set_constants()


def setup_gpu():
	global gpu_device_id, total_devices
	global device_prop
	
	global THREADS_PER_BLOCK_ON_ADD
	
	global THREADS_PER_BLOCK_ON_RHS_1, THREADS_PER_BLOCK_ON_RHS_2, THREADS_PER_BLOCK_ON_RHS_3
	global THREADS_PER_BLOCK_ON_RHS_4, THREADS_PER_BLOCK_ON_RHS_5, THREADS_PER_BLOCK_ON_RHS_6
	global THREADS_PER_BLOCK_ON_RHS_7, THREADS_PER_BLOCK_ON_RHS_8, THREADS_PER_BLOCK_ON_RHS_9
	
	global THREADS_PER_BLOCK_ON_X_SOLVE_1, THREADS_PER_BLOCK_ON_X_SOLVE_2, THREADS_PER_BLOCK_ON_X_SOLVE_3
	global THREADS_PER_BLOCK_ON_Y_SOLVE_1, THREADS_PER_BLOCK_ON_Y_SOLVE_2, THREADS_PER_BLOCK_ON_Y_SOLVE_3
	global THREADS_PER_BLOCK_ON_Z_SOLVE_1, THREADS_PER_BLOCK_ON_Z_SOLVE_2, THREADS_PER_BLOCK_ON_Z_SOLVE_3
	
	global THREADS_PER_BLOCK_ON_EXACT_RHS_1, THREADS_PER_BLOCK_ON_EXACT_RHS_2
	global THREADS_PER_BLOCK_ON_EXACT_RHS_3, THREADS_PER_BLOCK_ON_EXACT_RHS_4
	global THREADS_PER_BLOCK_ON_ERROR_NORM_1, THREADS_PER_BLOCK_ON_ERROR_NORM_2
	global THREADS_PER_BLOCK_ON_INITIALIZE
	global THREADS_PER_BLOCK_ON_RHS_NORM_1, THREADS_PER_BLOCK_ON_RHS_NORM_2
	
	global us_device, vs_device, ws_device, qs_device, rho_i_device, square_device, rhs_device
	global lhsA_device, lhsB_device, lhsC_device
	
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
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_ADD
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ADD = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ADD = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_4
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_4 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_4 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_5
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_5 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_5 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_6
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_6 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_6 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_7
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_7 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_7 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_8
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_8 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_8 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_9
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_9 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_9 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_X_SOLVE_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_X_SOLVE_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_X_SOLVE_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_X_SOLVE_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_X_SOLVE_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_X_SOLVE_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_X_SOLVE_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_X_SOLVE_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_X_SOLVE_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_Y_SOLVE_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Y_SOLVE_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Y_SOLVE_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_Y_SOLVE_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Y_SOLVE_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Y_SOLVE_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_Y_SOLVE_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Y_SOLVE_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Y_SOLVE_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_Z_SOLVE_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Z_SOLVE_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Z_SOLVE_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_Z_SOLVE_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Z_SOLVE_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Z_SOLVE_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_Z_SOLVE_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_Z_SOLVE_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_Z_SOLVE_3 = device_prop.WARP_SIZE
		
	# -- old
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_EXACT_RHS_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_EXACT_RHS_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_EXACT_RHS_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_3 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_EXACT_RHS_4
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_EXACT_RHS_4 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_EXACT_RHS_4 = device_prop.WARP_SIZE

	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_ERROR_NORM_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERROR_NORM_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERROR_NORM_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_ERROR_NORM_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_ERROR_NORM_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_ERROR_NORM_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_INITIALIZE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_INITIALIZE = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_INITIALIZE = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_NORM_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_NORM_1 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_NORM_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.BT_THREADS_PER_BLOCK_ON_RHS_NORM_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		THREADS_PER_BLOCK_ON_RHS_NORM_2 = aux_threads_per_block
	else: 
		THREADS_PER_BLOCK_ON_RHS_NORM_2 = device_prop.WARP_SIZE
		
	n_float64 = numpy.float64
	#forcing_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1, 5), n_float64)
	#u_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1, 5), n_float64)
	rhs_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1, 5), n_float64)
	us_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1), n_float64)
	vs_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1), n_float64)
	ws_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1), n_float64)
	qs_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1), n_float64)
	rho_i_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1), n_float64)
	square_device = cuda.device_array((npbparams.PROBLEM_SIZE, JMAXP+1, IMAXP+1), n_float64)
	lhsA_device = cuda.device_array((5, 5, npbparams.PROBLEM_SIZE+1, npbparams.PROBLEM_SIZE+1, npbparams.PROBLEM_SIZE+1), n_float64)
	lhsB_device = cuda.device_array((5, 5, npbparams.PROBLEM_SIZE+1, npbparams.PROBLEM_SIZE+1, npbparams.PROBLEM_SIZE+1), n_float64)
	lhsC_device = cuda.device_array((5, 5, npbparams.PROBLEM_SIZE+1, npbparams.PROBLEM_SIZE+1, npbparams.PROBLEM_SIZE+1), n_float64)
#END setup_gpu()


def round_amount_of_work(amount_of_work, amount_of_threads):
	rest = amount_of_work % amount_of_threads
	return amount_of_work if rest == 0 else (amount_of_work + amount_of_threads - rest)
#END round_amount_of_work()


#*****************************************************************
#************************* GPU FUNCTIONS *************************
#*****************************************************************
@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, int32)')
def add_gpu_kernel(u_device, 
				   rhs_device,
				   KMAX, JMAX, IMAX, PROBLEM_SIZE):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z+1
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	t_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = int(t_i / 5) + 1
	m = int(t_i % 5)

	if k > KMAX-2 or j+1 < 1 or j+1 > JMAX-2 or j >= PROBLEM_SIZE or i > IMAX-2:
		return

	j = j +1

	#u_device[((k * (JMAXP+1) + j) * (IMAXP+1) + i) * 5 + m] += rhs_device[((k * (JMAXP+1) + j) * (IMAXP+1) + i) * 5 + m];
	u_device[k, j, i, m] += rhs_device[k, j, i, m]
#END add_gpu_kernel()


# ---------------------------------------------------------------------
# addition of update to the vector u
# ---------------------------------------------------------------------
def add_gpu(u_device,
			rhs_device):
	amount_of_threads = [THREADS_PER_BLOCK_ON_ADD, 1, 1]
	amount_of_work    = [(grid_points[0]-2)*5, npbparams.PROBLEM_SIZE, grid_points[2]-2]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_ADD)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	add_gpu_kernel[blockSize, 
				threadSize,
				stream,
				size_shared_data_empty](u_device,
										rhs_device,
										KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_ADD)
#END add_gpu()


# ---------------------------------------------------------------------
# this function computes the left hand side for the three z-factors   
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], int32, int32, int32, int32)')
def z_solve_gpu_kernel_3(rhs,
						lhsA,
						lhsB,  
						lhsC,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	#extern __shared__ double tmp_l_lhs[];
	#double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];
	tmp_l_lhs = cuda.shared.array(shape=0, dtype=numba.float64)
	idx = cuda.blockDim.x * 3 * 5 * 5
	tmp_l_r = tmp_l_lhs[idx:]
	
	t_j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
	j = int(t_j / 5)
	m = int(t_j % 5)
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x+1
	l_i = cuda.threadIdx.x
	if j+1 < 1 or j+1 > JMAX - 2 or j >= PROBLEM_SIZE or i > IMAX - 2:
		return

	j = j+1

	#double (*tmp2_l_lhs)[3][5][5] = (double(*)[3][5][5])tmp_l_lhs;
	#double (*l_lhs)[5][5] = tmp2_l_lhs[l_i];
	D1, D2, D3 = 3, 5, 5
	l_lhs = tmp_l_lhs[(l_i*D1*D2*D3):] 
	
	#double (*tmp2_l_r)[2][5] = (double(*)[2][5])tmp_l_r;
	#double (*l_r)[5] = tmp2_l_r[l_i]; 
	M1, M2 = 2, 5
	l_r = tmp_l_r[(l_i*M1*M2):]

	ksize = KMAX - 1

	# ---------------------------------------------------------------------
	# compute the indices for storing the block-diagonal matrix;
	# determine c (labeled f) and s jacobians   
	# ---------------------------------------------------------------------
	# performs guaussian elimination on this cell.
	# ---------------------------------------------------------------------
	# assumes that unpacking routines for non-first cells 
	# preload C' and rhs' from previous cell.
	# ---------------------------------------------------------------------
	# assumed send happens outside this routine, but that
	# c'(KMAX) and rhs'(KMAX) will be sent to next cell.
	# ---------------------------------------------------------------------
	# outer most do loops - sweeping in i direction
	# ---------------------------------------------------------------------

	# load data
	for p in range(5):
		l_lhs[(BB*D2+p)*D3 +m] = lhsB[p, m, 0, j, i-1] #l_lhs[BB][p][m] to l_lhs[(BB*D2+p)*D3 +m]
		l_lhs[(CC*D2+p)*D3 +m] = lhsC[p, m, 0, j, i-1]

	l_r[1*M2+m] = rhs[0, j, i, m]

	cuda.syncthreads()

	# ---------------------------------------------------------------------
	# multiply c[0][j][i] by b_inverse and copy back to c
	# multiply rhs(0) by b_inverse(0) and copy to rhs
	# ---------------------------------------------------------------------
	for p in range(5):
		pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
		if m>p and m<5:
			l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
		if m<5:
			l_lhs[(CC*D2+m)*D3 +p] = l_lhs[(CC*D2+m)*D3 +p]*pivot
		if p==m:
			l_r[1*M2+p] = l_r[1*M2+p]*pivot

		cuda.syncthreads()

		if p != m:
			coeff = l_lhs[(BB*D2+p)*D3 +m]
			for n in range(p+1, 5):
				l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
			for n in range(5):
				l_lhs[(CC*D2+n)*D3 +m] = l_lhs[(CC*D2+n)*D3 +m] - coeff*l_lhs[(CC*D2+n)*D3 +p]
			l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]

		cuda.syncthreads()

	# update data
	rhs[0, j, i, m] = l_r[1*M2+m]

	# ---------------------------------------------------------------------
	# begin inner most do loop
	# do all the elements of the cell unless last 
	# ---------------------------------------------------------------------
	for k in range(1, ksize):
		# load data
		for n in range(5):
			l_lhs[(AA*D2+n)*D3 +m] = lhsA[n, m, k, j, i-1]
			l_lhs[(BB*D2+n)*D3 +m] = lhsB[n, m, k, j, i-1]

		l_r[0*M2+m] = l_r[1*M2+m]
		l_r[1*M2+m] = rhs[k, j, i, m]

		cuda.syncthreads()

		# ---------------------------------------------------------------------
		# subtract A*lhs_vector(k-1) from lhs_vector(k)
		# 
		# rhs(k) = rhs(k) - A*rhs(k-1)
		# ---------------------------------------------------------------------
		l_r[1*M2+m] = ( l_r[1*M2+m] - l_lhs[(AA*D2+0)*D3 +m]*l_r[0*M2+0]
			- l_lhs[(AA*D2+1)*D3 +m]*l_r[0*M2+1]
			- l_lhs[(AA*D2+2)*D3 +m]*l_r[0*M2+2]
			- l_lhs[(AA*D2+3)*D3 +m]*l_r[0*M2+3]
			- l_lhs[(AA*D2+4)*D3 +m]*l_r[0*M2+4] )

		# ---------------------------------------------------------------------
		# B(k) = B(k) - C(k-1)*A(k)
		# matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
		# ---------------------------------------------------------------------
		for p in range(5):
			l_lhs[(BB*D2+m)*D3 +p] = ( l_lhs[(BB*D2+m)*D3 +p] - l_lhs[(AA*D2+0)*D3 +p]*l_lhs[(CC*D2+m)*D3 +0]
				- l_lhs[(AA*D2+1)*D3 +p]*l_lhs[(CC*D2+m)*D3 +1]
				- l_lhs[(AA*D2+2)*D3 +p]*l_lhs[(CC*D2+m)*D3 +2]
				- l_lhs[(AA*D2+3)*D3 +p]*l_lhs[(CC*D2+m)*D3 +3]
				- l_lhs[(AA*D2+4)*D3 +p]*l_lhs[(CC*D2+m)*D3 +4] )

		cuda.syncthreads()

		# load data
		for n in range(5):
			l_lhs[(CC*D2+n)*D3 +m] = lhsC[n, m, k, j, i-1]

		cuda.syncthreads()

		# ---------------------------------------------------------------------
		# multiply c[k][j][i] by b_inverse and copy back to c
		# multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
		# ---------------------------------------------------------------------
		for p in range(5):
			pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
			if m>p and m<5:
				l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
			if m<5:
				l_lhs[(CC*D2+m)*D3 +p] = l_lhs[(CC*D2+m)*D3 +p]*pivot
			if p==m:
				l_r[1*M2+p] = l_r[1*M2+p]*pivot

			cuda.syncthreads()

			if p != m:
				coeff = l_lhs[(BB*D2+p)*D3 +m]
				for n in range(p+1, 5):
					l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
				for n in range(5):
					l_lhs[(CC*D2+n)*D3 +m] = l_lhs[(CC*D2+n)*D3 +m] - coeff*l_lhs[(CC*D2+n)*D3 +p]
				l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]

			cuda.syncthreads()

		# update data
		for n in range(5):
			lhsC[n, m, k, j, i-1] = l_lhs[(CC*D2+n)*D3 +m]

		rhs[k, j, i, m] = l_r[1*M2+m]
	#END for k in range(1, ksize):

	k = k + 1 #Increment k after loop

	# ---------------------------------------------------------------------
	# now finish up special cases for last cell
	# ---------------------------------------------------------------------
	# load data
	for n in range(5):
		l_lhs[(AA*D2+n)*D3 +m] = lhsA[n, m, k, j, i-1]
		l_lhs[(BB*D2+n)*D3 +m] = lhsB[n, m, k, j, i-1]

	l_r[0*M2+m] = l_r[1*M2+m]
	l_r[1*M2+m] = rhs[k, j, i, m]

	cuda.syncthreads()

	# ---------------------------------------------------------------------
	# rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
	# ---------------------------------------------------------------------
	l_r[1*M2+m] = ( l_r[1*M2+m] - l_lhs[(AA*D2+0)*D3 +m]*l_r[0*M2+0]
		- l_lhs[(AA*D2+1)*D3 +m]*l_r[0*M2+1]
		- l_lhs[(AA*D2+2)*D3 +m]*l_r[0*M2+2]
		- l_lhs[(AA*D2+3)*D3 +m]*l_r[0*M2+3]
		- l_lhs[(AA*D2+4)*D3 +m]*l_r[0*M2+4] )

	# ---------------------------------------------------------------------
	# B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
	# matmul_sub(AA,i,j,ksize,c,CC,i,j,ksize-1,c,BB,i,j,ksize)
	# ---------------------------------------------------------------------
	for p in range(5):
		l_lhs[(BB*D2+m)*D3 +p] = ( l_lhs[(BB*D2+m)*D3 +p] - l_lhs[(AA*D2+0)*D3 +p]*l_lhs[(CC*D2+m)*D3 +0]
			- l_lhs[(AA*D2+1)*D3 +p]*l_lhs[(CC*D2+m)*D3 +1]
			- l_lhs[(AA*D2+2)*D3 +p]*l_lhs[(CC*D2+m)*D3 +2]
			- l_lhs[(AA*D2+3)*D3 +p]*l_lhs[(CC*D2+m)*D3 +3]
			- l_lhs[(AA*D2+4)*D3 +p]*l_lhs[(CC*D2+m)*D3 +4] )

	# ---------------------------------------------------------------------
	# multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
	# ---------------------------------------------------------------------
	for p in range(5):
		pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
		if m>p and m<5:
			l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
		if p==m:
			l_r[1*M2+p] = l_r[1*M2+p]*pivot

		cuda.syncthreads()

		if p != m:
			coeff = l_lhs[(BB*D2+p)*D3 +m]
			for n in range(p+1, 5):
				l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
			l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]

		cuda.syncthreads()

	# update data
	rhs[k, j, i, m] = l_r[1*M2+m]

	cuda.syncthreads()

	# ---------------------------------------------------------------------
	# back solve: if last cell, then generate U(ksize)=rhs(ksize)
	# else assume U(ksize) is loaded in un pack backsub_info
	# so just use it
	# after u(kstart) will be sent to next cell
	# ---------------------------------------------------------------------
	for k in range(ksize-1, -1, -1):
		for n in range(M_SIZE):
			rhs[k, j, i, m] = rhs[k, j, i, m] - lhsC[n, m, k, j, i-1]*rhs[k+1, j, i, n]

		cuda.syncthreads()
#END z_solve_gpu_kernel_3()


# ---------------------------------------------------------------------
# this function computes the left hand side for the three z-factors   
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :], float64[:], float64, float64, float64, float64)', device=True)
def z_solve_gpu_device_fjac(l_fjac,
							t_u,
							square, 
							qs,
							c1, c2):
	tmp1 = 1.0 / t_u[0]
	tmp2 = tmp1 * tmp1

	l_fjac[0][0] = 0.0
	l_fjac[1][0] = 0.0
	l_fjac[2][0] = 0.0
	l_fjac[3][0] = 1.0
	l_fjac[4][0] = 0.0

	l_fjac[0][1] = - ( t_u[1]*t_u[3] ) * tmp2
	l_fjac[1][1] = t_u[3] * tmp1
	l_fjac[2][1] = 0.0
	l_fjac[3][1] = t_u[1] * tmp1
	l_fjac[4][1] = 0.0

	l_fjac[0][2] = - ( t_u[2]*t_u[3] ) * tmp2
	l_fjac[1][2] = 0.0
	l_fjac[2][2] = t_u[3] * tmp1
	l_fjac[3][2] = t_u[2] * tmp1
	l_fjac[4][2] = 0.0

	l_fjac[0][3] = ( - (t_u[3]*t_u[3] * tmp2 ) 
		+ c2 * qs )
	l_fjac[1][3] = - c2 *  t_u[1] * tmp1
	l_fjac[2][3] = - c2 *  t_u[2] * tmp1
	l_fjac[3][3] = ( 2.0 - c2 ) *  t_u[3] * tmp1
	l_fjac[4][3] = c2

	l_fjac[0][4] = ( ( c2 * 2.0 * square - c1 * t_u[4] )
		* t_u[3] * tmp2 )
	l_fjac[1][4] = - c2 * ( t_u[1]*t_u[3] ) * tmp2
	l_fjac[2][4] = - c2 * ( t_u[2]*t_u[3] ) * tmp2
	l_fjac[3][4] = ( c1 * ( t_u[4] * tmp1 )
		- c2 * ( qs + t_u[3]*t_u[3] * tmp2 ) )
	l_fjac[4][4] = c1 * t_u[3] * tmp1
#END z_solve_gpu_device_fjac()


@cuda.jit('void(float64[:, :], float64[:], float64, float64, float64, float64, float64)', device=True)
def z_solve_gpu_device_njac(l_njac,
							t_u,
							c3, c4, c3c4, con43, c1345):
	tmp1 = 1.0 / t_u[0]
	tmp2 = tmp1 * tmp1
	tmp3 = tmp1 * tmp2

	l_njac[0][0] = 0.0
	l_njac[1][0] = 0.0
	l_njac[2][0] = 0.0
	l_njac[3][0] = 0.0
	l_njac[4][0] = 0.0

	l_njac[0][1] = - c3c4 * tmp2 * t_u[1]
	l_njac[1][1] = c3c4 * tmp1
	l_njac[2][1] = 0.0
	l_njac[3][1] = 0.0
	l_njac[4][1] = 0.0

	l_njac[0][2] = - c3c4 * tmp2 * t_u[2]
	l_njac[1][2] = 0.0
	l_njac[2][2] = c3c4 * tmp1
	l_njac[3][2] = 0.0
	l_njac[4][2] = 0.0

	l_njac[0][3] = - con43 * c3c4 * tmp2 * t_u[3]
	l_njac[1][3] = 0.0
	l_njac[2][3] = 0.0
	l_njac[3][3] = con43 * c3 * c4 * tmp1
	l_njac[4][3] = 0.0

	l_njac[0][4] = ( - (  c3c4
			- c1345 ) * tmp3 * (t_u[1]*t_u[1])
		- ( c3c4 - c1345 ) * tmp3 * (t_u[2]*t_u[2])
		- ( con43 * c3c4
				- c1345 ) * tmp3 * (t_u[3]*t_u[3])
		- c1345 * tmp2 * t_u[4] )

	l_njac[1][4] = (  c3c4 - c1345 ) * tmp2 * t_u[1]
	l_njac[2][4] = (  c3c4 - c1345 ) * tmp2 * t_u[2]
	l_njac[3][4] = ( con43 * c3c4
			- c1345 ) * tmp2 * t_u[3]
	l_njac[4][4] = ( c1345 )* tmp1
#END z_solve_gpu_device_njac()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:], int32, int32, int32, int32)')
def z_solve_gpu_kernel_2(qs, 
						square, 
						u,
						lhsA, 
						lhsB,
						lhsC,
						const_arr,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	k = cuda.blockDim.z * cuda.blockIdx.z + cuda.threadIdx.z + 1
	j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x + 1

	if k > KMAX-2 or j+1 < 1 or j+1 > JMAX-2 or j >= PROBLEM_SIZE or i > IMAX-2:
		return

	j = j+1

	dt, tz1, tz2 = const_arr[0], const_arr[1], const_arr[2]
	dz1, dz2, dz3, dz4, dz5 = const_arr[3], const_arr[4], const_arr[5], const_arr[6], const_arr[7]
	c1, c2, c3, c4, con43, c3c4, c1345 = const_arr[8], const_arr[9], const_arr[10], const_arr[11], const_arr[12], const_arr[13], const_arr[14]
	
	fjac = cuda.local.array((5, 5), numba.float64)
	njac = cuda.local.array((5, 5), numba.float64)
	t_u = cuda.local.array(5, numba.float64)
	
	# ---------------------------------------------------------------------
	# compute the indices for storing the block-diagonal matrix;
	# determine c (labeled f) and s jacobians
	# ---------------------------------------------------------------------
	tmp1 = dt * tz1
	tmp2 = dt * tz2

	for m in range(5):
		t_u[m] = u[k-1, j, i, m]

	z_solve_gpu_device_fjac(fjac, t_u, square[k-1, j, i], qs[k-1, j, i], c1, c2)
	z_solve_gpu_device_njac(njac, t_u, c3, c4, c3c4, con43, c1345)

	lhsA[0, 0, k, j, i-1] = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dz1 
	lhsA[1, 0, k, j, i-1] = - tmp2 * fjac[1][0] - tmp1 * njac[1][0]
	lhsA[2, 0, k, j, i-1] = - tmp2 * fjac[2][0] - tmp1 * njac[2][0]
	lhsA[3, 0, k, j, i-1] = - tmp2 * fjac[3][0] - tmp1 * njac[3][0]
	lhsA[4, 0, k, j, i-1] = - tmp2 * fjac[4][0] - tmp1 * njac[4][0]

	lhsA[0, 1, k, j, i-1] = - tmp2 * fjac[0][1] - tmp1 * njac[0][1]
	lhsA[1, 1, k, j, i-1] = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dz2
	lhsA[2, 1, k, j, i-1] = - tmp2 * fjac[2][1] - tmp1 * njac[2][1]
	lhsA[3, 1, k, j, i-1] = - tmp2 * fjac[3][1] - tmp1 * njac[3][1]
	lhsA[4, 1, k, j, i-1] = - tmp2 * fjac[4][1] - tmp1 * njac[4][1]

	lhsA[0, 2, k, j, i-1] = - tmp2 * fjac[0][2] - tmp1 * njac[0][2]
	lhsA[1, 2, k, j, i-1] = - tmp2 * fjac[1][2] - tmp1 * njac[1][2]
	lhsA[2, 2, k, j, i-1] = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dz3
	lhsA[3, 2, k, j, i-1] = - tmp2 * fjac[3][2] - tmp1 * njac[3][2]
	lhsA[4, 2, k, j, i-1] = - tmp2 * fjac[4][2] - tmp1 * njac[4][2]

	lhsA[0, 3, k, j, i-1] = - tmp2 * fjac[0][3] - tmp1 * njac[0][3]
	lhsA[1, 3, k, j, i-1] = - tmp2 * fjac[1][3] - tmp1 * njac[1][3]
	lhsA[2, 3, k, j, i-1] = - tmp2 * fjac[2][3] - tmp1 * njac[2][3]
	lhsA[3, 3, k, j, i-1] = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dz4
	lhsA[4, 3, k, j, i-1] = - tmp2 * fjac[4][3] - tmp1 * njac[4][3]

	lhsA[0, 4, k, j, i-1] = - tmp2 * fjac[0][4] - tmp1 * njac[0][4]
	lhsA[1, 4, k, j, i-1] = - tmp2 * fjac[1][4] - tmp1 * njac[1][4]
	lhsA[2, 4, k, j, i-1] = - tmp2 * fjac[2][4] - tmp1 * njac[2][4]
	lhsA[3, 4, k, j, i-1] = - tmp2 * fjac[3][4] - tmp1 * njac[3][4]
	lhsA[4, 4, k, j, i-1] = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dz5
	
	for m in range(5):
		t_u[m] = u[k, j, i, m]
	z_solve_gpu_device_njac(njac, t_u, c3, c4, c3c4, con43, c1345)
    
	lhsB[0, 0, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * dz1
	lhsB[1, 0, k, j, i-1] = tmp1 * 2.0 * njac[1][0]
	lhsB[2, 0, k, j, i-1] = tmp1 * 2.0 * njac[2][0]
	lhsB[3, 0, k, j, i-1] = tmp1 * 2.0 * njac[3][0]
	lhsB[4, 0, k, j, i-1] = tmp1 * 2.0 * njac[4][0]
    
	lhsB[0, 1, k, j, i-1] = tmp1 * 2.0 * njac[0][1]
	lhsB[1, 1, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * dz2
	lhsB[2, 1, k, j, i-1] = tmp1 * 2.0 * njac[2][1]
	lhsB[3, 1, k, j, i-1] = tmp1 * 2.0 * njac[3][1]
	lhsB[4, 1, k, j, i-1] = tmp1 * 2.0 * njac[4][1]
    
	lhsB[0, 2, k, j, i-1] = tmp1 * 2.0 * njac[0][2]
	lhsB[1, 2, k, j, i-1] = tmp1 * 2.0 * njac[1][2]
	lhsB[2, 2, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * dz3
	lhsB[3, 2, k, j, i-1] = tmp1 * 2.0 * njac[3][2]
	lhsB[4, 2, k, j, i-1] = tmp1 * 2.0 * njac[4][2]
    
	lhsB[0, 3, k, j, i-1] = tmp1 * 2.0 * njac[0][3]
	lhsB[1, 3, k, j, i-1] = tmp1 * 2.0 * njac[1][3]
	lhsB[2, 3, k, j, i-1] = tmp1 * 2.0 * njac[2][3]
	lhsB[3, 3, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * dz4
	lhsB[4, 3, k, j, i-1] = tmp1 * 2.0 * njac[4][3]
    
	lhsB[0, 4, k, j, i-1] = tmp1 * 2.0 * njac[0][4]
	lhsB[1, 4, k, j, i-1] = tmp1 * 2.0 * njac[1][4]
	lhsB[2, 4, k, j, i-1] = tmp1 * 2.0 * njac[2][4]
	lhsB[3, 4, k, j, i-1] = tmp1 * 2.0 * njac[3][4]
	lhsB[4, 4, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * dz5
	
	for m in range(5):
		t_u[m] = u[k+1, j, i, m]
    
	z_solve_gpu_device_fjac(fjac, t_u, square[k+1, j, i], qs[k+1, j, i], c1, c2)
	z_solve_gpu_device_njac(njac, t_u, c3, c4, c3c4, con43, c1345)
    
	lhsC[0, 0, k, j, i-1] =  tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dz1
	lhsC[1, 0, k, j, i-1] =  tmp2 * fjac[1][0] - tmp1 * njac[1][0]
	lhsC[2, 0, k, j, i-1] =  tmp2 * fjac[2][0] - tmp1 * njac[2][0]
	lhsC[3, 0, k, j, i-1] =  tmp2 * fjac[3][0] - tmp1 * njac[3][0]
	lhsC[4, 0, k, j, i-1] =  tmp2 * fjac[4][0] - tmp1 * njac[4][0]
    
	lhsC[0, 1, k, j, i-1] =  tmp2 * fjac[0][1] - tmp1 * njac[0][1]
	lhsC[1, 1, k, j, i-1] =  tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dz2
	lhsC[2, 1, k, j, i-1] =  tmp2 * fjac[2][1] - tmp1 * njac[2][1]
	lhsC[3, 1, k, j, i-1] =  tmp2 * fjac[3][1] - tmp1 * njac[3][1]
	lhsC[4, 1, k, j, i-1] =  tmp2 * fjac[4][1] - tmp1 * njac[4][1]
    
	lhsC[0, 2, k, j, i-1] =  tmp2 * fjac[0][2] - tmp1 * njac[0][2]
	lhsC[1, 2, k, j, i-1] =  tmp2 * fjac[1][2] - tmp1 * njac[1][2]
	lhsC[2, 2, k, j, i-1] =  tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dz3
	lhsC[3, 2, k, j, i-1] =  tmp2 * fjac[3][2] - tmp1 * njac[3][2]
	lhsC[4, 2, k, j, i-1] =  tmp2 * fjac[4][2] - tmp1 * njac[4][2]
    
	lhsC[0, 3, k, j, i-1] =  tmp2 * fjac[0][3] - tmp1 * njac[0][3]
	lhsC[1, 3, k, j, i-1] =  tmp2 * fjac[1][3] - tmp1 * njac[1][3]
	lhsC[2, 3, k, j, i-1] =  tmp2 * fjac[2][3] - tmp1 * njac[2][3]
	lhsC[3, 3, k, j, i-1] =  tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dz4
	lhsC[4, 3, k, j, i-1] =  tmp2 * fjac[4][3] - tmp1 * njac[4][3]
    
	lhsC[0, 4, k, j, i-1] =  tmp2 * fjac[0][4] - tmp1 * njac[0][4]
	lhsC[1, 4, k, j, i-1] =  tmp2 * fjac[1][4] - tmp1 * njac[1][4]
	lhsC[2, 4, k, j, i-1] =  tmp2 * fjac[2][4] - tmp1 * njac[2][4]
	lhsC[3, 4, k, j, i-1] =  tmp2 * fjac[3][4] - tmp1 * njac[3][4]
	lhsC[4, 4, k, j, i-1] =  tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dz5
#END z_solve_gpu_kernel_2()


# ---------------------------------------------------------------------
# this function computes the left hand side for the three z-factors   
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], int32, int32, int32, int32)')
def z_solve_gpu_kernel_1(lhsA, 
						lhsB,
						lhsC,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	t_j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
	j = int(t_j % PROBLEM_SIZE)
	mn = int(t_j / PROBLEM_SIZE)
	m = int(mn / 5)
	n = int(mn % 5)
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x+1

	if j+1 < 1 or j+1 > JMAX-2 or j >= PROBLEM_SIZE or i > IMAX-2 or m >= 5:
		return

	j = j+1
	ksize = KMAX - 1

	# ---------------------------------------------------------------------
	# now jacobians set, so form left hand side in z direction
	# ---------------------------------------------------------------------
	lhsA[m, n, 0, j, i-1] = 0.0
	lhsB[m, n, 0, j, i-1] = (1.0 if m == n else 0.0) 
	lhsC[m, n, 0, j, i-1] = 0.0

	lhsA[m, n, ksize, j, i-1] = 0.0
	lhsB[m, n, ksize, j, i-1] = (1.0 if m == n else 0.0)
	lhsC[m, n, ksize, j, i-1] = 0.0
#END z_solve_gpu_kernel_1()


# ---------------------------------------------------------------------
# performs line solves in Z direction by first factoring
# the block-tridiagonal matrix into an upper triangular matrix, 
# and then performing back substitution to solve for the unknow
# vectors of each line.  
#  
# make sure we treat elements zero to cell_size in the direction
# of the sweep.
# ---------------------------------------------------------------------
def z_solve_gpu(qs_device, 
				rho_i_device, 
				square_device, 
				u_device,
				rhs_device,
				lhsA_device, 
				lhsB_device, 
				lhsC_device,
				const_z_solve_device):
	amount_of_threads = [THREADS_PER_BLOCK_ON_Z_SOLVE_1, 1]
	amount_of_work    = [grid_points[0]-2, npbparams.PROBLEM_SIZE*25]
	
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     1)
	threadSize = (amount_of_threads[0], amount_of_threads[1], 1)
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Z_SOLVE_1)
	#print("1threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("1blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	z_solve_gpu_kernel_1[blockSize, 
						threadSize, 
						stream,
						size_shared_data_empty](lhsA_device, 
												lhsB_device, 
												lhsC_device,
												KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Z_SOLVE_1)

	amount_of_threads = [THREADS_PER_BLOCK_ON_Z_SOLVE_2, 1, 1]
	amount_of_work    = [grid_points[0]-2, npbparams.PROBLEM_SIZE, grid_points[2]-2]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Z_SOLVE_2)
	#print("2threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("2blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	z_solve_gpu_kernel_2[blockSize, 
						threadSize, 
						stream,
						size_shared_data_empty](qs_device, 
												square_device, 
												u_device, 
												lhsA_device,
												lhsB_device,
												lhsC_device,
												const_z_solve_device,
												KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Z_SOLVE_2)
		
	sharedMemPerBlock = device_prop.MAX_SHARED_MEMORY_PER_BLOCK
	max_amount_of_threads_i = min( int(THREADS_PER_BLOCK_ON_Z_SOLVE_3 / 5), int(sharedMemPerBlock / (rhs_device.dtype.itemsize*(3*5*5+2*5))) ) 
	
	amount_of_threads = [max_amount_of_threads_i, 5]
	amount_of_work    = [grid_points[0]-2, npbparams.PROBLEM_SIZE * 5]
	
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     1)
	threadSize = (amount_of_threads[0], amount_of_threads[1], 1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Z_SOLVE_3)
	#print("3threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("3blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	size_shared_data = rhs_device.dtype.itemsize * max_amount_of_threads_i * (3*5*5+2*5)
	#print("sharedMemory=%d" % (size_shared_data))
	z_solve_gpu_kernel_3[blockSize, 
						threadSize, 
						stream,
						size_shared_data](rhs_device, 
										lhsA_device, 
										lhsB_device, 
										lhsC_device,
										KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Z_SOLVE_3)
#END z_solve_gpu()


@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], int32, int32, int32, int32)')
def y_solve_gpu_kernel_3(rhs,
						lhsA, 
						lhsB, 
						lhsC,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	#extern __shared__ double tmp_l_lhs[];
	#double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];
	tmp_l_lhs = cuda.shared.array(shape=0, dtype=numba.float64)
	idx = cuda.blockDim.x * 3 * 5 * 5
	tmp_l_r = tmp_l_lhs[idx:]

	k = int((cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y) / 5)
	m = int((cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y) % 5)
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x + 1
	l_i = cuda.threadIdx.x
	if k+0 < 1 or k+0 > KMAX-2 or k >= PROBLEM_SIZE or i > IMAX-2:
		return

	#double (*tmp2_l_lhs)[3][5][5] = (double(*)[3][5][5])tmp_l_lhs;
	#double (*l_lhs)[5][5] = tmp2_l_lhs[l_i];
	D1, D2, D3 = 3, 5, 5
	l_lhs = tmp_l_lhs[(l_i*D1*D2*D3):] 
	
	#double (*tmp2_l_r)[2][5] = (double(*)[2][5])tmp_l_r;
	#double (*l_r)[5] = tmp2_l_r[l_i]; 
	M1, M2 = 2, 5
	l_r = tmp_l_r[(l_i*M1*M2):]
	
	jsize = JMAX - 1

	# ---------------------------------------------------------------------
	# performs guaussian elimination on this cell.
	# ---------------------------------------------------------------------
	# assumes that unpacking routines for non-first cells 
	# preload C' and rhs' from previous cell.
	# ---------------------------------------------------------------------
	# assumed send happens outside this routine, but that
	# c'(JMAX) and rhs'(JMAX) will be sent to next cell
	# ---------------------------------------------------------------------
	# load data
	for p in range(5):
		l_lhs[(BB*D2+p)*D3 +m] = lhsB[p, m, k, 0, i-1] #l_lhs[BB][p][m] to l_lhs[(BB*D2+p)*D3 +m]
		l_lhs[(CC*D2+p)*D3 +m] = lhsC[p, m, k, 0, i-1]

	l_r[1*M2+m] = rhs[k][0][i][m]

	cuda.syncthreads()

	# ---------------------------------------------------------------------
	# multiply c[k][0][i] by b_inverse and copy back to c
	# multiply rhs(0) by b_inverse(0) and copy to rhs
	# ---------------------------------------------------------------------
	for p in range(5):
		pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
		if m > p and m < 5:
			l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
		if m < 5:
			l_lhs[(CC*D2+m)*D3 +p] = l_lhs[(CC*D2+m)*D3 +p]*pivot
		if p == m:
			l_r[1*M2+p] = l_r[1*M2+p]*pivot

		cuda.syncthreads()

		if p != m:
			coeff = l_lhs[(BB*D2+p)*D3 +m]
			for n in range(p+1, 5):
				l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
			for n in range(5):
				l_lhs[(CC*D2+n)*D3 +m] = l_lhs[(CC*D2+n)*D3 +m] - coeff*l_lhs[(CC*D2+n)*D3 +p]
			l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]

		cuda.syncthreads()

	# update data
	rhs[k][0][i][m] = l_r[1*M2+m]

	# ---------------------------------------------------------------------
	# begin inner most do loop
	# do all the elements of the cell unless last 
	# ---------------------------------------------------------------------
	for j in range(1, jsize):
		# load data
		for n in range(5):
			l_lhs[(AA*D2+n)*D3 +m] = lhsA[n, m, k, j, i-1]
			l_lhs[(BB*D2+n)*D3 +m] = lhsB[n, m, k, j, i-1]

		l_r[0*M2+m] = l_r[1*M2+m]
		l_r[1*M2+m] = rhs[k][j][i][m]

		cuda.syncthreads()

		# ---------------------------------------------------------------------
		# subtract A*lhs_vector(j-1) from lhs_vector(j)
		# 
		# rhs(j) = rhs(j) - A*rhs(j-1)
		# ---------------------------------------------------------------------
		l_r[1*M2+m] = ( l_r[1*M2+m] - l_lhs[(AA*D2+0)*D3 +m]*l_r[0*M2+0]
			- l_lhs[(AA*D2+1)*D3 +m]*l_r[0*M2+1]
			- l_lhs[(AA*D2+2)*D3 +m]*l_r[0*M2+2]
			- l_lhs[(AA*D2+3)*D3 +m]*l_r[0*M2+3]
			- l_lhs[(AA*D2+4)*D3 +m]*l_r[0*M2+4] )

		# ---------------------------------------------------------------------
		# B(j) = B(j) - C(j-1)*A(j)
		# ---------------------------------------------------------------------
		for p in range(5):
			l_lhs[(BB*D2+m)*D3 +p] = ( l_lhs[(BB*D2+m)*D3 +p] - l_lhs[(AA*D2+0)*D3 +p]*l_lhs[(CC*D2+m)*D3 +0]
				- l_lhs[(AA*D2+1)*D3 +p]*l_lhs[(CC*D2+m)*D3 +1]
				- l_lhs[(AA*D2+2)*D3 +p]*l_lhs[(CC*D2+m)*D3 +2]
				- l_lhs[(AA*D2+3)*D3 +p]*l_lhs[(CC*D2+m)*D3 +3]
				- l_lhs[(AA*D2+4)*D3 +p]*l_lhs[(CC*D2+m)*D3 +4] )

		cuda.syncthreads()

		# update data
		for n in range(5):
			l_lhs[(CC*D2+n)*D3 +m] = lhsC[n, m, k, j, i-1]

		cuda.syncthreads()

		# ---------------------------------------------------------------------
		# multiply c[k][j][i] by b_inverse and copy back to c
		# multiply rhs[k][0][i] by b_inverse[k][0][i] and copy to rhs
		# ---------------------------------------------------------------------
		for p in range(5):
			pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
			if m > p:
				l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
			l_lhs[(CC*D2+m)*D3 +p] = l_lhs[(CC*D2+m)*D3 +p]*pivot
			if p == m:
				l_r[1*M2+p] = l_r[1*M2+p]*pivot
			
			cuda.syncthreads()
			if p != m:
				coeff = l_lhs[(BB*D2+p)*D3 +m]
				for n in range(p+1, 5):
					l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
				for n in range(5):
					l_lhs[(CC*D2+n)*D3 +m] = l_lhs[(CC*D2+n)*D3 +m] - coeff*l_lhs[(CC*D2+n)*D3 +p]
				l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]

			cuda.syncthreads()

		# update global memory
		for n in range(5):
			lhsC[n, m, k, j, i-1] = l_lhs[(CC*D2+n)*D3 +m]

		rhs[k][j][i][m] = l_r[1*M2+m]
	#END for j in range(1, jsize):
	
	j = j+1 #Increment j after loop
	
	# load data
	for n in range(5):
		l_lhs[(AA*D2+n)*D3 +m] = lhsA[n, m, k, j, i-1]
		l_lhs[(BB*D2+n)*D3 +m] = lhsB[n, m, k, j, i-1]

	l_r[0*M2+m] = l_r[1*M2+m]
	l_r[1*M2+m] = rhs[k][j][i][m]

	cuda.syncthreads()

	# ---------------------------------------------------------------------
	# rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
	# ---------------------------------------------------------------------
	l_r[1*M2+m] = ( l_r[1*M2+m] - l_lhs[(AA*D2+0)*D3 +m]*l_r[0*M2+0]
		- l_lhs[(AA*D2+1)*D3 +m]*l_r[0*M2+1]
		- l_lhs[(AA*D2+2)*D3 +m]*l_r[0*M2+2]
		- l_lhs[(AA*D2+3)*D3 +m]*l_r[0*M2+3]
		- l_lhs[(AA*D2+4)*D3 +m]*l_r[0*M2+4] )

	# ---------------------------------------------------------------------
	# B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
	# matmul_sub(AA,i,jsize,k,c, CC,i,jsize-1,k,c,BB,i,jsize,k)
	# ---------------------------------------------------------------------
	for p in range(5): 
		l_lhs[(BB*D2+m)*D3 +p] = ( l_lhs[(BB*D2+m)*D3 +p] - l_lhs[(AA*D2+0)*D3 +p]*l_lhs[(CC*D2+m)*D3 +0]
			- l_lhs[(AA*D2+1)*D3 +p]*l_lhs[(CC*D2+m)*D3 +1]
			- l_lhs[(AA*D2+2)*D3 +p]*l_lhs[(CC*D2+m)*D3 +2]
			- l_lhs[(AA*D2+3)*D3 +p]*l_lhs[(CC*D2+m)*D3 +3]
			- l_lhs[(AA*D2+4)*D3 +p]*l_lhs[(CC*D2+m)*D3 +4] )

	# ---------------------------------------------------------------------
	# multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
	# ---------------------------------------------------------------------
	# binvrhs_p( lhs[jsize][BB], rhs[k][jsize][i], run_computation, m);
	# ---------------------------------------------------------------------
	for p in range(5):
		pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
		if m > p and m < 5: 
			l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
		if p == m:
			l_r[1*M2+p] = l_r[1*M2+p]*pivot

		cuda.syncthreads()

		if p != m:
			coeff = l_lhs[(BB*D2+p)*D3 +m]
			for n in range(p+1, 5):
				l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
			l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]

		cuda.syncthreads()

	rhs[k][j][i][m] = l_r[1*M2+m]

	cuda.syncthreads()

	# ---------------------------------------------------------------------
	# back solve: if last cell, then generate U(jsize)=rhs(jsize)
	# else assume U(jsize) is loaded in un pack backsub_info
	# so just use it
	# after u(jstart) will be sent to next cell
	# ---------------------------------------------------------------------
	for j in range(jsize-1, -1, -1):
		for n in range(M_SIZE):
			rhs[k][j][i][m] = rhs[k][j][i][m] - lhsC[n, m, k, j, i-1]*rhs[k][j+1][i][n]
		cuda.syncthreads()
#END y_solve_gpu_kernel_3()


@cuda.jit('void(float64[:, :], float64[:], float64, float64, float64, float64, float64)', device=True)
def y_solve_gpu_device_fjac(fjac, 
							t_u,
							rho_i, 
							square, 
							qs,
							c1, c2):
	tmp1 = rho_i
	tmp2 = tmp1 * tmp1

	fjac[0][0] = 0.0
	fjac[1][0] = 0.0
	fjac[2][0] = 1.0
	fjac[3][0] = 0.0
	fjac[4][0] = 0.0

	fjac[0][1] = - ( t_u[1]*t_u[2] ) * tmp2
	fjac[1][1] = t_u[2] * tmp1
	fjac[2][1] = t_u[1] * tmp1
	fjac[3][1] = 0.0
	fjac[4][1] = 0.0

	fjac[0][2] = - ( t_u[2]*t_u[2]*tmp2) + c2 * qs
	fjac[1][2] = - c2 *  t_u[1] * tmp1
	fjac[2][2] = ( 2.0 - c2 ) *  t_u[2] * tmp1
	fjac[3][2] = - c2 * t_u[3] * tmp1
	fjac[4][2] = c2

	fjac[0][3] = - ( t_u[2]*t_u[3] ) * tmp2
	fjac[1][3] = 0.0
	fjac[2][3] = t_u[3] * tmp1
	fjac[3][3] = t_u[2] * tmp1
	fjac[4][3] = 0.0

	fjac[0][4] = ( c2 * 2.0 * square - c1 * t_u[4] ) * t_u[2] * tmp2
	fjac[1][4] = - c2 * t_u[1]*t_u[2] * tmp2
	fjac[2][4] = c1 * t_u[4] * tmp1 - c2 * ( qs + t_u[2]*t_u[2] * tmp2 )
	fjac[3][4] = - c2 * ( t_u[2]*t_u[3] ) * tmp2
	fjac[4][4] = c1 * t_u[2] * tmp1
#END y_solve_gpu_device_fjac()


@cuda.jit('void(float64[:, :], float64[:], float64, float64, float64, float64)', device=True)
def y_solve_gpu_device_njac(njac, 
							t_u,
							rho_i,
							c3c4, con43, c1345):
	tmp1 = rho_i
	tmp2 = tmp1 * tmp1
	tmp3 = tmp1 * tmp2

	njac[0][0] = 0.0
	njac[1][0] = 0.0
	njac[2][0] = 0.0
	njac[3][0] = 0.0
	njac[4][0] = 0.0

	njac[0][1] = - c3c4 * tmp2 * t_u[1]
	njac[1][1] = c3c4 * tmp1
	njac[2][1] = 0.0
	njac[3][1] = 0.0
	njac[4][1] = 0.0

	njac[0][2] = - con43 * c3c4 * tmp2 * t_u[2]
	njac[1][2] = 0.0
	njac[2][2] = con43 * c3c4 * tmp1
	njac[3][2] = 0.0
	njac[4][2] = 0.0

	njac[0][3] = - c3c4 * tmp2 * t_u[3]
	njac[1][3] = 0.0
	njac[2][3] = 0.0
	njac[3][3] = c3c4 * tmp1
	njac[4][3] = 0.0

	njac[0][4] = ( - (  c3c4
			- c1345 ) * tmp3 * (t_u[1]*t_u[1])
		- ( con43 * c3c4
				- c1345 ) * tmp3 * (t_u[2]*t_u[2])
		- ( c3c4 - c1345 ) * tmp3 * (t_u[3]*t_u[3])
		- c1345 * tmp2 * t_u[4] )

	njac[1][4] = ( c3c4 - c1345 ) * tmp2 * t_u[1]
	njac[2][4] = ( con43 * c3c4 - c1345 ) * tmp2 * t_u[2]
	njac[3][4] = ( c3c4 - c1345 ) * tmp2 * t_u[3]
	njac[4][4] = ( c1345 ) * tmp1
#END y_solve_gpu_device_njac()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:], int32, int32, int32, int32)')
def y_solve_gpu_kernel_2(qs, 
						rho_i,
						square, 
						u, 
						lhsA,
						lhsB,
						lhsC,
						const_arr,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	k = cuda.blockDim.z * cuda.blockIdx.z + cuda.threadIdx.z
	j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y + 1
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x + 1
	if k+0 < 1 or k+0 > KMAX-2 or k >= PROBLEM_SIZE  or j > JMAX-2 or i > IMAX-2:
		return

	dt, ty1, ty2 = const_arr[0], const_arr[1], const_arr[2]
	dy1, dy2, dy3, dy4, dy5 = const_arr[3], const_arr[4], const_arr[5], const_arr[6], const_arr[7]
	c1, c2, con43, c3c4, c1345 = const_arr[8], const_arr[9], const_arr[10], const_arr[11], const_arr[12]
	
	fjac = cuda.local.array((5, 5), numba.float64)
	njac = cuda.local.array((5, 5), numba.float64)
	t_u = cuda.local.array(5, numba.float64)

	# ---------------------------------------------------------------------
	# this function computes the left hand side for the three y-factors   
	# ---------------------------------------------------------------------
	# compute the indices for storing the tri-diagonal matrix;
	# determine a (labeled f) and n jacobians for cell c
	# ---------------------------------------------------------------------
	tmp1 = dt * ty1
	tmp2 = dt * ty2

	for m in range(5):
		t_u[m] = u[k][j-1][i][m]
	y_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j-1][i], square[k][j-1][i], qs[k][j-1][i], c1, c2)
	y_solve_gpu_device_njac(njac, t_u, rho_i[k][j-1][i], c3c4, con43, c1345)

	lhsA[0, 0, k, j, i-1] = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dy1 
	lhsA[1, 0, k, j, i-1] = - tmp2 * fjac[1][0] - tmp1 * njac[1][0]
	lhsA[2, 0, k, j, i-1] = - tmp2 * fjac[2][0] - tmp1 * njac[2][0]
	lhsA[3, 0, k, j, i-1] = - tmp2 * fjac[3][0] - tmp1 * njac[3][0]
	lhsA[4, 0, k, j, i-1] = - tmp2 * fjac[4][0] - tmp1 * njac[4][0]

	lhsA[0, 1, k, j, i-1] = - tmp2 * fjac[0][1] - tmp1 * njac[0][1]
	lhsA[1, 1, k, j, i-1] = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dy2
	lhsA[2, 1, k, j, i-1] = - tmp2 * fjac[2][1] - tmp1 * njac[2][1]
	lhsA[3, 1, k, j, i-1] = - tmp2 * fjac[3][1] - tmp1 * njac[3][1]
	lhsA[4, 1, k, j, i-1] = - tmp2 * fjac[4][1] - tmp1 * njac[4][1]

	lhsA[0, 2, k, j, i-1] = - tmp2 * fjac[0][2] - tmp1 * njac[0][2]
	lhsA[1, 2, k, j, i-1] = - tmp2 * fjac[1][2] - tmp1 * njac[1][2]
	lhsA[2, 2, k, j, i-1] = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dy3
	lhsA[3, 2, k, j, i-1] = - tmp2 * fjac[3][2] - tmp1 * njac[3][2]
	lhsA[4, 2, k, j, i-1] = - tmp2 * fjac[4][2] - tmp1 * njac[4][2]

	lhsA[0, 3, k, j, i-1] = - tmp2 * fjac[0][3] - tmp1 * njac[0][3]
	lhsA[1, 3, k, j, i-1] = - tmp2 * fjac[1][3] - tmp1 * njac[1][3]
	lhsA[2, 3, k, j, i-1] = - tmp2 * fjac[2][3] - tmp1 * njac[2][3]
	lhsA[3, 3, k, j, i-1] = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dy4
	lhsA[4, 3, k, j, i-1] = - tmp2 * fjac[4][3] - tmp1 * njac[4][3]

	lhsA[0, 4, k, j, i-1] = - tmp2 * fjac[0][4] - tmp1 * njac[0][4]
	lhsA[1, 4, k, j, i-1] = - tmp2 * fjac[1][4] - tmp1 * njac[1][4]
	lhsA[2, 4, k, j, i-1] = - tmp2 * fjac[2][4] - tmp1 * njac[2][4]
	lhsA[3, 4, k, j, i-1] = - tmp2 * fjac[3][4] - tmp1 * njac[3][4]
	lhsA[4, 4, k, j, i-1] = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dy5
	
	for m in range(5):
		t_u[m] = u[k][j][i][m]
	y_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i], c3c4, con43, c1345)
    
	lhsB[0, 0, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[0][0] + tmp1 * 2.0 * dy1
	lhsB[1, 0, k, j, i-1] = tmp1 * 2.0 * njac[1][0]
	lhsB[2, 0, k, j, i-1] = tmp1 * 2.0 * njac[2][0]
	lhsB[3, 0, k, j, i-1] = tmp1 * 2.0 * njac[3][0]
	lhsB[4, 0, k, j, i-1] = tmp1 * 2.0 * njac[4][0]
    
	lhsB[0, 1, k, j, i-1] = tmp1 * 2.0 * njac[0][1]
	lhsB[1, 1, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[1][1] + tmp1 * 2.0 * dy2
	lhsB[2, 1, k, j, i-1] = tmp1 * 2.0 * njac[2][1]
	lhsB[3, 1, k, j, i-1] = tmp1 * 2.0 * njac[3][1]
	lhsB[4, 1, k, j, i-1] = tmp1 * 2.0 * njac[4][1]
    
	lhsB[0, 2, k, j, i-1] = tmp1 * 2.0 * njac[0][2]
	lhsB[1, 2, k, j, i-1] = tmp1 * 2.0 * njac[1][2]
	lhsB[2, 2, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[2][2] + tmp1 * 2.0 * dy3
	lhsB[3, 2, k, j, i-1] = tmp1 * 2.0 * njac[3][2]
	lhsB[4, 2, k, j, i-1] = tmp1 * 2.0 * njac[4][2]
    
	lhsB[0, 3, k, j, i-1] = tmp1 * 2.0 * njac[0][3]
	lhsB[1, 3, k, j, i-1] = tmp1 * 2.0 * njac[1][3]
	lhsB[2, 3, k, j, i-1] = tmp1 * 2.0 * njac[2][3]
	lhsB[3, 3, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[3][3] + tmp1 * 2.0 * dy4
	lhsB[4, 3, k, j, i-1] = tmp1 * 2.0 * njac[4][3]
    
	lhsB[0, 4, k, j, i-1] = tmp1 * 2.0 * njac[0][4]
	lhsB[1, 4, k, j, i-1] = tmp1 * 2.0 * njac[1][4]
	lhsB[2, 4, k, j, i-1] = tmp1 * 2.0 * njac[2][4]
	lhsB[3, 4, k, j, i-1] = tmp1 * 2.0 * njac[3][4]
	lhsB[4, 4, k, j, i-1] = 1.0 + tmp1 * 2.0 * njac[4][4] + tmp1 * 2.0 * dy5
	
	for m in range(5):
		t_u[m] = u[k][j+1][i][m]
	y_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j+1][i], square[k][j+1][i], qs[k][j+1][i], c1, c2)
	y_solve_gpu_device_njac(njac, t_u, rho_i[k][j+1][i], c3c4, con43, c1345)
    
	lhsC[0, 0, k, j, i-1] =  tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dy1
	lhsC[1, 0, k, j, i-1] =  tmp2 * fjac[1][0] - tmp1 * njac[1][0]
	lhsC[2, 0, k, j, i-1] =  tmp2 * fjac[2][0] - tmp1 * njac[2][0]
	lhsC[3, 0, k, j, i-1] =  tmp2 * fjac[3][0] - tmp1 * njac[3][0]
	lhsC[4, 0, k, j, i-1] =  tmp2 * fjac[4][0] - tmp1 * njac[4][0]
    
	lhsC[0, 1, k, j, i-1] =  tmp2 * fjac[0][1] - tmp1 * njac[0][1]
	lhsC[1, 1, k, j, i-1] =  tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dy2
	lhsC[2, 1, k, j, i-1] =  tmp2 * fjac[2][1] - tmp1 * njac[2][1]
	lhsC[3, 1, k, j, i-1] =  tmp2 * fjac[3][1] - tmp1 * njac[3][1]
	lhsC[4, 1, k, j, i-1] =  tmp2 * fjac[4][1] - tmp1 * njac[4][1]
    
	lhsC[0, 2, k, j, i-1] =  tmp2 * fjac[0][2] - tmp1 * njac[0][2]
	lhsC[1, 2, k, j, i-1] =  tmp2 * fjac[1][2] - tmp1 * njac[1][2]
	lhsC[2, 2, k, j, i-1] =  tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dy3
	lhsC[3, 2, k, j, i-1] =  tmp2 * fjac[3][2] - tmp1 * njac[3][2]
	lhsC[4, 2, k, j, i-1] =  tmp2 * fjac[4][2] - tmp1 * njac[4][2]
    
	lhsC[0, 3, k, j, i-1] =  tmp2 * fjac[0][3] - tmp1 * njac[0][3]
	lhsC[1, 3, k, j, i-1] =  tmp2 * fjac[1][3] - tmp1 * njac[1][3]
	lhsC[2, 3, k, j, i-1] =  tmp2 * fjac[2][3] - tmp1 * njac[2][3]
	lhsC[3, 3, k, j, i-1] =  tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dy4
	lhsC[4, 3, k, j, i-1] =  tmp2 * fjac[4][3] - tmp1 * njac[4][3]
    
	lhsC[0, 4, k, j, i-1] =  tmp2 * fjac[0][4] - tmp1 * njac[0][4]
	lhsC[1, 4, k, j, i-1] =  tmp2 * fjac[1][4] - tmp1 * njac[1][4]
	lhsC[2, 4, k, j, i-1] =  tmp2 * fjac[2][4] - tmp1 * njac[2][4]
	lhsC[3, 4, k, j, i-1] =  tmp2 * fjac[3][4] - tmp1 * njac[3][4]
	lhsC[4, 4, k, j, i-1] =  tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dy5
#END y_solve_gpu_kernel_2()


@cuda.jit('void(float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], int32, int32, int32, int32)')
def y_solve_gpu_kernel_1(lhsA,
						lhsB, 
						lhsC,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	t_k = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
	k = int(t_k % PROBLEM_SIZE)
	mn = int(t_k / PROBLEM_SIZE)
	m = int(mn / 5)
	n = int(mn % 5)
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x+1

	if k+0 < 1 or k+0 > KMAX-2 or k >= PROBLEM_SIZE or i > IMAX-2 or m >= 5:
		return

	jsize = JMAX - 1

	# ---------------------------------------------------------------------
	# now joacobians set, so form left hand side in y direction
	# ---------------------------------------------------------------------
	lhsA[m, n, k, 0, i-1] = 0.0
	lhsB[m, n, k, 0, i-1] = (1.0 if m == n else 0.0)
	lhsC[m, n, k, 0, i-1] = 0.0

	lhsA[m, n, k, jsize, i-1] = 0.0
	lhsB[m, n, k, jsize, i-1] = (1.0 if m == n else 0.0)
	lhsC[m, n, k, jsize, i-1] = 0.0
#END y_solve_gpu_kernel_1()


# ---------------------------------------------------------------------
# performs line solves in y direction by first factoring
# the block-tridiagonal matrix into an upper triangular matrix, 
# and then performing back substitution to solve for the unknow
# vectors of each line.  
#  
# make sure we treat elements zero to cell_size in the direction
# of the sweep.
# ---------------------------------------------------------------------
def y_solve_gpu(qs_device, 
				rho_i_device, 
				square_device, 
				u_device,
				rhs_device,
				lhsA_device, 
				lhsB_device, 
				lhsC_device,
				const_y_solve_device):
	# ---------------------------------------------------------------------
	# this function computes the left hand side for the three y-factors   
	# ---------------------------------------------------------------------
	# compute the indices for storing the tri-diagonal matrix;
	# determine a (labeled f) and n jacobians for cell c
	# ---------------------------------------------------------------------
	amount_of_threads = [THREADS_PER_BLOCK_ON_Y_SOLVE_1, 1]
	amount_of_work    = [grid_points[0]-2, npbparams.PROBLEM_SIZE*25]
	
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     1)
	threadSize = (amount_of_threads[0], amount_of_threads[1], 1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Y_SOLVE_1)
	#print("1threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("1blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	y_solve_gpu_kernel_1[blockSize, 
						threadSize, 
						stream,
						size_shared_data_empty](lhsA_device, 
												lhsB_device, 
												lhsC_device,
												KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Y_SOLVE_1)
		
	amount_of_work    = [grid_points[0]-2, grid_points[1]-2, npbparams.PROBLEM_SIZE]
	amount_of_threads = [THREADS_PER_BLOCK_ON_Y_SOLVE_2, 1, 1]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Y_SOLVE_2)
	#print("2threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("2blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	y_solve_gpu_kernel_2[blockSize, 
						threadSize, 
						stream,
						size_shared_data_empty](qs_device, 
												rho_i_device, 
												square_device, 
												u_device,
												lhsA_device,
												lhsB_device,
												lhsC_device,
												const_y_solve_device,
												KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Y_SOLVE_2)

	sharedMemPerBlock = device_prop.MAX_SHARED_MEMORY_PER_BLOCK
	max_amount_of_threads_i = min( int(THREADS_PER_BLOCK_ON_Y_SOLVE_3 / 5), int(sharedMemPerBlock / (rhs_device.dtype.itemsize*(3*5*5+2*5))) ) 
	max_amount_of_threads_i = int(max_amount_of_threads_i / 2)
	
	amount_of_threads = [max_amount_of_threads_i, 5]
	amount_of_work    = [grid_points[0]-2, npbparams.PROBLEM_SIZE * 5]
	
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     1)
	threadSize = (amount_of_threads[0], amount_of_threads[1], 1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_Y_SOLVE_3)
	#print("3threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("3blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	size_shared_data = rhs_device.dtype.itemsize * max_amount_of_threads_i * (3*5*5+2*5)
	#print("sharedMemory=%d" % (size_shared_data))
	y_solve_gpu_kernel_3[blockSize, 
						threadSize, 
						stream,
						size_shared_data](rhs_device, 
										lhsA_device, 
										lhsB_device, 
										lhsC_device,
										KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_Y_SOLVE_3)
#END y_solve_gpu()


# ---------------------------------------------------------------------
# this function computes the left hand side in the xi-direction
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], int32, int32, int32, int32)')
def x_solve_gpu_kernel_3(rhs,
						lhsA, 
						lhsB, 
						lhsC,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	#extern __shared__ double tmp_l_lhs[];
	#double *tmp_l_r = &tmp_l_lhs[blockDim.x * 3 * 5 * 5];
	tmp_l_lhs = cuda.shared.array(shape=0, dtype=numba.float64)
	idx = cuda.blockDim.x * 3 * 5 * 5
	tmp_l_r = tmp_l_lhs[idx:]
	
	k = int((cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y) / 5)
	m = int((cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y) % 5)
	j = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x+1
	l_j = cuda.threadIdx.x
	if k+0 < 1 or k+0 > KMAX-2 or k >= PROBLEM_SIZE or j > JMAX-2:
		return
	
	#double (*tmp2_l_lhs)[3][5][5] = (double(*)[3][5][5])tmp_l_lhs;
	#double (*l_lhs)[5][5] = tmp2_l_lhs[l_j];
	D1, D2, D3 = 3, 5, 5
	l_lhs = tmp_l_lhs[(l_j*D1*D2*D3):] 
	
	#double (*tmp2_l_r)[2][5] = (double(*)[2][5])tmp_l_r;
	#double (*l_r)[5] = tmp2_l_r[l_j]; 
	M1, M2 = 2, 5
	l_r = tmp_l_r[(l_j*M1*M2):]
	
	isize = IMAX - 1

	# ---------------------------------------------------------------------
	# performs guaussian elimination on this cell.
	# ---------------------------------------------------------------------
	# assumes that unpacking routines for non-first cells 
	# preload C' and rhs' from previous cell.
	# ---------------------------------------------------------------------
	# assumed send happens outside this routine, but that
	# c'(IMAX) and rhs'(IMAX) will be sent to next cell
	# ---------------------------------------------------------------------
	# outer most do loops - sweeping in i direction
	# ---------------------------------------------------------------------
	
	# load data
	for p in range(5):
		l_lhs[(BB*D2+p)*D3 +m] = lhsB[p, m, k, 0, j-1]
		l_lhs[(CC*D2+p)*D3 +m] = lhsC[p, m, k, 0, j-1]

	l_r[1*M2+m] = rhs[k][j][0][m]

	cuda.syncthreads()
	
	# ---------------------------------------------------------------------
	# multiply c[k][j][0] by b_inverse and copy back to c
	# multiply rhs(0) by b_inverse(0) and copy to rhs
	# ---------------------------------------------------------------------
	for p in range(5):
		pivot = 1.00 / l_lhs[(BB*D2+p)*D3 +p]
		if m>p and m<5:
			l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
		if m < 5:
			l_lhs[(CC*D2+m)*D3 +p] = l_lhs[(CC*D2+m)*D3 +p]*pivot
		if p == m:
			l_r[1*M2+p] = l_r[1*M2+p]*pivot
		cuda.syncthreads()
    
		if p != m:
			coeff = l_lhs[(BB*D2+p)*D3 +m]
			for n in range(p+1, 5):
				l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
			for n in range(5):
				l_lhs[(CC*D2+n)*D3 +m] = l_lhs[(CC*D2+n)*D3 +m] - coeff*l_lhs[(CC*D2+n)*D3 +p]
			l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]
		cuda.syncthreads()

	# update data
	rhs[k][j][0][m] = l_r[1*M2+m]
	
	# ---------------------------------------------------------------------
	# begin inner most do loop
	# do all the elements of the cell unless last 
	# ---------------------------------------------------------------------
	for i in range(1, isize):
		for n in range(5):
			l_lhs[(AA*D2+n)*D3 +m] = lhsA[n, m, k, i, j-1]
			l_lhs[(BB*D2+n)*D3 +m] = lhsB[n, m, k, i, j-1]
		
		l_r[0*M2+m] = l_r[1*M2+m]
		l_r[1*M2+m] = rhs[k][j][i][m]
    
		cuda.syncthreads()
		
		# ---------------------------------------------------------------------
		# rhs(i) = rhs(i) - A*rhs(i-1)
		# ---------------------------------------------------------------------
		l_r[1*M2+m] = ( l_r[1*M2+m] - l_lhs[(AA*D2+0)*D3 +m]*l_r[0*M2+0]
			- l_lhs[(AA*D2+1)*D3 +m]*l_r[0*M2+1]
			- l_lhs[(AA*D2+2)*D3 +m]*l_r[0*M2+2]
			- l_lhs[(AA*D2+3)*D3 +m]*l_r[0*M2+3]
			- l_lhs[(AA*D2+4)*D3 +m]*l_r[0*M2+4] )
    
		# ---------------------------------------------------------------------
		# B(i) = B(i) - C(i-1)*A(i)
		# ---------------------------------------------------------------------
		for p in range(5):
			l_lhs[(BB*D2+m)*D3 +p] = ( l_lhs[(BB*D2+m)*D3 +p] - l_lhs[(AA*D2+0)*D3 +p]*l_lhs[(CC*D2+m)*D3 +0]
				- l_lhs[(AA*D2+1)*D3 +p]*l_lhs[(CC*D2+m)*D3 +1]
				- l_lhs[(AA*D2+2)*D3 +p]*l_lhs[(CC*D2+m)*D3 +2]
				- l_lhs[(AA*D2+3)*D3 +p]*l_lhs[(CC*D2+m)*D3 +3]
				- l_lhs[(AA*D2+4)*D3 +p]*l_lhs[(CC*D2+m)*D3 +4])
    
		cuda.syncthreads()
		
		for n in range(5): 
			l_lhs[(CC*D2+n)*D3 +m] = lhsC[n, m, k, i, j-1]
    
		cuda.syncthreads()
    
		# ---------------------------------------------------------------------
		# multiply c[k][j][i] by b_inverse and copy back to c
		# multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
		# ---------------------------------------------------------------------
		for p in range(5):
			pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
			if m > p:
				l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
			l_lhs[(CC*D2+m)*D3 +p] = l_lhs[(CC*D2+m)*D3 +p]*pivot
			if p == m:
				l_r[1*M2+p] = l_r[1*M2+p]*pivot
    
			cuda.syncthreads()
    
			if p != m:
				coeff = l_lhs[(BB*D2+p)*D3 +m]
				for n in range(p+1, 5):
					l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
				for n in range(5):
					l_lhs[(CC*D2+n)*D3 +m] = l_lhs[(CC*D2+n)*D3 +m] - coeff*l_lhs[(CC*D2+n)*D3 +p]
				l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]
    
			cuda.syncthreads()
    
		for n in range(5):
			lhsC[n, m, k, i, j-1] = l_lhs[(CC*D2+n)*D3 +m]
    
		rhs[k][j][i][m] = l_r[1*M2+m]
	#END for i in range(1, isize):
    
	i = i+1 #Increment i after loop
	for n in range(5):
		l_lhs[(AA*D2+n)*D3 +m] = lhsA[n, m, k, i, j-1]
		l_lhs[(BB*D2+n)*D3 +m] = lhsB[n, m, k, i, j-1]
    
	l_r[0*M2+m] = l_r[1*M2+m]
	l_r[1*M2+m] = rhs[k][j][i][m]
    
	cuda.syncthreads()
	
	# ---------------------------------------------------------------------
	# rhs(isize) = rhs(isize) - A*rhs(isize-1)
	# ---------------------------------------------------------------------
	l_r[1*M2+m] = ( l_r[1*M2+m] - l_lhs[(AA*D2+0)*D3 +m]*l_r[0*M2+0]
		- l_lhs[(AA*D2+1)*D3 +m]*l_r[0*M2+1]
		- l_lhs[(AA*D2+2)*D3 +m]*l_r[0*M2+2]
		- l_lhs[(AA*D2+3)*D3 +m]*l_r[0*M2+3]
		- l_lhs[(AA*D2+4)*D3 +m]*l_r[0*M2+4] )
    
	# ---------------------------------------------------------------------
	# B(isize) = B(isize) - C(isize-1)*A(isize)
	# ---------------------------------------------------------------------
	for p in range(5):
		l_lhs[(BB*D2+m)*D3 +p] = ( l_lhs[(BB*D2+m)*D3 +p] - l_lhs[(AA*D2+0)*D3 +p]*l_lhs[(CC*D2+m)*D3 +0]
			- l_lhs[(AA*D2+1)*D3 +p]*l_lhs[(CC*D2+m)*D3 +1]
			- l_lhs[(AA*D2+2)*D3 +p]*l_lhs[(CC*D2+m)*D3 +2]
			- l_lhs[(AA*D2+3)*D3 +p]*l_lhs[(CC*D2+m)*D3 +3]
			- l_lhs[(AA*D2+4)*D3 +p]*l_lhs[(CC*D2+m)*D3 +4] )
    
	# ---------------------------------------------------------------------
	# multiply rhs() by b_inverse() and copy to rhs
	# ---------------------------------------------------------------------
	for p in range(5):
		pivot = 1.00/l_lhs[(BB*D2+p)*D3 +p]
		if m > p and m < 5:
			l_lhs[(BB*D2+m)*D3 +p] = l_lhs[(BB*D2+m)*D3 +p]*pivot
		if p == m:
			l_r[1*M2+p] = l_r[1*M2+p]*pivot
    
		cuda.syncthreads()
    
		if p != m:
			coeff = l_lhs[(BB*D2+p)*D3 +m]
			for n in range(p+1, 5):
				l_lhs[(BB*D2+n)*D3 +m] = l_lhs[(BB*D2+n)*D3 +m] - coeff*l_lhs[(BB*D2+n)*D3 +p]
			l_r[1*M2+m] = l_r[1*M2+m] - coeff*l_r[1*M2+p]
        
		cuda.syncthreads()
    
	rhs[k][j][i][m] = l_r[1*M2+m]
    
	cuda.syncthreads()
    
	# ---------------------------------------------------------------------
	# back solve: if last cell, then generate U(isize)=rhs(isize)
	# else assume U(isize) is loaded in un pack backsub_info
	# so just use it
	# after u(istart) will be sent to next cell
	# ---------------------------------------------------------------------
	for i in range(isize-1, -1, -1):
		for n in range(M_SIZE):
			rhs[k][j][i][m] = rhs[k][j][i][m] - lhsC[n, m, k, i, j-1] * rhs[k][j][i+1][n]
		cuda.syncthreads()
#END x_solve_gpu_kernel_3()


# ---------------------------------------------------------------------
# this function computes the left hand side in the xi-direction
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :], float64[:], float64, float64, float64, float64, float64)', device=True)
def x_solve_gpu_device_fjac(fjac, 
							t_u, 
							rho_i,
							qs, 
							square,
							c1, c2):
	# ---------------------------------------------------------------------
	# determine a (labeled f) and n jacobians
	# ---------------------------------------------------------------------
	tmp1 = rho_i
	tmp2 = tmp1 * tmp1
	
	fjac[0][0] = 0.0
	fjac[1][0] = 1.0
	fjac[2][0] = 0.0
	fjac[3][0] = 0.0
	fjac[4][0] = 0.0
	
	fjac[0][1] = ( -(t_u[1] * tmp2 * t_u[1])
		+ c2 * qs )
	fjac[1][1] = ( 2.0 - c2 ) * ( t_u[1] / t_u[0] )
	fjac[2][1] = - c2 * ( t_u[2] * tmp1 )
	fjac[3][1] = - c2 * ( t_u[3] * tmp1 )
	fjac[4][1] = c2

	fjac[0][2] = - ( t_u[1]*t_u[2] ) * tmp2
	fjac[1][2] = t_u[2] * tmp1
	fjac[2][2] = t_u[1] * tmp1
	fjac[3][2] = 0.0
	fjac[4][2] = 0.0

	fjac[0][3] = - ( t_u[1]*t_u[3] ) * tmp2
	fjac[1][3] = t_u[3] * tmp1
	fjac[2][3] = 0.0
	fjac[3][3] = t_u[1] * tmp1
	fjac[4][3] = 0.0

	fjac[0][4] = ( ( c2 * 2.0 * square - c1 * t_u[4] )
		* ( t_u[1] * tmp2 ) )
	fjac[1][4] = ( c1 *  t_u[4] * tmp1 
		- c2 * ( t_u[1]*t_u[1] * tmp2 + qs ) )
	fjac[2][4] = - c2 * ( t_u[2]*t_u[1] ) * tmp2
	fjac[3][4] = - c2 * ( t_u[3]*t_u[1] ) * tmp2
	fjac[4][4] = c1 * ( t_u[1] * tmp1 )
#END x_solve_gpu_device_fjac()


@cuda.jit('void(float64[:, :], float64[:], float64, float64, float64, float64)', device=True)
def x_solve_gpu_device_njac(njac, 
							t_u,
							rho_i,
							con43, c3c4, c1345):
	# ---------------------------------------------------------------------
	# determine a (labeled f) and n jacobians
	# ---------------------------------------------------------------------
	tmp1 = rho_i
	tmp2 = tmp1 * tmp1
	tmp3 = tmp1 * tmp2

	njac[0][0] = 0.0
	njac[1][0] = 0.0
	njac[2][0] = 0.0
	njac[3][0] = 0.0
	njac[4][0] = 0.0

	njac[0][1] = - con43 * c3c4 * tmp2 * t_u[1]
	njac[1][1] = con43 * c3c4 * tmp1
	njac[2][1] = 0.0
	njac[3][1] = 0.0
	njac[4][1] = 0.0

	njac[0][2] = - c3c4 * tmp2 * t_u[2]
	njac[1][2] = 0.0
	njac[2][2] = c3c4 * tmp1
	njac[3][2] = 0.0
	njac[4][2] = 0.0

	njac[0][3] = - c3c4 * tmp2 * t_u[3]
	njac[1][3] = 0.0
	njac[2][3] = 0.0
	njac[3][3] = c3c4 * tmp1
	njac[4][3] = 0.0

	njac[0][4] = ( - ( con43 * c3c4
			- c1345 ) * tmp3 * (t_u[1]*t_u[1])
		- ( c3c4 - c1345 ) * tmp3 * (t_u[2]*t_u[2])
		- ( c3c4 - c1345 ) * tmp3 * (t_u[3]*t_u[3])
		- c1345 * tmp2 * t_u[4] )

	njac[1][4] = ( ( con43 * c3c4
			- c1345 ) * tmp2 * t_u[1] )
	njac[2][4] = ( c3c4 - c1345 ) * tmp2 * t_u[2]
	njac[3][4] = ( c3c4 - c1345 ) * tmp2 * t_u[3]
	njac[4][4] = ( c1345 ) * tmp1
#END x_solve_gpu_device_njac()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:], int32, int32, int32, int32)')
def x_solve_gpu_kernel_2(qs, 
						rho_i,
						square, 
						u, 
						lhsA,
						lhsB, 
						lhsC, 
						const_arr,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	k = cuda.blockDim.z * cuda.blockIdx.z + cuda.threadIdx.z
	j = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y+1
	i = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x+1

	if k+0 < 1 or k+0 > KMAX-2 or k >= PROBLEM_SIZE or j > JMAX-2 or i > IMAX-2:
		return
	
	dt, tx1, tx2 = const_arr[0], const_arr[1], const_arr[2]
	dx1, dx2, dx3, dx4, dx5 = const_arr[3], const_arr[4], const_arr[5], const_arr[6], const_arr[7]
	c1, c2, con43, c3c4, c1345 = const_arr[8], const_arr[9], const_arr[10], const_arr[11], const_arr[12]
	
	fjac = cuda.local.array((5, 5), numba.float64)
	njac = cuda.local.array((5, 5), numba.float64)
	t_u = cuda.local.array(5, numba.float64)

	tmp1 = dt * tx1
	tmp2 = dt * tx2

	for m in range(5):
		t_u[m] = u[k][j][i-1][m]
	
	x_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j][i-1], qs[k][j][i-1], square[k][j][i-1], c1, c2)
	x_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i-1], con43, c3c4, c1345)

	lhsA[0, 0, k, i, j-1] = - tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dx1
	lhsA[1, 0, k, i, j-1] = - tmp2 * fjac[1][0] - tmp1 * njac[1][0]
	lhsA[2, 0, k, i, j-1] = - tmp2 * fjac[2][0] - tmp1 * njac[2][0]
	lhsA[3, 0, k, i, j-1] = - tmp2 * fjac[3][0] - tmp1 * njac[3][0]
	lhsA[4, 0, k, i, j-1] = - tmp2 * fjac[4][0] - tmp1 * njac[4][0]
    
	lhsA[0, 1, k, i, j-1] = - tmp2 * fjac[0][1] - tmp1 * njac[0][1]
	lhsA[1, 1, k, i, j-1] = - tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dx2
	lhsA[2, 1, k, i, j-1] = - tmp2 * fjac[2][1] - tmp1 * njac[2][1]
	lhsA[3, 1, k, i, j-1] = - tmp2 * fjac[3][1] - tmp1 * njac[3][1]
	lhsA[4, 1, k, i, j-1] = - tmp2 * fjac[4][1] - tmp1 * njac[4][1]
    
	lhsA[0, 2, k, i, j-1] = - tmp2 * fjac[0][2] - tmp1 * njac[0][2]
	lhsA[1, 2, k, i, j-1] = - tmp2 * fjac[1][2] - tmp1 * njac[1][2]
	lhsA[2, 2, k, i, j-1] = - tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dx3
	lhsA[3, 2, k, i, j-1] = - tmp2 * fjac[3][2] - tmp1 * njac[3][2]
	lhsA[4, 2, k, i, j-1] = - tmp2 * fjac[4][2] - tmp1 * njac[4][2]
    
	lhsA[0, 3, k, i, j-1] = - tmp2 * fjac[0][3] - tmp1 * njac[0][3]
	lhsA[1, 3, k, i, j-1] = - tmp2 * fjac[1][3] - tmp1 * njac[1][3]
	lhsA[2, 3, k, i, j-1] = - tmp2 * fjac[2][3] - tmp1 * njac[2][3]
	lhsA[3, 3, k, i, j-1] = - tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dx4
	lhsA[4, 3, k, i, j-1] = - tmp2 * fjac[4][3] - tmp1 * njac[4][3]
    
	lhsA[0, 4, k, i, j-1] = - tmp2 * fjac[0][4] - tmp1 * njac[0][4]
	lhsA[1, 4, k, i, j-1] = - tmp2 * fjac[1][4] - tmp1 * njac[1][4]
	lhsA[2, 4, k, i, j-1] = - tmp2 * fjac[2][4] - tmp1 * njac[2][4]
	lhsA[3, 4, k, i, j-1] = - tmp2 * fjac[3][4] - tmp1 * njac[3][4]
	lhsA[4, 4, k, i, j-1] = - tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dx5
	
	for m in range(5):
		t_u[m] = u[k][j][i][m]
	x_solve_gpu_device_njac(fjac, t_u, rho_i[k][j][i], con43, c3c4, c1345)
    
	lhsB[0, 0, k, i, j-1] = 1.0 + tmp1 * 2.0 * fjac[0][0] + tmp1 * 2.0 * dx1
	lhsB[1, 0, k, i, j-1] = tmp1 * 2.0 * fjac[1][0]
	lhsB[2, 0, k, i, j-1] = tmp1 * 2.0 * fjac[2][0]
	lhsB[3, 0, k, i, j-1] = tmp1 * 2.0 * fjac[3][0]
	lhsB[4, 0, k, i, j-1] = tmp1 * 2.0 * fjac[4][0]
    
	lhsB[0, 1, k, i, j-1] = tmp1 * 2.0 * fjac[0][1]
	lhsB[1, 1, k, i, j-1] = 1.0 + tmp1 * 2.0 * fjac[1][1] + tmp1 * 2.0 * dx2
	lhsB[2, 1, k, i, j-1] = tmp1 * 2.0 * fjac[2][1]
	lhsB[3, 1, k, i, j-1] = tmp1 * 2.0 * fjac[3][1]
	lhsB[4, 1, k, i, j-1] = tmp1 * 2.0 * fjac[4][1]
    
	lhsB[0, 2, k, i, j-1] = tmp1 * 2.0 * fjac[0][2]
	lhsB[1, 2, k, i, j-1] = tmp1 * 2.0 * fjac[1][2]
	lhsB[2, 2, k, i, j-1] = 1.0 + tmp1 * 2.0 * fjac[2][2] + tmp1 * 2.0 * dx3
	lhsB[3, 2, k, i, j-1] = tmp1 * 2.0 * fjac[3][2]
	lhsB[4, 2, k, i, j-1] = tmp1 * 2.0 * fjac[4][2]
    
	lhsB[0, 3, k, i, j-1] = tmp1 * 2.0 * fjac[0][3]
	lhsB[1, 3, k, i, j-1] = tmp1 * 2.0 * fjac[1][3]
	lhsB[2, 3, k, i, j-1] = tmp1 * 2.0 * fjac[2][3]
	lhsB[3, 3, k, i, j-1] = 1.0 + tmp1 * 2.0 * fjac[3][3] + tmp1 * 2.0 * dx4
	lhsB[4, 3, k, i, j-1] = tmp1 * 2.0 * fjac[4][3]
    
	lhsB[0, 4, k, i, j-1] = tmp1 * 2.0 * fjac[0][4]
	lhsB[1, 4, k, i, j-1] = tmp1 * 2.0 * fjac[1][4]
	lhsB[2, 4, k, i, j-1] = tmp1 * 2.0 * fjac[2][4]
	lhsB[3, 4, k, i, j-1] = tmp1 * 2.0 * fjac[3][4]
	lhsB[4, 4, k, i, j-1] = 1.0 + tmp1 * 2.0 * fjac[4][4] + tmp1 * 2.0 * dx5
	
	for m in range(5):
		t_u[m] = u[k][j][i+1][m]
	x_solve_gpu_device_fjac(fjac, t_u, rho_i[k][j][i+1], qs[k][j][i+1], square[k][j][i+1], c1, c2)
	x_solve_gpu_device_njac(njac, t_u, rho_i[k][j][i+1], con43, c3c4, c1345)
    
	lhsC[0, 0, k, i, j-1] = tmp2 * fjac[0][0] - tmp1 * njac[0][0] - tmp1 * dx1
	lhsC[1, 0, k, i, j-1] = tmp2 * fjac[1][0] - tmp1 * njac[1][0]
	lhsC[2, 0, k, i, j-1] = tmp2 * fjac[2][0] - tmp1 * njac[2][0]
	lhsC[3, 0, k, i, j-1] = tmp2 * fjac[3][0] - tmp1 * njac[3][0]
	lhsC[4, 0, k, i, j-1] = tmp2 * fjac[4][0] - tmp1 * njac[4][0]
    
	lhsC[0, 1, k, i, j-1] = tmp2 * fjac[0][1] - tmp1 * njac[0][1]
	lhsC[1, 1, k, i, j-1] = tmp2 * fjac[1][1] - tmp1 * njac[1][1] - tmp1 * dx2
	lhsC[2, 1, k, i, j-1] = tmp2 * fjac[2][1] - tmp1 * njac[2][1]
	lhsC[3, 1, k, i, j-1] = tmp2 * fjac[3][1] - tmp1 * njac[3][1]
	lhsC[4, 1, k, i, j-1] = tmp2 * fjac[4][1] - tmp1 * njac[4][1]
    
	lhsC[0, 2, k, i, j-1] = tmp2 * fjac[0][2] - tmp1 * njac[0][2]
	lhsC[1, 2, k, i, j-1] = tmp2 * fjac[1][2] - tmp1 * njac[1][2]
	lhsC[2, 2, k, i, j-1] = tmp2 * fjac[2][2] - tmp1 * njac[2][2] - tmp1 * dx3
	lhsC[3, 2, k, i, j-1] = tmp2 * fjac[3][2] - tmp1 * njac[3][2]
	lhsC[4, 2, k, i, j-1] = tmp2 * fjac[4][2] - tmp1 * njac[4][2]
    
	lhsC[0, 3, k, i, j-1] = tmp2 * fjac[0][3] - tmp1 * njac[0][3]
	lhsC[1, 3, k, i, j-1] = tmp2 * fjac[1][3] - tmp1 * njac[1][3]
	lhsC[2, 3, k, i, j-1] = tmp2 * fjac[2][3] - tmp1 * njac[2][3]
	lhsC[3, 3, k, i, j-1] = tmp2 * fjac[3][3] - tmp1 * njac[3][3] - tmp1 * dx4
	lhsC[4, 3, k, i, j-1] = tmp2 * fjac[4][3] - tmp1 * njac[4][3]
    
	lhsC[0, 4, k, i, j-1] = tmp2 * fjac[0][4] - tmp1 * njac[0][4]
	lhsC[1, 4, k, i, j-1] = tmp2 * fjac[1][4] - tmp1 * njac[1][4]
	lhsC[2, 4, k, i, j-1] = tmp2 * fjac[2][4] - tmp1 * njac[2][4]
	lhsC[3, 4, k, i, j-1] = tmp2 * fjac[3][4] - tmp1 * njac[3][4]
	lhsC[4, 4, k, i, j-1] = tmp2 * fjac[4][4] - tmp1 * njac[4][4] - tmp1 * dx5
#END x_solve_gpu_kernel_2()


# ---------------------------------------------------------------------
# this function computes the left hand side in the xi-direction
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :, :], float64[:, :, :, :, :], float64[:, :, :, :, :], int32, int32, int32, int32)')
def x_solve_gpu_kernel_1(lhsA, 
						lhsB, 
						lhsC,
						KMAX, JMAX, IMAX, PROBLEM_SIZE):
	t_k = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
	mn = int(t_k / PROBLEM_SIZE)
	k = int(t_k % PROBLEM_SIZE)
	m = int(mn / 5)
	n = int(mn % 5)
	j = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x+1

	if k+0 < 1 or k+0 > KMAX-2 or k >= PROBLEM_SIZE or j > JMAX-2 or m >= 5:
		return

	k += 0
	isize = IMAX - 1

	# ---------------------------------------------------------------------
	# now jacobians set, so form left hand side in x direction
	# ---------------------------------------------------------------------
	lhsA[m, n, k, 0, j-1] = 0.0
	lhsB[m, n, k, 0, j-1] = (1.0 if m == n else 0.0)
	lhsC[m, n, k, 0, j-1] = 0.0

	lhsA[m, n, k, isize, j-1] = 0.0
	lhsB[m, n, k, isize, j-1] = (1.0 if m == n else 0.0)
	lhsC[m, n, k, isize, j-1] = 0.0
#END x_solve_gpu_kernel_1()


# ---------------------------------------------------------------------
# performs line solves in X direction by first factoring
# the block-tridiagonal matrix into an upper triangular matrix, 
# and then performing back substitution to solve for the unknow
# vectors of each line.  
# 
# make sure we treat elements zero to cell_size in the direction
# of the sweep. 
# ---------------------------------------------------------------------
def x_solve_gpu(qs_device, 
				rho_i_device, 
				square_device, 
				u_device,
				rhs_device,
				lhsA_device, 
				lhsB_device, 
				lhsC_device,
				const_x_solve_device):
	# ---------------------------------------------------------------------
	# this function computes the left hand side in the xi-direction
	# ---------------------------------------------------------------------
	# determine a (labeled f) and n jacobians
	# ---------------------------------------------------------------------
	amount_of_work    = [grid_points[1]-2, npbparams.PROBLEM_SIZE*5*5]
	amount_of_threads = [THREADS_PER_BLOCK_ON_X_SOLVE_1, 1]
	
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     1)
	threadSize = (amount_of_threads[0], amount_of_threads[1], 1)
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_X_SOLVE_1)
	#print("threadSize=[%d, %d]" % (threadSize[0], threadSize[1]))
	#print("blockSize=[%d, %d]" % (blockSize[0], blockSize[1]))
	x_solve_gpu_kernel_1[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](lhsA_device, 
								lhsB_device, 
								lhsC_device,
								KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_X_SOLVE_1)
		
	amount_of_work    = [grid_points[0]-2, grid_points[1]-2, npbparams.PROBLEM_SIZE]
	amount_of_threads = [1, THREADS_PER_BLOCK_ON_X_SOLVE_2, 1]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_X_SOLVE_2)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	x_solve_gpu_kernel_2[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](qs_device, 
								rho_i_device, 
								square_device, 
								u_device,
								lhsA_device,
								lhsB_device, 
								lhsC_device, 
								const_x_solve_device,
								KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_X_SOLVE_2)

	sharedMemPerBlock = device_prop.MAX_SHARED_MEMORY_PER_BLOCK
	max_amount_of_threads_j = min( int(THREADS_PER_BLOCK_ON_X_SOLVE_3 / 5), int(sharedMemPerBlock / (rhs_device.dtype.itemsize*(3*5*5+2*5))) ) 
	max_amount_of_threads_j = int(max_amount_of_threads_j / 2)
	
	amount_of_threads = [max_amount_of_threads_j, 5]
	amount_of_work    = [grid_points[1]-2, npbparams.PROBLEM_SIZE * 5]
	
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]),
			     1)
	threadSize = (amount_of_threads[0], amount_of_threads[1], 1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_X_SOLVE_3)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	#print("blockSize=[%d, %d, %d]" % (blockSize[0], blockSize[1], blockSize[2]))
	size_shared_data = rhs_device.dtype.itemsize * max_amount_of_threads_j * (3*5*5+2*5)
	#print("sharedMemory=%d" % (size_shared_data))
	x_solve_gpu_kernel_3[blockSize,
		threadSize,
		stream,
		size_shared_data](rhs_device, 
						lhsA_device, 
						lhsB_device, 
						lhsC_device,
						KMAX, JMAX, IMAX, npbparams.PROBLEM_SIZE)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_X_SOLVE_3)
#END x_solve_gpu()


# ---------------------------------------------------------------------
# compute eta-direction fluxes 
# ---------------------------------------------------------------------
# Input(write buffer) - us_device, vs_device, ws_device, qs_device, rho_i_device, square_device, u_device, rhs_device
# ---------------------------------------------------------------------
# Input(write buffer) - us_device, vs_device, ws_device, qs_device, rho_i_device, square_device, u_device, rhs_device
# ---------------------------------------------------------------------
# Output(read buffer) - rhs_device
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :], int32, int32, int32, float64)')
def compute_rhs_gpu_kernel_9(rhs,
							KMAX, JMAX, IMAX,
							dt):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y+1
	t_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = int(t_i / 5) + 1
	m = int(t_i % 5) 

	if k+0 < 1 or k+0 > KMAX-2 or k >= KMAX or j > JMAX-2 or i > IMAX-2:
		return

	rhs[k][j][i][m] = rhs[k][j][i][m] * dt
#END compute_rhs_gpu_kernel_9()


@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, float64)')
def compute_rhs_gpu_kernel_8(u,
							rhs,
							KMAX, JMAX, IMAX,
							dssp):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y+1
	t_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = int(t_i / 5) + 1            
	m = int(t_i % 5)
	
	if k+0 < 1 or k+0 > KMAX-2 or k >= KMAX or j > JMAX-2 or i > IMAX-2: 
		return

	if k == 1:
		rhs[k][j][i][m] = ( rhs[k][j][i][m]- dssp * 
			( 5.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] +
			  u[k+2][j][i][m]) )
	elif k == 2:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp * 
			(-4.0*u[k-1][j][i][m] + 6.0*u[k][j][i][m] -
			 4.0*u[k+1][j][i][m] + u[k+2][j][i][m]) )
	elif k == KMAX-3:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp *
			( u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
			  6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] ) )
	elif k == KMAX-2:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp *
			( u[k-2][j][i][m] - 4.*u[k-1][j][i][m] +
			  5.*u[k][j][i][m] ) )
	else:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp * 
			(  u[k-2][j][i][m] - 4.0*u[k-1][j][i][m] + 
			   6.0*u[k][j][i][m] - 4.0*u[k+1][j][i][m] + 
			   u[k+2][j][i][m] ) )
#END compute_rhs_gpu_kernel_8()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def compute_rhs_gpu_kernel_7(us,
							vs,
							ws,
							qs,
							rho_i,
							square,
							u,
							rhs,
							KMAX, JMAX, IMAX,
							dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1,
							zzcon2, zzcon3, zzcon4, zzcon5,
							tz2, con43, c1, c2):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y+1
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x+1

	if k+0 < 1 or k+0 > KMAX-2 or k >= KMAX or j > JMAX-2 or i > IMAX-2: 
		return

	wijk = ws[k][j][i]
	wp1 = ws[k+1][j][i]
	wm1 = ws[k-1][j][i]

	rhs[k][j][i][0] = ( rhs[k][j][i][0] + dz1tz1 * 
		(u[k+1][j][i][0] - 2.0*u[k][j][i][0] + 
		 u[k-1][j][i][0]) -
		tz2 * (u[k+1][j][i][3] - u[k-1][j][i][3]) )
	rhs[k][j][i][1] = ( rhs[k][j][i][1] + dz2tz1 * 
		(u[k+1][j][i][1] - 2.0*u[k][j][i][1] + 
		 u[k-1][j][i][1]) +
		zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + 
				us[k-1][j][i]) -
		tz2 * (u[k+1][j][i][1]*wp1 - 
				u[k-1][j][i][1]*wm1) )
	rhs[k][j][i][2] = ( rhs[k][j][i][2] + dz3tz1 * 
		(u[k+1][j][i][2] - 2.0*u[k][j][i][2] + 
		 u[k-1][j][i][2]) +
		zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + 
				vs[k-1][j][i]) -
		tz2 * (u[k+1][j][i][2]*wp1 - 
				u[k-1][j][i][2]*wm1) )
	rhs[k][j][i][3] = ( rhs[k][j][i][3] + dz4tz1 * 
		(u[k+1][j][i][3] - 2.0*u[k][j][i][3] + 
		 u[k-1][j][i][3]) +
		zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
		tz2 * (u[k+1][j][i][3]*wp1 - 
				u[k-1][j][i][3]*wm1 +
				(u[k+1][j][i][4] - square[k+1][j][i] - 
				 u[k-1][j][i][4] + square[k-1][j][i])
				*c2) )
	rhs[k][j][i][4] = ( rhs[k][j][i][4] + dz5tz1 * 
		(u[k+1][j][i][4] - 2.0*u[k][j][i][4] + 
		 u[k-1][j][i][4]) +
		zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + 
				qs[k-1][j][i]) +
		zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + 
				wm1*wm1) +
		zzcon5 * (u[k+1][j][i][4]*rho_i[k+1][j][i] - 
				2.0*u[k][j][i][4]*rho_i[k][j][i] +
				u[k-1][j][i][4]*rho_i[k-1][j][i]) -
		tz2 * ( (c1*u[k+1][j][i][4] - 
					c2*square[k+1][j][i])*wp1 -
				(c1*u[k-1][j][i][4] - 
				 c2*square[k-1][j][i])*wm1) )
#END compute_rhs_gpu_kernel_7()


@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, float64)')
def compute_rhs_gpu_kernel_6(u,
							rhs,
							KMAX, JMAX, IMAX,
							dssp):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y+1
	t_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = int(t_i / 5) + 1
	m = int(t_i % 5)

	if k+0 < 1 or k+0 > KMAX-2 or k >= KMAX or j > JMAX-2 or i > IMAX-2: 
		return

	if j == 1:
		rhs[k][j][i][m] = ( rhs[k][j][i][m]- dssp * 
			( 5.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] +
			  u[k][j+2][i][m]) )
	elif j == 2:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp * 
			(-4.0*u[k][j-1][i][m] + 6.0*u[k][j][i][m] -
			 4.0*u[k][j+1][i][m] + u[k][j+2][i][m]) )
	elif j == JMAX-3:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp *
			( u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
			  6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] ) )
	elif j == JMAX-2:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp *
			( u[k][j-2][i][m] - 4.*u[k][j-1][i][m] +
			  5.*u[k][j][i][m] ) )
	else:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp * 
			(  u[k][j-2][i][m] - 4.0*u[k][j-1][i][m] + 
			   6.0*u[k][j][i][m] - 4.0*u[k][j+1][i][m] + 
			   u[k][j+2][i][m] ) )
#END compute_rhs_gpu_kernel_6()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def compute_rhs_gpu_kernel_5(us,
							vs,
							ws,
							qs,
							rho_i,
							square,
							u,
							rhs,
							KMAX, JMAX, IMAX,
							dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
							yycon2, yycon3, yycon4, yycon5, 
							ty2, con43, c1, c2):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y+1
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x+1

	if k+0 < 1 or k+0 > KMAX-2 or k >= KMAX or j > JMAX-2 or i > IMAX-2: 
		return

	vijk = vs[k][j][i]
	vp1 = vs[k][j+1][i]
	vm1 = vs[k][j-1][i]
	rhs[k][j][i][0] = ( rhs[k][j][i][0] + dy1ty1 * 
		(u[k][j+1][i][0] - 2.0*u[k][j][i][0] + 
		 u[k][j-1][i][0]) -
		ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]) )
	rhs[k][j][i][1] = ( rhs[k][j][i][1] + dy2ty1 * 
		(u[k][j+1][i][1] - 2.0*u[k][j][i][1] + 
		 u[k][j-1][i][1]) +
		yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + 
				us[k][j-1][i]) -
		ty2 * (u[k][j+1][i][1]*vp1 - 
				u[k][j-1][i][1]*vm1) )
	rhs[k][j][i][2] = ( rhs[k][j][i][2] + dy3ty1 * 
		(u[k][j+1][i][2] - 2.0*u[k][j][i][2] + 
		 u[k][j-1][i][2]) +
		yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
		ty2 * (u[k][j+1][i][2]*vp1 - 
				u[k][j-1][i][2]*vm1 +
				(u[k][j+1][i][4] - square[k][j+1][i] - 
				 u[k][j-1][i][4] + square[k][j-1][i])
				*c2) )
	rhs[k][j][i][3] = ( rhs[k][j][i][3] + dy4ty1 * 
		(u[k][j+1][i][3] - 2.0*u[k][j][i][3] + 
		 u[k][j-1][i][3]) +
		yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + 
				ws[k][j-1][i]) -
		ty2 * (u[k][j+1][i][3]*vp1 - 
				u[k][j-1][i][3]*vm1) )
	rhs[k][j][i][4] = ( rhs[k][j][i][4] + dy5ty1 * 
		(u[k][j+1][i][4] - 2.0*u[k][j][i][4] + 
		 u[k][j-1][i][4]) +
		yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + 
				qs[k][j-1][i]) +
		yycon4 * (vp1*vp1       - 2.0*vijk*vijk + 
				vm1*vm1) +
		yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] - 
				2.0*u[k][j][i][4]*rho_i[k][j][i] +
				u[k][j-1][i][4]*rho_i[k][j-1][i]) -
		ty2 * ((c1*u[k][j+1][i][4] - 
					c2*square[k][j+1][i]) * vp1 -
				(c1*u[k][j-1][i][4] - 
				 c2*square[k][j-1][i]) * vm1) )
#END compute_rhs_gpu_kernel_4()


# ---------------------------------------------------------------------
# compute xi-direction fluxes 
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, float64)')
def compute_rhs_gpu_kernel_4(u,
							rhs,
							KMAX, JMAX, IMAX,
							dssp):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y+1
	t_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = int(t_i / 5) + 1
	m = int(t_i % 5)

	if k+0 < 1 or k+0 > KMAX - 2 or k >= KMAX or j > JMAX-2 or i > IMAX-2:
		return

	if i == 1:
		rhs[k][j][i][m] = ( rhs[k][j][i][m]- dssp * 
			( 5.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] +
			  u[k][j][i+2][m]) )
	elif i == 2:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp * 
			(-4.0*u[k][j][i-1][m] + 6.0*u[k][j][i][m] -
			 4.0*u[k][j][i+1][m] + u[k][j][i+2][m] ) )
	elif i == IMAX-3:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp *
			( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
			  6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] ) )
	elif i == IMAX-2:
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp *
			( u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] +
			  5.0*u[k][j][i][m] ) )
	else: 
		rhs[k][j][i][m] = ( rhs[k][j][i][m] - dssp * 
			(  u[k][j][i-2][m] - 4.0*u[k][j][i-1][m] + 
			   6.0*u[k][j][i][m] - 4.0*u[k][j][i+1][m] + 
			   u[k][j][i+2][m] ) )
#END compute_rhs_gpu_kernel_4()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)')
def compute_rhs_gpu_kernel_3(us,
							vs,
							ws,
							qs,
							rho_i,
							square,
							u,
							rhs,
							KMAX, JMAX, IMAX,
							dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1,
							xxcon2, xxcon3, xxcon4, xxcon5,
							tx2, con43, c1, c2):
	#1 <= k <= KMAX-2
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + 1
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + 1

	if (k+0) < 1 or (k+0) > KMAX-2 or k >= KMAX or j > JMAX-2 or i > IMAX-2:
		return

	uijk = us[k][j][i]
	up1 = us[k][j][i+1]
	um1 = us[k][j][i-1]

	rhs[k][j][i][0] = ( rhs[k][j][i][0] + dx1tx1 * 
		(u[k][j][i+1][0] - 2.0*u[k][j][i][0] + 
		 u[k][j][i-1][0]) -
		tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]) )

	rhs[k][j][i][1] = ( rhs[k][j][i][1] + dx2tx1 * 
		(u[k][j][i+1][1] - 2.0*u[k][j][i][1] + 
		 u[k][j][i-1][1]) +
		xxcon2*con43 * (up1 - 2.0*uijk + um1) -
		tx2 * (u[k][j][i+1][1]*up1 - 
				u[k][j][i-1][1]*um1 +
				(u[k][j][i+1][4]- square[k][j][i+1]-
				 u[k][j][i-1][4]+ square[k][j][i-1])*
				c2) )

	rhs[k][j][i][2] = ( rhs[k][j][i][2] + dx3tx1 * 
		(u[k][j][i+1][2] - 2.0*u[k][j][i][2] +
		 u[k][j][i-1][2]) +
		xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
				vs[k][j][i-1]) -
		tx2 * (u[k][j][i+1][2]*up1 - 
				u[k][j][i-1][2]*um1) )

	rhs[k][j][i][3] = ( rhs[k][j][i][3] + dx4tx1 * 
		(u[k][j][i+1][3] - 2.0*u[k][j][i][3] +
		 u[k][j][i-1][3]) +
		xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
				ws[k][j][i-1]) -
		tx2 * (u[k][j][i+1][3]*up1 - 
				u[k][j][i-1][3]*um1) )

	rhs[k][j][i][4] = ( rhs[k][j][i][4] + dx5tx1 * 
		(u[k][j][i+1][4] - 2.0*u[k][j][i][4] +
		 u[k][j][i-1][4]) +
		xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
				qs[k][j][i-1]) +
		xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
				um1*um1) +
		xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] - 
				2.0*u[k][j][i][4]*rho_i[k][j][i] +
				u[k][j][i-1][4]*rho_i[k][j][i-1]) -
		tx2 * ( (c1*u[k][j][i+1][4] - 
					c2*square[k][j][i+1])*up1 -
				(c1*u[k][j][i-1][4] - 
				 c2*square[k][j][i-1])*um1 ) )
#END compute_rhs_gpu_kernel_3()


# ---------------------------------------------------------------------
# copy the exact forcing term to the right hand side; because 
# this forcing term is known, we can store it on the whole grid
# including the boundary                   
# ---------------------------------------------------------------------
@cuda.jit('void(float64[:, :, :, :], float64[:, :, :, :], int32, int32, int32)')
def compute_rhs_gpu_kernel_2(rhs,  
							forcing,
							KMAX, JMAX, IMAX):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	t_i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	i = int(t_i / 5)
	m = int(t_i % 5)

	if (k + 0) > KMAX-1 or k >= KMAX or j > JMAX-1 or i > IMAX-1:
		return

	rhs[k][j][i][m] = forcing[k][j][i][m]
#END compute_rhs_gpu_kernel_2()


@cuda.jit('void(float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :], float64[:, :, :, :], int32, int32, int32)')
def compute_rhs_gpu_kernel_1(rho_i, 
							us, 
							vs,
							ws, 
							qs,
							square,
							u,
							KMAX, JMAX, IMAX):
	k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if k >= KMAX or j > JMAX-1 or i > IMAX-1:
		return

	t_u = cuda.local.array(4, numba.float64)

	for m in range(4):
		t_u[m] = u[k][j][i][m]

	rho_inv = 1.0 / t_u[0]
	rho_i[k][j][i] = rho_inv
	us[k][j][i] = t_u[1] * rho_inv
	vs[k][j][i] = t_u[2] * rho_inv
	ws[k][j][i] = t_u[3] * rho_inv
	square[k][j][i] = ( 0.5* (
			t_u[1]*t_u[1] + 
			t_u[2]*t_u[2] +
			t_u[3]*t_u[3] ) * rho_inv )
	qs[k][j][i] = square[k][j][i] * rho_inv
#END compute_rhs_gpu_kernel_1()


def compute_rhs_gpu(rho_i_device, 
					us_device, 
					vs_device, 
					ws_device, 
					qs_device, 
					square_device, 
					u_device,
					rhs_device,
					forcing_device):
	work_base = 0
	work_num_item = min(npbparams.PROBLEM_SIZE, grid_points[2] - work_base)
	copy_num_item = min(npbparams.PROBLEM_SIZE, grid_points[2] - work_base)

	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_1, 1, 1]
	amount_of_work    = [grid_points[0], grid_points[1], copy_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_1)
	
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_1[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](rho_i_device, 
				us_device, 
				vs_device, 
				ws_device, 
				qs_device, 
				square_device, 
				u_device,
				KMAX, JMAX, IMAX)
		
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_1)
		
	# ---------------------------------------------------------------------
	# copy the exact forcing term to the right hand side; because 
	# this forcing term is known, we can store it on the whole grid
	# including the boundary                   
	# ---------------------------------------------------------------------
	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_2, 1, 1]
	amount_of_work    = [grid_points[0] * 5, grid_points[1], work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_2)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_2[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](rhs_device,
								forcing_device,
								KMAX, JMAX, IMAX)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_2)
		
	# ---------------------------------------------------------------------
	# compute xi-direction fluxes 
	# ---------------------------------------------------------------------
	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_3, 1, 1]
	amount_of_work    = [grid_points[0] - 2, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_3)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_3[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](
				us_device, 
				vs_device, 
				ws_device, 
				qs_device, 
				rho_i_device, 
				square_device,
				u_device,
				rhs_device,
				KMAX, JMAX, IMAX,
				dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1,
				xxcon2, xxcon3, xxcon4, xxcon5,
				tx2, con43, c1, c2)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_3)

	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_4, 1, 1]
	amount_of_work    = [(grid_points[0] - 2) * 5, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_4)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_4[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](u_device,
								rhs_device,
								KMAX, JMAX, IMAX,
								dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_4)
		
	# ---------------------------------------------------------------------
	# compute eta-direction fluxes 
	# ---------------------------------------------------------------------
	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_5, 1, 1]
	amount_of_work    = [grid_points[0] - 2, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_5)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_5[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](us_device, 
								vs_device, 
								ws_device, 
								qs_device, 
								rho_i_device, 
								square_device,
								u_device, 
								rhs_device,
								KMAX, JMAX, IMAX,
								dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
								yycon2, yycon3, yycon4, yycon5, 
								ty2, con43, c1, c2)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_5)

	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_6, 1, 1]
	amount_of_work    = [(grid_points[0] - 2) * 5, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_6)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_6[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](u_device,
							rhs_device,
							KMAX, JMAX, IMAX,
							dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_6)

	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_7, 1, 1]
	amount_of_work    = [grid_points[0] - 2, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_7)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_7[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](us_device, 
							vs_device, 
							ws_device, 
							qs_device, 
							rho_i_device, 
							square_device,
							u_device, 
							rhs_device,
							KMAX, JMAX, IMAX,
							dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1,
							zzcon2, zzcon3, zzcon4, zzcon5,
							tz2, con43, c1, c2)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_7)
		
	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_8, 1, 1]
	amount_of_work    = [(grid_points[0] - 2) * 5, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_8)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_8[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](u_device, 
							rhs_device,
							KMAX, JMAX, IMAX,
							dssp)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_8)
		
	amount_of_threads = [THREADS_PER_BLOCK_ON_RHS_9, 1, 1]
	amount_of_work    = [(grid_points[0] - 2) * 5, grid_points[1] - 2, work_num_item]
	
	amount_of_work[2] = round_amount_of_work(amount_of_work[2], amount_of_threads[2])
	amount_of_work[1] = round_amount_of_work(amount_of_work[1], amount_of_threads[1])
	amount_of_work[0] = round_amount_of_work(amount_of_work[0], amount_of_threads[0])

	blockSize = (int(amount_of_work[0]/amount_of_threads[0]), 
			     int(amount_of_work[1]/amount_of_threads[1]), 
			     int(amount_of_work[2]/amount_of_threads[2]))
	threadSize = (amount_of_threads[0], amount_of_threads[1], amount_of_threads[2])

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RHS_9)
	#print("threadSize=[%d, %d, %d]" % (threadSize[0], threadSize[1], threadSize[2]))
	compute_rhs_gpu_kernel_9[blockSize,
		threadSize,
		stream,
		size_shared_data_empty](rhs_device,
							KMAX, JMAX, IMAX,
							dt)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RHS_9)
#END compute_rhs_gpu()


def adi_gpu(rho_i_device, 
			us_device, 
			vs_device, 
			ws_device, 
			qs_device, 
			square_device, 
			u_device,
			rhs_device,
			forcing_device):
	
	global lhsA_device, lhsB_device, lhsC_device
	global const_x_solve_device, const_y_solve_device, const_z_solve_device
	
	# ---------------------------------------------------------------------
	# compute the reciprocal of density, and the kinetic energy, 
	# and the speed of sound.
	# ---------------------------------------------------------------------
	compute_rhs_gpu(rho_i_device, us_device, vs_device, ws_device, 
				 qs_device, square_device, u_device, rhs_device, forcing_device)
	x_solve_gpu(qs_device, rho_i_device, square_device, u_device, rhs_device, 
				lhsA_device, lhsB_device, lhsC_device, const_x_solve_device)
	y_solve_gpu(qs_device, rho_i_device, square_device, u_device, rhs_device, 
				lhsA_device, lhsB_device, lhsC_device, const_y_solve_device)
	z_solve_gpu(qs_device, rho_i_device, square_device, u_device, rhs_device, 
				lhsA_device, lhsB_device, lhsC_device, const_z_solve_device)
	add_gpu(u_device, rhs_device)
#END adi_gpu()


#*****************************************************************
#************************* CPU FUNCTIONS *************************
#*****************************************************************
def rhs_norm(rms): #rms[5]
	for m in range(5): 
		rms[m] = 0.0
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				for m in range(5):
					add = rhs[k][j][i][m]
					rms[m] = rms[m] + add * add

	m_sqrt = math.sqrt
	for m in range(5):
		for d in range(3):
			rms[m] = rms[m] / (grid_points[d]-2)
		rms[m] = m_sqrt(rms[m])
#END rhs_norm()


# ---------------------------------------------------------------------
# this function computes the norm of the difference between the
# computed solution and the exact solution
# ---------------------------------------------------------------------
def error_norm(rms): #rms[5]
	u_exact = numpy.empty(5, dtype=numpy.float64)
	
	for m in range(5):
		rms[m] = 0.0
	for k in range(grid_points[2]):
		zeta = k * dKMAXm1
		for j in range(grid_points[1]):
			eta = j * dJMAXm1
			for i in range(grid_points[0]):
				xi = i * dIMAXm1
				exact_solution(xi, eta, zeta, u_exact)
				for m in range(5):
					add = u[k][j][i][m] - u_exact[m]
					rms[m] = rms[m] + add * add
	#END for k in range(grid_points[2]):
	
	m_sqrt = math.sqrt
	for m in range(5):
		for d in range(3):
			rms[m] = rms[m] / (grid_points[d]-2)
		rms[m] = m_sqrt(rms[m])
#END error_norm()


# ---------------------------------------------------------------------
# verification routine                         
# ---------------------------------------------------------------------
#
def verify():
	global u, forcing, rhs
	global u_device, forcing_device, rhs_device
	global rho_i_device, us_device, vs_device, ws_device, qs_device, square_device
	
	xce = numpy.empty(5, dtype=numpy.float64)
	xcr = numpy.empty(5, dtype=numpy.float64)
	# ---------------------------------------------------------------------
	# tolerance level
	# ---------------------------------------------------------------------
	epsilon=1.0e-08
	# ---------------------------------------------------------------------
	# compute the error norm and the residual norm, and exit if not printing
	# ---------------------------------------------------------------------
	error_norm(xce)
	
	u_device = cuda.to_device(u)
	forcing_device = cuda.to_device(forcing)
	
	compute_rhs_gpu(rho_i_device, us_device, vs_device, ws_device, 
				 qs_device, square_device, u_device, rhs_device, forcing_device)
	
	rhs = rhs_device.copy_to_host()
	
	rhs_norm(xcr)
	for m in range(5):
		xcr[m] = xcr[m] / dt
	
	verified = True
	dtref = 0.0
	xcrref = numpy.repeat(1.0, 5)
	xceref = numpy.repeat(1.0, 5)
	
	# ---------------------------------------------------------------------
	# reference data for 12X12X12 grids after 60 time steps, with DT = 1.0e-02
	# ---------------------------------------------------------------------
	if npbparams.CLASS == 'S':
		dtref = 1.0e-2
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 1.7034283709541311e-01
		xcrref[1] = 1.2975252070034097e-02
		xcrref[2] = 3.2527926989486055e-02
		xcrref[3] = 2.6436421275166801e-02
		xcrref[4] = 1.9211784131744430e-01
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 4.9976913345811579e-04
		xceref[1] = 4.5195666782961927e-05
		xceref[2] = 7.3973765172921357e-05
		xceref[3] = 7.3821238632439731e-05
		xceref[4] = 8.9269630987491446e-04
	# ---------------------------------------------------------------------
	# reference data for 24X24X24 grids after 200 time steps, with DT = 0.8d-3
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'W':
		dtref = 0.8e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.1125590409344e+03
		xcrref[1] = 0.1180007595731e+02
		xcrref[2] = 0.2710329767846e+02
		xcrref[3] = 0.2469174937669e+02
		xcrref[4] = 0.2638427874317e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.4419655736008e+01
		xceref[1] = 0.4638531260002e+00
		xceref[2] = 0.1011551749967e+01
		xceref[3] = 0.9235878729944e+00
		xceref[4] = 0.1018045837718e+02
	# ---------------------------------------------------------------------
	# reference data for 64X64X64 grids after 200 time steps, with DT = 0.8d-3
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'A':
		dtref = 0.8e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 1.0806346714637264e+02
		xcrref[1] = 1.1319730901220813e+01
		xcrref[2] = 2.5974354511582465e+01
		xcrref[3] = 2.3665622544678910e+01
		xcrref[4] = 2.5278963211748344e+02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 4.2348416040525025e+00
		xceref[1] = 4.4390282496995698e-01
		xceref[2] = 9.6692480136345650e-01
		xceref[3] = 8.8302063039765474e-01
		xceref[4] = 9.7379901770829278e+00
	# ---------------------------------------------------------------------
	# reference data for 102X102X102 grids after 200 time steps,
	# with DT = 3.0e-04
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'B':
		dtref = 3.0e-4
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 1.4233597229287254e+03
		xcrref[1] = 9.9330522590150238e+01
		xcrref[2] = 3.5646025644535285e+02
		xcrref[3] = 3.2485447959084092e+02
		xcrref[4] = 3.2707541254659363e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 5.2969847140936856e+01
		xceref[1] = 4.4632896115670668e+00
		xceref[2] = 1.3122573342210174e+01
		xceref[3] = 1.2006925323559144e+01
		xceref[4] = 1.2459576151035986e+02
	# ---------------------------------------------------------------------
	# reference data for 162X162X162 grids after 200 time steps,
	# with DT = 1.0e-04
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'C':
		dtref = 1.0e-4
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.62398116551764615e+04
		xcrref[1] = 0.50793239190423964e+03
		xcrref[2] = 0.15423530093013596e+04
		xcrref[3] = 0.13302387929291190e+04
		xcrref[4] = 0.11604087428436455e+05
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.16462008369091265e+03
		xceref[1] = 0.11497107903824313e+02
		xceref[2] = 0.41207446207461508e+02
		xceref[3] = 0.37087651059694167e+02
		xceref[4] = 0.36211053051841265e+03
	# ---------------------------------------------------------------------
	# reference data for 408x408x408 grids after 250 time steps,
	# with DT = 0.2e-04
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'D':
		dtref = 0.2e-4
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.2533188551738e+05
		xcrref[1] = 0.2346393716980e+04
		xcrref[2] = 0.6294554366904e+04
		xcrref[3] = 0.5352565376030e+04
		xcrref[4] = 0.3905864038618e+05
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.3100009377557e+03
		xceref[1] = 0.2424086324913e+02
		xceref[2] = 0.7782212022645e+02
		xceref[3] = 0.6835623860116e+02
		xceref[4] = 0.6065737200368e+03
	# ---------------------------------------------------------------------
	# reference data for 1020x1020x1020 grids after 250 time steps,
	# with DT = 0.4e-05
	# ---------------------------------------------------------------------
	elif npbparams.CLASS == 'E':
		dtref = 0.4e-5
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual.
		# ---------------------------------------------------------------------
		xcrref[0] = 0.9795372484517e+05
		xcrref[1] = 0.9739814511521e+04
		xcrref[2] = 0.2467606342965e+05
		xcrref[3] = 0.2092419572860e+05
		xcrref[4] = 0.1392138856939e+06
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error.
		# ---------------------------------------------------------------------
		xceref[0] = 0.4327562208414e+03 
		xceref[1] = 0.3699051964887e+02 
		xceref[2] = 0.1089845040954e+03 
		xceref[3] = 0.9462517622043e+02 
		xceref[4] = 0.7765512765309e+03 
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
# this function returns the exact solution at point xi, eta, zeta  
# ---------------------------------------------------------------------
@njit
def exact_solution(xi, eta, zeta, dtemp): #double dtemp[5]
	for m in range(5):
		dtemp[m] = ( ce[m][0]+
			xi*(ce[m][1]+
					xi*(ce[m][4]+
						xi*(ce[m][7]+
							xi*ce[m][10])))+
			eta*(ce[m][2]+
					eta*(ce[m][5]+
						eta*(ce[m][8]+
							eta*ce[m][11])))+
			zeta*(ce[m][3]+
					zeta*(ce[m][6]+
						zeta*(ce[m][9]+ 
							zeta*ce[m][12]))) )
#END exact_solution()


# ---------------------------------------------------------------------
# compute the right hand side based on exact solution
# ---------------------------------------------------------------------
@njit
def exact_rhs(forcing, ue, buf, cuf, q):
	dtemp = numpy.empty(5, dtype=numpy.float64)
	
	# ---------------------------------------------------------------------
	# initialize                                  
	# ---------------------------------------------------------------------
	for k in range(grid_points[2]):
		for j in range(grid_points[1]):
			for i in range(grid_points[0]):
				for m in range(5):
					forcing[k][j][i][m] = 0.0
	
	# ---------------------------------------------------------------------
	# xi-direction flux differences                      
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		zeta = k * dKMAXm1
		for j in range(1, grid_points[1]-1):
			eta = j * dJMAXm1
			for i in range(grid_points[0]):
				xi = i * dIMAXm1
				exact_solution(xi, eta, zeta, dtemp)
				for m in range(5):
					ue[i][m] = dtemp[m]

				dtpp = 1.0 / dtemp[0]
				for m in range(1, 5):
					buf[i][m] = dtpp * dtemp[m]

				cuf[i] = buf[i][1] * buf[i][1]
				buf[i][0] = cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3]
				q[i] = ( 0.5*(buf[i][1]*ue[i][1] + buf[i][2]*ue[i][2] +
						buf[i][3]*ue[i][3]) )

			for i in range(1, grid_points[0]-1):
				im1 = i-1
				ip1 = i+1
				forcing[k][j][i][0] = ( forcing[k][j][i][0] -
					tx2*( ue[ip1][1]-ue[im1][1] )+
					dx1tx1*(ue[ip1][0]-2.0*ue[i][0]+ue[im1][0]) )

				forcing[k][j][i][1] = ( forcing[k][j][i][1] - tx2 * (
						(ue[ip1][1]*buf[ip1][1]+c2*(ue[ip1][4]-q[ip1]))-
						(ue[im1][1]*buf[im1][1]+c2*(ue[im1][4]-q[im1])))+
					xxcon1*(buf[ip1][1]-2.0*buf[i][1]+buf[im1][1])+
					dx2tx1*( ue[ip1][1]-2.0* ue[i][1]+ue[im1][1]) )

				forcing[k][j][i][2] = ( forcing[k][j][i][2] - tx2 * (
						ue[ip1][2]*buf[ip1][1]-ue[im1][2]*buf[im1][1])+
					xxcon2*(buf[ip1][2]-2.0*buf[i][2]+buf[im1][2])+
					dx3tx1*( ue[ip1][2]-2.0*ue[i][2] +ue[im1][2]) )

				forcing[k][j][i][3] = ( forcing[k][j][i][3] - tx2*(
						ue[ip1][3]*buf[ip1][1]-ue[im1][3]*buf[im1][1])+
					xxcon2*(buf[ip1][3]-2.0*buf[i][3]+buf[im1][3])+
					dx4tx1*( ue[ip1][3]-2.0* ue[i][3]+ ue[im1][3]) )

				forcing[k][j][i][4] = ( forcing[k][j][i][4] - tx2*(
						buf[ip1][1]*(c1*ue[ip1][4]-c2*q[ip1])-
						buf[im1][1]*(c1*ue[im1][4]-c2*q[im1]))+
					0.5*xxcon3*(buf[ip1][0]-2.0*buf[i][0]+
							buf[im1][0])+
					xxcon4*(cuf[ip1]-2.0*cuf[i]+cuf[im1])+
					xxcon5*(buf[ip1][4]-2.0*buf[i][4]+buf[im1][4])+
					dx5tx1*( ue[ip1][4]-2.0* ue[i][4]+ ue[im1][4]) )
			#END for i in range(1, grid_points[0]-1):
			
			# ---------------------------------------------------------------------
			# fourth-order dissipation                         
			# ---------------------------------------------------------------------
			for m in range(5):
				i = 1
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(5.0*ue[i][m] - 4.0*ue[i+1][m] +ue[i+2][m]) )
				i = 2
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(-4.0*ue[i-1][m] + 6.0*ue[i][m] -
					 4.0*ue[i+1][m] +     ue[i+2][m]) )

			for i in range(3, grid_points[0]-3):
				for m in range(5):
					forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp*
						(ue[i-2][m] - 4.0*ue[i-1][m] +
						 6.0*ue[i][m] - 4.0*ue[i+1][m] + ue[i+2][m]) )

			for m in range(5):
				i = grid_points[0]-3
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(ue[i-2][m] - 4.0*ue[i-1][m] +
					 6.0*ue[i][m] - 4.0*ue[i+1][m]) )
				i = grid_points[0]-2
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(ue[i-2][m] - 4.0*ue[i-1][m] + 5.0*ue[i][m]) )
		#END for j in range(1, grid_points[1]-1):
	#END for k in range(1, grid_points[2]-1):
	
	# ---------------------------------------------------------------------
	# eta-direction flux differences             
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		zeta = k * dKMAXm1
		for i in range(1, grid_points[0]-1):
			xi = i * dIMAXm1
			for j in range(grid_points[1]):
				eta = j * dJMAXm1
				exact_solution(xi, eta, zeta, dtemp)
				for m in range(5):
					ue[j][m] = dtemp[m]

				dtpp = 1.0 / dtemp[0]
				for m in range(1, 5):
					buf[j][m] = dtpp * dtemp[m]

				cuf[j] = buf[j][2] * buf[j][2]
				buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3]
				q[j] = ( 0.5*(buf[j][1]*ue[j][1] + buf[j][2]*ue[j][2] +
						buf[j][3]*ue[j][3]) )

			for j in range(1, grid_points[1]-1):
				jm1 = j-1
				jp1 = j+1
				
				forcing[k][j][i][0] = ( forcing[k][j][i][0] -
					ty2*( ue[jp1][2]-ue[jm1][2] )+
					dy1ty1*(ue[jp1][0]-2.0*ue[j][0]+ue[jm1][0]) )

				forcing[k][j][i][1] = ( forcing[k][j][i][1] - ty2*(
						ue[jp1][1]*buf[jp1][2]-ue[jm1][1]*buf[jm1][2])+
					yycon2*(buf[jp1][1]-2.0*buf[j][1]+buf[jm1][1])+
					dy2ty1*( ue[jp1][1]-2.0* ue[j][1]+ ue[jm1][1]) )

				forcing[k][j][i][2] = ( forcing[k][j][i][2] - ty2*(
						(ue[jp1][2]*buf[jp1][2]+c2*(ue[jp1][4]-q[jp1]))-
						(ue[jm1][2]*buf[jm1][2]+c2*(ue[jm1][4]-q[jm1])))+
					yycon1*(buf[jp1][2]-2.0*buf[j][2]+buf[jm1][2])+
					dy3ty1*( ue[jp1][2]-2.0*ue[j][2] +ue[jm1][2]) )

				forcing[k][j][i][3] = ( forcing[k][j][i][3] - ty2*(
						ue[jp1][3]*buf[jp1][2]-ue[jm1][3]*buf[jm1][2])+
					yycon2*(buf[jp1][3]-2.0*buf[j][3]+buf[jm1][3])+
					dy4ty1*( ue[jp1][3]-2.0*ue[j][3]+ ue[jm1][3]) )

				forcing[k][j][i][4] = ( forcing[k][j][i][4] - ty2*(
						buf[jp1][2]*(c1*ue[jp1][4]-c2*q[jp1])-
						buf[jm1][2]*(c1*ue[jm1][4]-c2*q[jm1]))+
					0.5*yycon3*(buf[jp1][0]-2.0*buf[j][0]+
							buf[jm1][0])+
					yycon4*(cuf[jp1]-2.0*cuf[j]+cuf[jm1])+
					yycon5*(buf[jp1][4]-2.0*buf[j][4]+buf[jm1][4])+
					dy5ty1*(ue[jp1][4]-2.0*ue[j][4]+ue[jm1][4]) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation                      
			# ---------------------------------------------------------------------
			for m in range(5):
				j = 1
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(5.0*ue[j][m] - 4.0*ue[j+1][m] +ue[j+2][m]) )
				j = 2
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(-4.0*ue[j-1][m] + 6.0*ue[j][m] -
					 4.0*ue[j+1][m] +       ue[j+2][m]) )

			for j in range(3, grid_points[1]-3):
				for m in range(5):
					forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp*
						(ue[j-2][m] - 4.0*ue[j-1][m] +
						 6.0*ue[j][m] - 4.0*ue[j+1][m] + ue[j+2][m]) )

			for m in range(5):
				j = grid_points[1]-3
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(ue[j-2][m] - 4.0*ue[j-1][m] +
					 6.0*ue[j][m] - 4.0*ue[j+1][m]) )
				j = grid_points[1]-2
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(ue[j-2][m] - 4.0*ue[j-1][m] + 5.0*ue[j][m]) )
		#END for i in range(1, grid_points[0]-1):
	#END for k in range(1, grid_points[2]-1):
	
	# ---------------------------------------------------------------------
	# zeta-direction flux differences                      
	# ---------------------------------------------------------------------
	for j in range(1, grid_points[1]-1): 
		eta = j * dJMAXm1
		for i in range(1, grid_points[0]-1):
			xi = i * dIMAXm1
			for k in range(grid_points[2]):
				zeta = k * dKMAXm1
				exact_solution(xi, eta, zeta, dtemp)
				for m in range(5):
					ue[k][m] = dtemp[m]

				dtpp = 1.0 / dtemp[0]
				for m in range(1, 5):
					buf[k][m] = dtpp * dtemp[m]

				cuf[k]    = buf[k][3] * buf[k][3]
				buf[k][0] = cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2]
				q[k] = ( 0.5*(buf[k][1]*ue[k][1] + buf[k][2]*ue[k][2] +
						buf[k][3]*ue[k][3]) )

			for k in range(1, grid_points[2]-1):
				km1 = k-1
				kp1 = k+1
				forcing[k][j][i][0] = ( forcing[k][j][i][0] -
					tz2*( ue[kp1][3]-ue[km1][3] )+
					dz1tz1*(ue[kp1][0]-2.0*ue[k][0]+ue[km1][0]) )

				forcing[k][j][i][1] = ( forcing[k][j][i][1] - tz2 * (
						ue[kp1][1]*buf[kp1][3]-ue[km1][1]*buf[km1][3])+
					zzcon2*(buf[kp1][1]-2.0*buf[k][1]+buf[km1][1])+
					dz2tz1*( ue[kp1][1]-2.0* ue[k][1]+ ue[km1][1]) )

				forcing[k][j][i][2] = ( forcing[k][j][i][2] - tz2 * (
						ue[kp1][2]*buf[kp1][3]-ue[km1][2]*buf[km1][3])+
					zzcon2*(buf[kp1][2]-2.0*buf[k][2]+buf[km1][2])+
					dz3tz1*(ue[kp1][2]-2.0*ue[k][2]+ue[km1][2]) )

				forcing[k][j][i][3] = ( forcing[k][j][i][3] - tz2 * (
						(ue[kp1][3]*buf[kp1][3]+c2*(ue[kp1][4]-q[kp1]))-
						(ue[km1][3]*buf[km1][3]+c2*(ue[km1][4]-q[km1])))+
					zzcon1*(buf[kp1][3]-2.0*buf[k][3]+buf[km1][3])+
					dz4tz1*( ue[kp1][3]-2.0*ue[k][3] +ue[km1][3]) )

				forcing[k][j][i][4] = ( forcing[k][j][i][4] - tz2 * (
						buf[kp1][3]*(c1*ue[kp1][4]-c2*q[kp1])-
						buf[km1][3]*(c1*ue[km1][4]-c2*q[km1]))+
					0.5*zzcon3*(buf[kp1][0]-2.0*buf[k][0]
							+buf[km1][0])+
					zzcon4*(cuf[kp1]-2.0*cuf[k]+cuf[km1])+
					zzcon5*(buf[kp1][4]-2.0*buf[k][4]+buf[km1][4])+
					dz5tz1*( ue[kp1][4]-2.0*ue[k][4]+ ue[km1][4]) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation                        
			# ---------------------------------------------------------------------
			for m in range(5):
				k = 1
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(5.0*ue[k][m] - 4.0*ue[k+1][m] +ue[k+2][m]) )
				k = 2
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(-4.0*ue[k-1][m] + 6.0*ue[k][m] -
					 4.0*ue[k+1][m] +       ue[k+2][m]) )

			for k in range(3, grid_points[2]-3):
				for m in range(5):
					forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp*
						(ue[k-2][m] - 4.0*ue[k-1][m] +
						 6.0*ue[k][m] - 4.0*ue[k+1][m] + ue[k+2][m]) )

			for m in range(5):
				k = grid_points[2]-3
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(ue[k-2][m] - 4.0*ue[k-1][m] +
					 6.0*ue[k][m] - 4.0*ue[k+1][m]) )
				k = grid_points[2]-2
				forcing[k][j][i][m] = ( forcing[k][j][i][m] - dssp *
					(ue[k-2][m] - 4.0*ue[k-1][m] + 5.0*ue[k][m]) )
		#END for i in range(1, grid_points[0]-1):
	#END for j in range(1, grid_points[1]-1): 

	# ---------------------------------------------------------------------
	# now change the sign of the forcing function
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				for m in range(5):
					forcing[k][j][i][m] = -1.0 * forcing[k][j][i][m]
#END exact_rhs()


# ---------------------------------------------------------------------
# this subroutine initializes the field variable u using 
# tri-linear transfinite interpolation of the boundary values     
# ---------------------------------------------------------------------
@njit
def initialize(u):
	#double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];
	Pface = numpy.empty((2, 3, 5), dtype=numpy.float64)
	temp = numpy.empty(5, dtype=numpy.float64)
	
	# ---------------------------------------------------------------------
	# later (in compute_rhs) we compute 1/u for every element. a few of 
	# the corner elements are not used, but it convenient (and faster) 
	# to compute the whole thing with a simple loop. make sure those 
	# values are nonzero by initializing the whole thing here. 
	# ---------------------------------------------------------------------
	for k in range(grid_points[2]):
		for j in range(grid_points[1]):
			for i in range(grid_points[0]):
				for m in range(5):
					u[k][j][i][m] = 1.0

	# ---------------------------------------------------------------------
	# first store the "interpolated" values everywhere on the grid    
	# ---------------------------------------------------------------------
	for k in range(grid_points[2]):
		zeta = k * dKMAXm1
		for j in range(grid_points[1]):
			eta = j * dJMAXm1 
			for i in range(grid_points[0]):
				xi = i * dIMAXm1
				for ix in range(2):
					Pface_aux = Pface[ix][0]
					exact_solution(ix, eta, zeta, Pface_aux)
				for iy in range(2):
					Pface_aux = Pface[iy][1]
					exact_solution(xi, iy, zeta, Pface_aux)
				for iz in range(2):
					Pface_aux = Pface[iz][2]
					exact_solution(xi, eta, iz, Pface_aux)
				for m in range(5):
					Pxi = xi*Pface[1][0][m]+(1.0-xi)*Pface[0][0][m]
					Peta = eta*Pface[1][1][m]+(1.0-eta)*Pface[0][1][m]
					Pzeta = zeta*Pface[1][2][m]+(1.0-zeta)*Pface[0][2][m]
					u[k][j][i][m] = ( Pxi+Peta+Pzeta- 
						Pxi*Peta-Pxi*Pzeta-Peta*Pzeta+ 
						Pxi*Peta*Pzeta )
	#END for k in range(grid_points[2]):
	
	# ---------------------------------------------------------------------
	# now store the exact values on the boundaries        
	# ---------------------------------------------------------------------
	# west face                                                  
	# ---------------------------------------------------------------------
	i = 0
	xi = 0.0
	for k in range(grid_points[2]):
		zeta = k * dKMAXm1
		for j in range(grid_points[1]):
			eta = j * dJMAXm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# east face                                                      
	# ---------------------------------------------------------------------
	i = grid_points[0]-1
	xi = 1.0
	for k in range(grid_points[2]):
		zeta = k * dKMAXm1
		for j in range(grid_points[1]):
			eta = j * dJMAXm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# south face                                                 
	# ---------------------------------------------------------------------
	j = 0
	eta = 0.0
	for k in range(grid_points[2]):
		zeta = k * dKMAXm1
		for i in range(grid_points[0]):
			xi = i * dIMAXm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# north face                                    
	# ---------------------------------------------------------------------
	j = grid_points[1]-1
	eta = 1.0
	for k in range(grid_points[2]):
		zeta = k * dKMAXm1
		for i in range(grid_points[0]):
			xi = i * dIMAXm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# bottom face                                       
	# ---------------------------------------------------------------------
	k = 0
	zeta = 0.0
	for j in range(grid_points[1]):
		eta = j * dJMAXm1
		for i in range(grid_points[0]):
			xi = i * dIMAXm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# top face     
	# ---------------------------------------------------------------------
	k = grid_points[2]-1
	zeta = 1.0
	for j in range(grid_points[1]):
		eta = j * dJMAXm1
		for i in range(grid_points[0]):
			xi = i * dIMAXm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]
#END initialize()


def main():
	global dt
	global grid_points
	global timeron
	global u
	global forcing, ue, buf, cuf, q
	global rho_i, us, vs, ws, square, qs, rhs
	global fjac, njac, lhs
	
	global u_device
	global forcing_device
	global rho_i_device, us_device, vs_device, ws_device, square_device, qs_device, rhs_device
	
	# ---------------------------------------------------------------------
	# root node reads input file (if it exists) else takes
	# defaults from parameters
	# ---------------------------------------------------------------------
	niter = 0
	
	fp = os.path.isfile("inputbt.data")
	if fp:
		print(" Reading from input file inputbt.data") 
		print(" ERROR - Not implemented") 
		sys.exit()
	else:
		print(" No input file inputbt.data. Using compiled defaults")
		niter = npbparams.NITER_DEFAULT
		dt = npbparams.DT_DEFAULT
		grid_points[0] = npbparams.PROBLEM_SIZE
		grid_points[1] = npbparams.PROBLEM_SIZE
		grid_points[2] = npbparams.PROBLEM_SIZE
	
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		# new
		c_timers.timer_clear(PROFILING_ADD)
		c_timers.timer_clear(PROFILING_RHS_1)
		c_timers.timer_clear(PROFILING_RHS_2)
		c_timers.timer_clear(PROFILING_RHS_3)
		c_timers.timer_clear(PROFILING_RHS_4)
		c_timers.timer_clear(PROFILING_RHS_5)
		c_timers.timer_clear(PROFILING_RHS_6)
		c_timers.timer_clear(PROFILING_RHS_7)
		c_timers.timer_clear(PROFILING_RHS_8)
		c_timers.timer_clear(PROFILING_RHS_9)
		c_timers.timer_clear(PROFILING_X_SOLVE_1)
		c_timers.timer_clear(PROFILING_X_SOLVE_2)
		c_timers.timer_clear(PROFILING_X_SOLVE_3)
		c_timers.timer_clear(PROFILING_Y_SOLVE_1)
		c_timers.timer_clear(PROFILING_Y_SOLVE_2)
		c_timers.timer_clear(PROFILING_Y_SOLVE_3)
		c_timers.timer_clear(PROFILING_Z_SOLVE_1)
		c_timers.timer_clear(PROFILING_Z_SOLVE_2)
		c_timers.timer_clear(PROFILING_Z_SOLVE_3)
		# old
		c_timers.timer_clear(PROFILING_EXACT_RHS_1)
		c_timers.timer_clear(PROFILING_EXACT_RHS_2)
		c_timers.timer_clear(PROFILING_EXACT_RHS_3)
		c_timers.timer_clear(PROFILING_EXACT_RHS_4)
		c_timers.timer_clear(PROFILING_ERROR_NORM_1)
		c_timers.timer_clear(PROFILING_ERROR_NORM_2)
		c_timers.timer_clear(PROFILING_INITIALIZE)
		c_timers.timer_clear(PROFILING_RHS_NORM_1)
		c_timers.timer_clear(PROFILING_RHS_NORM_2)
		
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - BT Benchmark\n")
	print(" Size: %4dx%4dx%4d" % (grid_points[0], grid_points[1], grid_points[2]))
	print(" Iterations: %4d    dt: %10.6f" % (niter, dt))
	print()
	if (grid_points[0] > IMAX) or (grid_points[1] > JMAX) or (grid_points[2] > KMAX):
		print(" %d, %d, %d" % (grid_points[0], grid_points[1], grid_points[2]))
		print(" Problem size too big for compiled array sizes")
		sys.exit()

	setup_gpu()
	set_constants()
	initialize(u)
	exact_rhs(forcing, ue, buf, cuf, q)
	
	# ---------------------------------------------------------------------
	# do one time step to touch all code, and reinitialize
	# ---------------------------------------------------------------------
	u_device = cuda.to_device(u)
	forcing_device = cuda.to_device(forcing)
	
	adi_gpu(rho_i_device, us_device, vs_device, ws_device, 
		 qs_device, square_device, u_device, rhs_device, forcing_device)
	
	qs = qs_device.copy_to_host()
	square = square_device.copy_to_host()
	rho_i = rho_i_device.copy_to_host()
	rhs = rhs_device.copy_to_host()
	u = u_device.copy_to_host()
	
	initialize(u)
	
	u_device = cuda.to_device(u)
	forcing_device = cuda.to_device(forcing)
	
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	for step in range(1, niter+1):
		if (step % 20) == 0 or step == 1:
			print(" Time step %4d" % (step))
		adi_gpu(rho_i_device, us_device, vs_device, ws_device, 
		 qs_device, square_device, u_device, rhs_device, forcing_device)
	
	c_timers.timer_stop_assync(PROFILING_TOTAL_TIME)
	tmax = c_timers.timer_read(PROFILING_TOTAL_TIME)
	
	rhs = rhs_device.copy_to_host()
	u = u_device.copy_to_host()
	
	verified = verify()
	n3 = 1.0 * grid_points[0] * grid_points[1] * grid_points[2]
	navg = (grid_points[0] + grid_points[1] + grid_points[2]) / 3.0
	mflops = 0.0
	if tmax != 0.0:
		mflops = ( 1.0e-6 * niter *
			(3478.8*n3 - 17655.7*(navg*navg) + 28023.7*navg)
			/ tmax )
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_ADD)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-add", THREADS_PER_BLOCK_ON_ADD, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-1", THREADS_PER_BLOCK_ON_RHS_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-2", THREADS_PER_BLOCK_ON_RHS_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-3", THREADS_PER_BLOCK_ON_RHS_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_4)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-4", THREADS_PER_BLOCK_ON_RHS_4, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_5)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-5", THREADS_PER_BLOCK_ON_RHS_5, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_6)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-6", THREADS_PER_BLOCK_ON_RHS_6, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_7)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-7", THREADS_PER_BLOCK_ON_RHS_7, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_8)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-8", THREADS_PER_BLOCK_ON_RHS_8, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_9)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-9", THREADS_PER_BLOCK_ON_RHS_9, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_X_SOLVE_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-x-solve-1", THREADS_PER_BLOCK_ON_X_SOLVE_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_X_SOLVE_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-x-solve-2", THREADS_PER_BLOCK_ON_X_SOLVE_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_X_SOLVE_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-x-solve-3", THREADS_PER_BLOCK_ON_X_SOLVE_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Y_SOLVE_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-y-solve-1", THREADS_PER_BLOCK_ON_Y_SOLVE_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Y_SOLVE_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-y-solve-2", THREADS_PER_BLOCK_ON_Y_SOLVE_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Y_SOLVE_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-y-solve-3", THREADS_PER_BLOCK_ON_Y_SOLVE_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Z_SOLVE_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-z-solve-1", THREADS_PER_BLOCK_ON_Z_SOLVE_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Z_SOLVE_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-z-solve-2", THREADS_PER_BLOCK_ON_Z_SOLVE_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_Z_SOLVE_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-z-solve-3", THREADS_PER_BLOCK_ON_Z_SOLVE_3, t1, (t1 * 100 / tt))
		# OLD
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-exact-rhs-1", THREADS_PER_BLOCK_ON_EXACT_RHS_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-exact-rhs-2", THREADS_PER_BLOCK_ON_EXACT_RHS_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-exact-rhs-3", THREADS_PER_BLOCK_ON_EXACT_RHS_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EXACT_RHS_4)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-exact-rhs-4", THREADS_PER_BLOCK_ON_EXACT_RHS_4, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERROR_NORM_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-error-norm-1", THREADS_PER_BLOCK_ON_ERROR_NORM_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_ERROR_NORM_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-error-norm-2", THREADS_PER_BLOCK_ON_ERROR_NORM_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_INITIALIZE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-initialize", THREADS_PER_BLOCK_ON_INITIALIZE, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_NORM_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-norm-1", THREADS_PER_BLOCK_ON_RHS_NORM_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RHS_NORM_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" bt-rhs-norm-2", THREADS_PER_BLOCK_ON_RHS_NORM_2, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" bt-add", THREADS_PER_BLOCK_ON_ADD)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-1", THREADS_PER_BLOCK_ON_RHS_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-2", THREADS_PER_BLOCK_ON_RHS_2)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-3", THREADS_PER_BLOCK_ON_RHS_3)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-4", THREADS_PER_BLOCK_ON_RHS_4)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-5", THREADS_PER_BLOCK_ON_RHS_5)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-6", THREADS_PER_BLOCK_ON_RHS_6)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-7", THREADS_PER_BLOCK_ON_RHS_7)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-8", THREADS_PER_BLOCK_ON_RHS_8)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-9", THREADS_PER_BLOCK_ON_RHS_9)
		gpu_config_string += "%29s\t%25d\n" % (" bt-x-solve-1", THREADS_PER_BLOCK_ON_X_SOLVE_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-x-solve-2", THREADS_PER_BLOCK_ON_X_SOLVE_2)
		gpu_config_string += "%29s\t%25d\n" % (" bt-x-solve-3", THREADS_PER_BLOCK_ON_X_SOLVE_3)
		gpu_config_string += "%29s\t%25d\n" % (" bt-y-solve-1", THREADS_PER_BLOCK_ON_Y_SOLVE_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-y-solve-2", THREADS_PER_BLOCK_ON_Y_SOLVE_2)
		gpu_config_string += "%29s\t%25d\n" % (" bt-y-solve-3", THREADS_PER_BLOCK_ON_Y_SOLVE_3)
		gpu_config_string += "%29s\t%25d\n" % (" bt-z-solve-1", THREADS_PER_BLOCK_ON_Z_SOLVE_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-z-solve-2", THREADS_PER_BLOCK_ON_Z_SOLVE_2)
		gpu_config_string += "%29s\t%25d\n" % (" bt-z-solve-3", THREADS_PER_BLOCK_ON_Z_SOLVE_3)
		# OLD
		gpu_config_string += "%29s\t%25d\n" % (" bt-exact-rhs-1", THREADS_PER_BLOCK_ON_EXACT_RHS_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-exact-rhs-2", THREADS_PER_BLOCK_ON_EXACT_RHS_2)
		gpu_config_string += "%29s\t%25d\n" % (" bt-exact-rhs-3", THREADS_PER_BLOCK_ON_EXACT_RHS_3)
		gpu_config_string += "%29s\t%25d\n" % (" bt-exact-rhs-4", THREADS_PER_BLOCK_ON_EXACT_RHS_4)
		gpu_config_string += "%29s\t%25d\n" % (" bt-error-norm-1", THREADS_PER_BLOCK_ON_ERROR_NORM_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-error-norm-2", THREADS_PER_BLOCK_ON_ERROR_NORM_2)
		gpu_config_string += "%29s\t%25d\n" % (" bt-initialize", THREADS_PER_BLOCK_ON_INITIALIZE)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-norm-1", THREADS_PER_BLOCK_ON_RHS_NORM_1)
		gpu_config_string += "%29s\t%25d\n" % (" bt-rhs-norm-2", THREADS_PER_BLOCK_ON_RHS_NORM_2)
	
	c_print_results.c_print_results("BT",
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
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA BT')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_bt_info(args.CLASS)
	set_global_variables()
	
	main()
