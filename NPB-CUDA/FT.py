# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	D. Bailey
#	W. Saphir
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
import c_randdp
import c_timers
import c_print_results

sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')))
import gpu_config

# ---------------------------------------------------------------------
# u0, u1, u2 are the main arrays in the problem. 
# depending on the decomposition, these arrays will have different 
# dimensions. to accomodate all possibilities, we allocate them as 
# one-dimensional arrays and pass them to subroutines for different 
# views
# - u0 contains the initial (transformed) initial condition
# - u1 and u2 are working arrays
# - twiddle contains exponents for the time evolution operator. 
# ---------------------------------------------------------------------
# large arrays are in common so that they are allocated on the
# heap rather than the stack. this common block is not
# referenced directly anywhere else. padding is to avoid accidental 
# cache problems, since all array sizes are powers of two.
# ---------------------------------------------------------------------
# we need a bunch of logic to keep track of how
# arrays are laid out. 
#
# note: this serial version is the derived from the parallel 0D case
# of the ft NPB.
# the computation proceeds logically as
#
# set up initial conditions
# fftx(1)
# transpose (1->2)
# ffty(2)
# transpose (2->3)
# fftz(3)
# time evolution
# fftz(3)
# transpose (3->2)
# ffty(2)
# transpose (2->1)
# fftx(1)
# compute residual(1)
# 
# for the 0D, 1D, 2D strategies, the layouts look like xxx
#
#            0D        1D        2D
# 1:        xyz       xyz       xyz
# 2:        xyz       xyz       yxz
# 3:        xyz       zyx       zxy
# the array dimensions are stored in dims(coord, phase)
# ---------------------------------------------------------------------
# if processor array is 1x1 -> 0D grid decomposition
# 
# cache blocking params. these values are good for most
# RISC processors.  
# FFT parameters:
# fftblock controls how many ffts are done at a time. 
# the default is appropriate for most cache-based machines
# on vector machines, the FFT can be vectorized with vector
# length equal to the block size, so the block size should
# be as large as possible. this is the size of the smallest
# dimension of the problem: 128 for class A, 256 for class B
# and 512 for class C.
# ---------------------------------------------------------------------


# Global variables
#FFTBLOCK_DEFAULT = 0
#FFTBLOCKPAD_DEFAULT = 0
#FFTBLOCK = 0
#FFTBLOCKPAD = 0
SEED = 314159265.0
A = 1220703125.0
PI = 3.141592653589793238
ALPHA = 1.0e-6
AP = -4.0 * ALPHA * PI * PI
PROFILING_TOTAL_TIME = 0
PROFILING_INDEXMAP = 1
PROFILING_INITIAL_CONDITIONS = 2
PROFILING_INIT_UI = 3
PROFILING_EVOLVE = 4
PROFILING_FFTX_1 = 5
PROFILING_FFTX_2 = 6
PROFILING_FFTX_3 = 7
PROFILING_FFTY_1 = 8
PROFILING_FFTY_2 = 9
PROFILING_FFTY_3 = 10
PROFILING_FFTZ_1 = 11
PROFILING_FFTZ_2 = 12
PROFILING_FFTZ_3 = 13
PROFILING_CHECKSUM = 14
PROFILING_INIT = 15
CHECKSUM_TASKS = 1024

NX = 0
NY = 0
NZ = 0

sums = None
#twiddle = None
u = None
#u0 = None
#u1 = None

niter = 0

# GPU variables
starts_device = None
twiddle_device = None
sums_device = None
u_device = None
u0_device = None
u1_device = None
y0_device = None
y1_device = None

blocks_per_grid_on_compute_indexmap = 0
blocks_per_grid_on_compute_initial_conditions = 0
blocks_per_grid_on_init_ui = 0
blocks_per_grid_on_evolve = 0
blocks_per_grid_on_fftx_1 = 0
blocks_per_grid_on_fftx_2 = 0
blocks_per_grid_on_fftx_3 = 0
blocks_per_grid_on_ffty_1 = 0
blocks_per_grid_on_ffty_2 = 0
blocks_per_grid_on_ffty_3 = 0
blocks_per_grid_on_fftz_1 = 0
blocks_per_grid_on_fftz_2 = 0
blocks_per_grid_on_fftz_3 = 0
blocks_per_grid_on_checksum = 0

threads_per_block_on_compute_indexmap = 0
threads_per_block_on_compute_initial_conditions = 0
threads_per_block_on_init_ui = 0
threads_per_block_on_evolve = 0
threads_per_block_on_fftx_1 = 0
threads_per_block_on_fftx_2 = 0
threads_per_block_on_fftx_3 = 0
threads_per_block_on_ffty_1 = 0
threads_per_block_on_ffty_2 = 0
threads_per_block_on_ffty_3 = 0
threads_per_block_on_fftz_1 = 0
threads_per_block_on_fftz_2 = 0
threads_per_block_on_fftz_3 = 0
threads_per_block_on_checksum = 0

stream = 0
size_shared_data = 0

gpu_device_id = 0
total_devices = 0
device_prop = None


def set_global_variables():
	#global FFTBLOCK_DEFAULT, FFTBLOCKPAD_DEFAULT, FFTBLOCK, FFTBLOCKPAD
	global NX, NY, NZ
	global sums, u
	#glogal twiddle, u0, u1
	
	#FFTBLOCK_DEFAULT = npbparams.DEFAULT_BEHAVIOR
	#FFTBLOCKPAD_DEFAULT = npbparams.DEFAULT_BEHAVIOR
	#FFTBLOCK = FFTBLOCK_DEFAULT
	#FFTBLOCKPAD = FFTBLOCKPAD_DEFAULT
	
	NX = npbparams.NX
	NY = npbparams.NY
	NZ = npbparams.NZ
	
	sums = numpy.empty(npbparams.NITER_DEFAULT+1, dtype=numpy.complex128)
	#twiddle = numpy.repeat(0.0, npbparams.NTOTAL)
	u = numpy.empty(npbparams.MAXDIM, dtype=numpy.complex128)
	#u0 = numpy.repeat(complex(0.0, 0.0), npbparams.NTOTAL)
	#u1 = numpy.repeat(complex(0.0, 0.0), npbparams.NTOTAL)
#END set_global_variables()


def verify(d1,
		d2,
		d3,
		nt):
	# ---------------------------------------------------------------------
	# reference checksums
	# ---------------------------------------------------------------------
	csum_ref = numpy.empty(nt+1, dtype=numpy.complex128)

	epsilon = 1.0e-12

	if npbparams.CLASS == 'S':
		# ---------------------------------------------------------------------
		# sample size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1] = complex(5.546087004964E+02, 4.845363331978E+02)
		csum_ref[2] = complex(5.546385409189E+02, 4.865304269511E+02)
		csum_ref[3] = complex(5.546148406171E+02, 4.883910722336E+02)
		csum_ref[4] = complex(5.545423607415E+02, 4.901273169046E+02)
		csum_ref[5] = complex(5.544255039624E+02, 4.917475857993E+02)
		csum_ref[6] = complex(5.542683411902E+02, 4.932597244941E+02)
	elif npbparams.CLASS == 'W':
		# ---------------------------------------------------------------------
		# class_npb W size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1] = complex(5.673612178944E+02, 5.293246849175E+02)
		csum_ref[2] = complex(5.631436885271E+02, 5.282149986629E+02)
		csum_ref[3] = complex(5.594024089970E+02, 5.270996558037E+02)
		csum_ref[4] = complex(5.560698047020E+02, 5.260027904925E+02)
		csum_ref[5] = complex(5.530898991250E+02, 5.249400845633E+02)
		csum_ref[6] = complex(5.504159734538E+02, 5.239212247086E+02)
	elif npbparams.CLASS == 'A':
		# ---------------------------------------------------------------------
		# class_npb A size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1] = complex(5.046735008193E+02, 5.114047905510E+02)
		csum_ref[2] = complex(5.059412319734E+02, 5.098809666433E+02)
		csum_ref[3] = complex(5.069376896287E+02, 5.098144042213E+02)
		csum_ref[4] = complex(5.077892868474E+02, 5.101336130759E+02)
		csum_ref[5] = complex(5.085233095391E+02, 5.104914655194E+02)
		csum_ref[6] = complex(5.091487099959E+02, 5.107917842803E+02)
	elif npbparams.CLASS == 'B':
		# --------------------------------------------------------------------
		# class_npb B size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1]  = complex(5.177643571579E+02, 5.077803458597E+02)
		csum_ref[2]  = complex(5.154521291263E+02, 5.088249431599E+02)
		csum_ref[3]  = complex(5.146409228649E+02, 5.096208912659E+02)
		csum_ref[4]  = complex(5.142378756213E+02, 5.101023387619E+02)
		csum_ref[5]  = complex(5.139626667737E+02, 5.103976610617E+02)
		csum_ref[6]  = complex(5.137423460082E+02, 5.105948019802E+02)
		csum_ref[7]  = complex(5.135547056878E+02, 5.107404165783E+02)
		csum_ref[8]  = complex(5.133910925466E+02, 5.108576573661E+02)
		csum_ref[9]  = complex(5.132470705390E+02, 5.109577278523E+02)
		csum_ref[10] = complex(5.131197729984E+02, 5.110460304483E+02)
		csum_ref[11] = complex(5.130070319283E+02, 5.111252433800E+02)
		csum_ref[12] = complex(5.129070537032E+02, 5.111968077718E+02)
		csum_ref[13] = complex(5.128182883502E+02, 5.112616233064E+02)
		csum_ref[14] = complex(5.127393733383E+02, 5.113203605551E+02)
		csum_ref[15] = complex(5.126691062020E+02, 5.113735928093E+02)
		csum_ref[16] = complex(5.126064276004E+02, 5.114218460548E+02)
		csum_ref[17] = complex(5.125504076570E+02, 5.114656139760E+02)
		csum_ref[18] = complex(5.125002331720E+02, 5.115053595966E+02)
		csum_ref[19] = complex(5.124551951846E+02, 5.115415130407E+02)
		csum_ref[20] = complex(5.124146770029E+02, 5.115744692211E+02)
	elif npbparams.CLASS == 'C':
		# ---------------------------------------------------------------------
		# class_npb C size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1]  = complex(5.195078707457E+02, 5.149019699238E+02)
		csum_ref[2]  = complex(5.155422171134E+02, 5.127578201997E+02)
		csum_ref[3]  = complex(5.144678022222E+02, 5.122251847514E+02)
		csum_ref[4]  = complex(5.140150594328E+02, 5.121090289018E+02)
		csum_ref[5]  = complex(5.137550426810E+02, 5.121143685824E+02)
		csum_ref[6]  = complex(5.135811056728E+02, 5.121496764568E+02)
		csum_ref[7]  = complex(5.134569343165E+02, 5.121870921893E+02)
		csum_ref[8]  = complex(5.133651975661E+02, 5.122193250322E+02)
		csum_ref[9]  = complex(5.132955192805E+02, 5.122454735794E+02)
		csum_ref[10] = complex(5.132410471738E+02, 5.122663649603E+02)
		csum_ref[11] = complex(5.131971141679E+02, 5.122830879827E+02)
		csum_ref[12] = complex(5.131605205716E+02, 5.122965869718E+02)
		csum_ref[13] = complex(5.131290734194E+02, 5.123075927445E+02)
		csum_ref[14] = complex(5.131012720314E+02, 5.123166486553E+02)
		csum_ref[15] = complex(5.130760908195E+02, 5.123241541685E+02)
		csum_ref[16] = complex(5.130528295923E+02, 5.123304037599E+02)
		csum_ref[17] = complex(5.130310107773E+02, 5.123356167976E+02)
		csum_ref[18] = complex(5.130103090133E+02, 5.123399592211E+02)
		csum_ref[19] = complex(5.129905029333E+02, 5.123435588985E+02)
		csum_ref[20] = complex(5.129714421109E+02, 5.123465164008E+02)
	elif npbparams.CLASS == 'D':
		# ---------------------------------------------------------------------
		# class_npb D size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1]  = complex(5.122230065252E+02, 5.118534037109E+02)
		csum_ref[2]  = complex(5.120463975765E+02, 5.117061181082E+02)
		csum_ref[3]  = complex(5.119865766760E+02, 5.117096364601E+02)
		csum_ref[4]  = complex(5.119518799488E+02, 5.117373863950E+02)
		csum_ref[5]  = complex(5.119269088223E+02, 5.117680347632E+02)
		csum_ref[6]  = complex(5.119082416858E+02, 5.117967875532E+02)
		csum_ref[7]  = complex(5.118943814638E+02, 5.118225281841E+02)
		csum_ref[8]  = complex(5.118842385057E+02, 5.118451629348E+02)
		csum_ref[9]  = complex(5.118769435632E+02, 5.118649119387E+02)
		csum_ref[10] = complex(5.118718203448E+02, 5.118820803844E+02)
		csum_ref[11] = complex(5.118683569061E+02, 5.118969781011E+02)
		csum_ref[12] = complex(5.118661708593E+02, 5.119098918835E+02)
		csum_ref[13] = complex(5.118649768950E+02, 5.119210777066E+02)
		csum_ref[14] = complex(5.118645605626E+02, 5.119307604484E+02)
		csum_ref[15] = complex(5.118647586618E+02, 5.119391362671E+02)
		csum_ref[16] = complex(5.118654451572E+02, 5.119463757241E+02)
		csum_ref[17] = complex(5.118665212451E+02, 5.119526269238E+02)
		csum_ref[18] = complex(5.118679083821E+02, 5.119580184108E+02)
		csum_ref[19] = complex(5.118695433664E+02, 5.119626617538E+02)
		csum_ref[20] = complex(5.118713748264E+02, 5.119666538138E+02)
		csum_ref[21] = complex(5.118733606701E+02, 5.119700787219E+02)
		csum_ref[22] = complex(5.118754661974E+02, 5.119730095953E+02)
		csum_ref[23] = complex(5.118776626738E+02, 5.119755100241E+02)
		csum_ref[24] = complex(5.118799262314E+02, 5.119776353561E+02)
		csum_ref[25] = complex(5.118822370068E+02, 5.119794338060E+02)
	elif npbparams.CLASS == 'E':
		# ---------------------------------------------------------------------
		# class_npb E size reference checksums
		# ---------------------------------------------------------------------
		csum_ref[1]  = complex(5.121601045346E+02, 5.117395998266E+02)
		csum_ref[2]  = complex(5.120905403678E+02, 5.118614716182E+02)
		csum_ref[3]  = complex(5.120623229306E+02, 5.119074203747E+02)
		csum_ref[4]  = complex(5.120438418997E+02, 5.119345900733E+02)
		csum_ref[5]  = complex(5.120311521872E+02, 5.119551325550E+02)
		csum_ref[6]  = complex(5.120226088809E+02, 5.119720179919E+02)
		csum_ref[7]  = complex(5.120169296534E+02, 5.119861371665E+02)
		csum_ref[8]  = complex(5.120131225172E+02, 5.119979364402E+02)
		csum_ref[9]  = complex(5.120104767108E+02, 5.120077674092E+02)
		csum_ref[10] = complex(5.120085127969E+02, 5.120159443121E+02)
		csum_ref[11] = complex(5.120069224127E+02, 5.120227453670E+02)
		csum_ref[12] = complex(5.120055158164E+02, 5.120284096041E+02)
		csum_ref[13] = complex(5.120041820159E+02, 5.120331373793E+02)
		csum_ref[14] = complex(5.120028605402E+02, 5.120370938679E+02)
		csum_ref[15] = complex(5.120015223011E+02, 5.120404138831E+02)
		csum_ref[16] = complex(5.120001570022E+02, 5.120432068837E+02)
		csum_ref[17] = complex(5.119987650555E+02, 5.120455615860E+02)
		csum_ref[18] = complex(5.119973525091E+02, 5.120475499442E+02)
		csum_ref[19] = complex(5.119959279472E+02, 5.120492304629E+02)
		csum_ref[20] = complex(5.119945006558E+02, 5.120506508902E+02)
		csum_ref[21] = complex(5.119930795911E+02, 5.120518503782E+02)
		csum_ref[22] = complex(5.119916728462E+02, 5.120528612016E+02)
		csum_ref[23] = complex(5.119902874185E+02, 5.120537101195E+02)
		csum_ref[24] = complex(5.119889291565E+02, 5.120544194514E+02)
		csum_ref[25] = complex(5.119876028049E+02, 5.120550079284E+02)

	verified = True
	for i in range(1, nt+1):
		err = abs((sums[i] - csum_ref[i]) / csum_ref[i])
		if not (err <= epsilon):
			verified = False
			break

	if verified:
		print(" Result verification successful")
	else:
		print(" Result verification failed")

	print(" class_npb = %c" % (npbparams.CLASS))
	
	return verified
#END verify()


def setup(): 
	global niter

	niter = npbparams.NITER_DEFAULT

	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - FT Benchmark\n")
	print(" Size                : %4dx%4dx%4d" % (NX, NY, NZ))
	print(" Iterations                  :%7d" % (niter))
	print()
#END setup()


def setup_gpu():
	global gpu_device_id, total_devices
	global device_prop
	
	global threads_per_block_on_compute_indexmap, threads_per_block_on_compute_initial_conditions, threads_per_block_on_init_ui
	global threads_per_block_on_evolve, threads_per_block_on_checksum
	global threads_per_block_on_fftx_1, threads_per_block_on_fftx_2, threads_per_block_on_fftx_3
	global threads_per_block_on_ffty_1, threads_per_block_on_ffty_2, threads_per_block_on_ffty_3
	global threads_per_block_on_fftz_1, threads_per_block_on_fftz_2, threads_per_block_on_fftz_3
	
	global blocks_per_grid_on_compute_indexmap, blocks_per_grid_on_compute_initial_conditions, blocks_per_grid_on_init_ui
	global blocks_per_grid_on_evolve, blocks_per_grid_on_checksum
	global blocks_per_grid_on_fftx_1, blocks_per_grid_on_fftx_2, blocks_per_grid_on_fftx_3
	global blocks_per_grid_on_ffty_1, blocks_per_grid_on_ffty_2, blocks_per_grid_on_ffty_3
	global blocks_per_grid_on_fftz_1, blocks_per_grid_on_fftz_2, blocks_per_grid_on_fftz_3
		
	global sums_device, starts_device, twiddle_device
	global u_device, u0_device, u1_device, y0_device, y1_device
	
	global size_shared_data
	
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
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_COMPUTE_INDEXMAP
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_compute_indexmap = aux_threads_per_block
	else: 
		threads_per_block_on_compute_indexmap = device_prop.WARP_SIZE
	
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_COMPUTE_INITIAL_CONDITIONS
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_compute_initial_conditions = aux_threads_per_block
	else: 
		threads_per_block_on_compute_initial_conditions = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_INIT_UI
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_init_ui = aux_threads_per_block
	else: 
		threads_per_block_on_init_ui = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_EVOLVE
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_evolve = aux_threads_per_block
	else: 
		threads_per_block_on_evolve = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTX_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_fftx_1 = aux_threads_per_block
	else: 
		threads_per_block_on_fftx_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTX_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_fftx_2 = aux_threads_per_block
	else: 
		threads_per_block_on_fftx_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTX_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_fftx_3 = aux_threads_per_block
	else: 
		threads_per_block_on_fftx_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTY_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_ffty_1 = aux_threads_per_block
	else: 
		threads_per_block_on_ffty_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTY_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_ffty_2 = aux_threads_per_block
	else: 
		threads_per_block_on_ffty_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTY_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_ffty_3 = aux_threads_per_block
	else: 
		threads_per_block_on_ffty_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTZ_1
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_fftz_1 = aux_threads_per_block
	else: 
		threads_per_block_on_fftz_1 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTZ_2
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_fftz_2 = aux_threads_per_block
	else: 
		threads_per_block_on_fftz_2 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_FFTZ_3
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_fftz_3 = aux_threads_per_block
	else: 
		threads_per_block_on_fftz_3 = device_prop.WARP_SIZE
		
	aux_threads_per_block = gpu_config.FT_THREADS_PER_BLOCK_ON_CHECKSUM
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_checksum = aux_threads_per_block
	else: 
		threads_per_block_on_checksum = device_prop.WARP_SIZE
		
	m_ceil = math.ceil
	blocks_per_grid_on_compute_indexmap = m_ceil(npbparams.NTOTAL / threads_per_block_on_compute_indexmap)
	blocks_per_grid_on_compute_initial_conditions = m_ceil(NZ / threads_per_block_on_compute_initial_conditions)
	blocks_per_grid_on_init_ui = m_ceil(npbparams.NTOTAL / threads_per_block_on_init_ui)
	blocks_per_grid_on_evolve = m_ceil(npbparams.NTOTAL / threads_per_block_on_evolve)
	blocks_per_grid_on_fftx_1 = m_ceil((NX*NY*NZ) / threads_per_block_on_fftx_1)
	blocks_per_grid_on_fftx_2 = m_ceil((NY*NZ) / threads_per_block_on_fftx_2)
	blocks_per_grid_on_fftx_3 = m_ceil((NX*NY*NZ) / threads_per_block_on_fftx_3)
	blocks_per_grid_on_ffty_1 = m_ceil((NX*NY*NZ) / threads_per_block_on_ffty_1)
	blocks_per_grid_on_ffty_2 = m_ceil((NX*NZ) / threads_per_block_on_ffty_2)
	blocks_per_grid_on_ffty_3 = m_ceil((NX*NY*NZ) / threads_per_block_on_ffty_3)
	blocks_per_grid_on_fftz_1 = m_ceil((NX*NY*NZ) / threads_per_block_on_fftz_1)
	blocks_per_grid_on_fftz_2 = m_ceil((NX*NY) / threads_per_block_on_fftz_2)
	blocks_per_grid_on_fftz_3 = m_ceil((NX*NY*NZ) / threads_per_block_on_fftz_3)
	blocks_per_grid_on_checksum = m_ceil(CHECKSUM_TASKS / threads_per_block_on_checksum)
	
	n_float64 = numpy.float64
	n_complex = numpy.complex128
	sums_device = cuda.device_array((npbparams.NITER_DEFAULT + 1)*2, n_float64)
	starts_device = cuda.device_array(NZ, n_float64)
	twiddle_device = cuda.device_array(npbparams.NTOTAL, n_float64)
	u_device = cuda.device_array(npbparams.MAXDIM, n_complex)
	u0_device = cuda.device_array(npbparams.NTOTAL, n_complex)
	u1_device = cuda.device_array(npbparams.NTOTAL, n_complex)
	y0_device = cuda.device_array(npbparams.NTOTAL, n_complex)
	y1_device = cuda.device_array(npbparams.NTOTAL, n_complex)
	
	size_shared_data = threads_per_block_on_checksum * u1_device.dtype.itemsize
#END setup_gpu()


#*****************************************************************
#************************* CPU FUNCTIONS *************************
#*****************************************************************
@njit
def ilog2(n):
	if n == 1: 
		return 0
	lg = 1
	nn = 2
	while nn < n:
		nn *= 2
		lg += 1
	return lg
#END ilog2() 


@njit
def ipow46(a,
		exponent):
	# --------------------------------------------------------------------
	# use
	# a^n = a^(n/2)*a^(n/2) if n even else
	# a^n = a*a^(n-1)       if n odd
	# -------------------------------------------------------------------
	result = 1
	if exponent == 0:
		return result
	q = a
	r = 1.0
	n = exponent

	while n > 1:
		n2 = int(n / 2)
		if (n2 * 2) == n:
			aux, q = randlc(q, q)
			n = n2
		else: 
			aux, r = randlc(r, q)
			n = n - 1
	
	aux, r = randlc(r, q)
	result = r
	
	return result
#END ipow46()


#*****************************************************************
#************************* GPU FUNCTIONS *************************
#*****************************************************************
@cuda.jit('void(int32, complex128[:], float64[:], int32, int32, int32, int32)')
def checksum_gpu_kernel(iteration, 
						u1, 
						sums,
						NX_aux, NY_aux, NZ_aux, NTOTAL_aux):
	#dcomplex* share_sums = (dcomplex*)(extern_share_data);
	share_sums = cuda.shared.array(shape=0, dtype=numba.complex128)
	
	j = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) + 1
	
	if j <= CHECKSUM_TASKS:
		q = j % NX_aux
		r = 3*j % NY_aux
		s = 5*j % NZ_aux
		share_sums[cuda.threadIdx.x] = u1[ q + r*NX_aux + s*NX_aux*NY_aux ]
	else:
		share_sums[cuda.threadIdx.x] = complex(0.0, 0.0)

	cuda.syncthreads()
	
	i = int(cuda.blockDim.x / 2)
	for aux in range(1000):
		if cuda.threadIdx.x < i:
			share_sums[cuda.threadIdx.x] = share_sums[cuda.threadIdx.x] + share_sums[cuda.threadIdx.x+i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break
	
	if cuda.threadIdx.x == 0:
		share_sums[0] = complex(share_sums[0].real / NTOTAL_aux, share_sums[0].imag / NTOTAL_aux)
		idx = iteration * 2
		cuda.atomic.add(sums, idx, share_sums[0].real) #Even = real
		cuda.atomic.add(sums, idx+1, share_sums[0].imag) #Odd = imag
#END checksum_gpu_kernel()


def checksum_gpu(iteration,
				u1,
				sums_device):
	global sums
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_CHECKSUM)

	checksum_gpu_kernel[blocks_per_grid_on_checksum,
		threads_per_block_on_checksum,
		stream,
		size_shared_data](iteration, 
						u1, 
						sums_device,
						NX, NY, NZ, npbparams.NTOTAL)
		
	local_sums = sums_device.copy_to_host()
	idx = iteration * 2  #Even = real, Odd = imag
	sums[iteration] = complex(local_sums[idx], local_sums[idx+1])

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_CHECKSUM)
#END checksum_gpu()


@cuda.jit('void(complex128[:], complex128[:], float64[:], int32, int32, int32)')
def evolve_gpu_kernel(u0, 
					u1,
					twiddle,
					NX_aux, NY_aux, NZ_aux):
	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if thread_id >= (NZ_aux * NY_aux * NX_aux):
		return

	u0[thread_id] = u0[thread_id] * twiddle[thread_id]
	u1[thread_id] = u0[thread_id]
#END evolve_gpu_kernel()


def evolve_gpu(u0, 
			u1,
			twiddle):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_EVOLVE)

	evolve_gpu_kernel[blocks_per_grid_on_evolve,
		threads_per_block_on_evolve](u0, 
									u1,
									twiddle,
									NX, NY, NZ)
	cuda.synchronize()
	
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_EVOLVE)
#END evolve_gpu()


# ----------------------------------------------------------------------
# x_out[z][y][x] = y0[z][y][x]
#
# x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY] 
# ----------------------------------------------------------------------
@cuda.jit('void(complex128[:], complex128[:], int32, int32, int32)')
def cffts3_gpu_kernel_3(x_out,
						y0,
						NX_aux, NY_aux, NZ_aux):
	x_y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y_z >= (NX_aux * NY_aux * NZ_aux):
		return
	x_out[x_y_z] = y0[x_y_z]
#END cffts3_gpu_kernel_3()


# ----------------------------------------------------------------------
# pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
#
# index_arg = i + j*NX
#
# size_arg = NX*NY
# ----------------------------------------------------------------------
@cuda.jit('void(int32, int32, int32, int32, complex128[:], complex128[:], complex128[:], int32, int32)', device=True)
def cffts3_gpu_fftz2_device(iss, 
							l, 
							m, 
							n, 
							u, 
							x, 
							y, 
							index_arg, 
							size_arg):
	# ---------------------------------------------------------------------
	# set initial parameters.
	# ---------------------------------------------------------------------
	n1 = int(n / 2)
	lk = 1 << (l - 1)
	li = 1 << (m - l)
	lj = 2 * lk
	ku = li
	for i in range(li):
		i11 = i * lk
		i12 = i11 + n1
		i21 = i * lj
		i22 = i21 + lk
		u1 = complex(0.0, 0.0)
		if iss >= 1:
			u1 = u[ku+i]
		else:
			u1 = u[ku+i].conjugate()

		for k in range(lk):
			x11 = x[(i11+k)*size_arg+index_arg]
			x21 = x[(i12+k)*size_arg+index_arg]
			y[(i21+k)*size_arg+index_arg] = x11 + x21
			y[(i22+k)*size_arg+index_arg] = u1 * (x11 - x21)
#END cffts3_gpu_fftz2_device()


# ----------------------------------------------------------------------
# pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
#
# index_arg = i + j*NX
#
# size_arg = NX*NY
# ----------------------------------------------------------------------
@cuda.jit('void(int32, int32, int32, complex128[:], complex128[:], complex128[:], int32, int32)', device=True)
def cffts3_gpu_cfftz_device(iss, 
							m, 
							n, 
							x, 
							y, 
							u_device, 
							index_arg, 
							size_arg):
	# ---------------------------------------------------------------------
	# perform one variant of the Stockham FFT.
	# ---------------------------------------------------------------------
	for l in range(1, m+1, 2):
		cffts3_gpu_fftz2_device(iss, l, m, n, u_device, x, y, index_arg, size_arg)
		if l == m:
			break
		cffts3_gpu_fftz2_device(iss, l + 1, m, n, u_device, y, x, index_arg, size_arg)

	# ---------------------------------------------------------------------
	# copy Y to X.
	# ---------------------------------------------------------------------
	if (m % 2) == 1:
		for j in range(n):
			x[j*size_arg+index_arg] = y[j*size_arg+index_arg]
#END cffts3_gpu_cfftz_device()


# ----------------------------------------------------------------------
# pattern = i + j*NX + variable*NX*NY | variable is z and transforms z axis
# ----------------------------------------------------------------------
@cuda.jit('void(int32, complex128[:], complex128[:], complex128[:], int32, int32, int32, int32)')
def cffts3_gpu_kernel_2(iss, 
						gty1, 
						gty2, 
						u_device,
						NX_aux, NY_aux, NZ_aux, logd3_aux):
	x_y = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y >= (NX_aux * NY_aux):
		return
	cffts3_gpu_cfftz_device(iss, 
			logd3_aux, 
			NZ_aux, 
			gty1 , 
			gty2, 
			u_device, 
			x_y, # index_arg
			NX_aux * NY_aux) # size_arg
#END cffts3_gpu_kernel_2()


# ----------------------------------------------------------------------
# y0[z][y][x] = x_in[z][y][x] 
#
# y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY] 
# ----------------------------------------------------------------------
@cuda.jit('void(complex128[:], complex128[:], int32, int32, int32)')
def cffts3_gpu_kernel_1(x_in, 
						y0,
						NX_aux, NY_aux, NZ_aux):
	x_y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y_z >= (NX_aux * NY_aux * NZ_aux):
		return
	y0[x_y_z] = x_in[x_y_z]
#END cffts3_gpu_kernel_1()


def cffts3_gpu(iss, 
			u, 
			x_in, 
			x_out, 
			y0, 
			y1):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTZ_1)
	cffts3_gpu_kernel_1[blocks_per_grid_on_fftz_1,
		threads_per_block_on_fftz_1](x_in, 
									y0,
									NX, NY, NZ)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTZ_1)
		
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTZ_2)
	logd3 = ilog2(NZ)
	cffts3_gpu_kernel_2[blocks_per_grid_on_fftz_2,
		threads_per_block_on_fftz_2](iss, 
									y0, 
									y1, 
									u,
									NX, NY, NZ, logd3)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTZ_2)
		
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTZ_3)
	cffts3_gpu_kernel_3[blocks_per_grid_on_fftz_3,
		threads_per_block_on_fftz_3](x_out, 
									y0,
									NX, NY, NZ)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTZ_3)
#END cffts3_gpu()


# ----------------------------------------------------------------------
# x_out[z][y][x] = y0[z][y][x]
#
# x_out[x + y*NX + z*NX*NY] = y0[x + y*NX + z*NX*NY] 
# ----------------------------------------------------------------------
@cuda.jit('void(complex128[:], complex128[:], int32, int32, int32)')
def cffts2_gpu_kernel_3(x_out, 
						y0,
						NX_aux, NY_aux, NZ_aux):
	x_y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y_z >= (NX_aux * NY_aux * NZ_aux):
		return
	x_out[x_y_z] = y0[x_y_z]
#END cffts2_gpu_kernel_3()


# ----------------------------------------------------------------------
# pattern = i + variable*NX + k*NX*NY | variable is j and transforms y axis
# ----------------------------------------------------------------------
@cuda.jit('void(int32, complex128[:], complex128[:], complex128[:], int32, int32, int32, int32)')
def cffts2_gpu_kernel_2(iss, 
						gty1, 
						gty2, 
						u_device,
						NX_aux, NY_aux, NZ_aux, logd2_aux):
	x_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if x_z >= (NX_aux * NZ_aux):
		return
	
	i = x_z % NX_aux # i = x
	k = int(x_z / NX_aux) % NZ_aux # k = z

	for l in range(1, logd2_aux+1, 2):
		n1 = int(NY_aux / 2)
		lk = 1 << (l - 1)
		li = 1 << (logd2_aux - l)
		lj = 2 * lk
		ku = li
		for i1 in range(0, li):
			i11 = i1 * lk
			i12 = i11 + n1
			i21 = i1 * lj
			i22 = i21 + lk
			u1 = complex(0.0, 0.0)
			if iss >= 1:
				u1 = u_device[ku+i1]
			else:
				u1 = u_device[ku+i1].conjugate()
			
			for k1 in range(0, lk):
				# gty1[k][i11+k1][i]
				x11 = gty1[i + (i11+k1)*NX_aux + k*NX_aux*NY_aux]
				# gty1[k][i12+k1][i]
				x21 = gty1[i + (i12+k1)*NX_aux + k*NX_aux*NY_aux]
				# gty2[k][i21+k1][i]
				gty2[i + (i21+k1)*NX_aux + k*NX_aux*NY_aux] = x11 + x21
				# gty2[k][i22+k1][i]
				gty2[i + (i22+k1)*NX_aux + k*NX_aux*NY_aux] = u1 * (x11 - x21)
		#END for i1 in range(0, li)

		if l == logd2_aux:
			for j1 in range(NY_aux):
				# gty1[k][j1][i]
				gty1[i + j1*NX_aux + k*NX_aux*NY_aux] = gty2[i + j1*NX_aux + k*NX_aux*NY_aux]
		else:
			n1 = int(NY_aux / 2)
			lk = 1 << (l+1 - 1)
			li = 1 << (logd2_aux - (l+1))
			lj = 2 * lk
			ku = li
			for i1 in range(0, li):
				i11 = i1 * lk
				i12 = i11 + n1
				i21 = i1 * lj
				i22 = i21 + lk
				u2 = complex(0.0, 0.0)
				if iss >= 1:
					u2 = u_device[ku+i1]
				else:
					u2 = u_device[ku+i1].conjugate()
				
				for k1 in range(0, lk):
					# gty2[k][i11+k1][i]
					x12 = gty2[i + (i11+k1)*NX_aux + k*NX_aux*NY_aux]
					# gty2[k][i12+k1][i]
					x22 = gty2[i + (i12+k1)*NX_aux + k*NX_aux*NY_aux]
					# gty1[k][i21+k1][i]
					gty1[i + (i21+k1)*NX_aux + k*NX_aux*NY_aux] = x12 + x22
					# gty1[k][i22+k1][i]
					gty1[i + (i22+k1)*NX_aux + k*NX_aux*NY_aux] = u2 * (x12 - x22)
		#END if l == logd2_aux:
	#END for l in range(1, logd2_aux+1, 2):
#END cffts2_gpu_kernel_2()


# ----------------------------------------------------------------------
# y0[z][y][x] = x_in[z][y][x] 
#
# y0[x + y*NX + z*NX*NY]  = x_in[x + y*NX + z*NX*NY] 
# ----------------------------------------------------------------------
@cuda.jit('void(complex128[:], complex128[:], int32, int32, int32)')
def cffts2_gpu_kernel_1(x_in, 
						y0,
						NX_aux, NY_aux, NZ_aux):
	x_y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y_z >= (NX_aux*NY_aux*NZ_aux):
		return
	y0[x_y_z] = x_in[x_y_z]
#END cffts2_gpu_kernel_1()


def cffts2_gpu(iss, 
			u, 
			x_in, 
			x_out, 
			y0, 
			y1):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTY_1)
	cffts2_gpu_kernel_1[blocks_per_grid_on_ffty_1,
		threads_per_block_on_ffty_1](x_in, 
									y0,
									NX, NY, NZ)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTY_1)
		
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTY_2)
	logd2 = ilog2(NY)
	cffts2_gpu_kernel_2[blocks_per_grid_on_ffty_2,
		threads_per_block_on_ffty_2](iss, 
									y0, 
									y1, 
									u,
									NX, NY, NZ, logd2)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTY_2)
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTY_3)
	cffts2_gpu_kernel_3[blocks_per_grid_on_ffty_3,
		threads_per_block_on_ffty_3](x_out, 
									y0,
									NX, NY, NZ)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTY_3)
#END cffts2_gpu()


# ----------------------------------------------------------------------
# x_out[z][y][x] = y0[z][x][y] 
#
# x_out[x + y*NX + z*NX*NY] = y0[y + x*NY + z*NX*NY]  
# ----------------------------------------------------------------------
@cuda.jit('void(complex128[:], complex128[:], int32, int32, int32)')
def cffts1_gpu_kernel_3(x_out, 
						y0,
						NX_aux, NY_aux, NZ_aux):
	x_y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y_z >= (NX_aux * NY_aux * NZ_aux):
		return

	x = x_y_z % NX_aux
	y = int(x_y_z / NX_aux) % NY_aux
	z = int(x_y_z / (NX_aux * NY_aux))
	x_out[x_y_z] = y0[y+(x*NY_aux)+(z*NX_aux*NY_aux)]
#END cffts1_gpu_kernel_3()


# ----------------------------------------------------------------------
# pattern = j + variable*NY + k*NX*NY | variable is i and transforms x axis
# ----------------------------------------------------------------------
@cuda.jit('void(int32, complex128[:], complex128[:], complex128[:], int32, int32, int32, int32)')
def cffts1_gpu_kernel_2(iss, 
						gty1, 
						gty2, 
						u_device,
						NX_aux, NY_aux, NZ_aux, logd1_aux):
	y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if y_z >= (NY_aux * NZ_aux):
		return

	j = y_z % NY_aux # j = y
	k = int(y_z / NY_aux) % NZ_aux # k = z

	for l in range(1, logd1_aux+1, 2):
		n1 = int(NX_aux / 2)
		lk = 1 << (l - 1)
		li = 1 << (logd1_aux - l)
		lj = 2 * lk
		ku = li
		
		for i1 in range(0, li):
			i11 = i1 * lk
			i12 = i11 + n1
			i21 = i1 * lj
			i22 = i21 + lk
			u1 = complex(0.0, 0.0)
			if iss >= 1:
				u1 = u_device[ku+i1]
			else:
				u1 = u_device[ku+i1].conjugate()
			
			for k1 in range(0, lk):
				# gty1[k][i11+k1][j]
				x11 = gty1[j + (i11+k1)*NY_aux + k*NX_aux*NY_aux]
				# gty1[k][i12+k1][j]
				x21 = gty1[j + (i12+k1)*NY_aux + k*NX_aux*NY_aux]
				# gty2[k][i21+k1][j]
				gty2[j + (i21+k1)*NY_aux + k*NX_aux*NY_aux] = x11 + x21
				#gty2[k][i22+k1][j]
				gty2[j + (i22+k1)*NY_aux + k*NX_aux*NY_aux] = u1 * (x11 - x21)
		#END for i1 in range(0, li)

		if l == logd1_aux:
			for j1 in range(NX_aux):
				# gty1[k][j1][j]
				gty1[j + j1*NY_aux + k*NX_aux*NY_aux] = gty2[j + j1*NY_aux + k*NX_aux*NY_aux]
		
		else:
			n1 = int(NX_aux / 2)
			lk = 1 << (l+1 - 1)
			li = 1 << (logd1_aux - (l+1))
			lj = 2 * lk
			ku = li
			for i1 in range(0, li):
				i11 = i1 * lk
				i12 = i11 + n1
				i21 = i1 * lj
				i22 = i21 + lk
				u1 = complex(0.0, 0.0)
				if iss >= 1:
					u1 = u_device[ku+i1]
				else:
					u1 = u_device[ku+i1].conjugate()
				
				for k1 in range(0, lk):
					# gty2[k][i11+k1][j]
					x12 = gty2[j + (i11+k1)*NY_aux + k*NX_aux*NY_aux]
					# gty2[k][i12+k1][j]
					x22 = gty2[j + (i12+k1)*NY_aux + k*NX_aux*NY_aux]
					# gty1[k][i21+k1][j]
					gty1[j + (i21+k1)*NY_aux + k*NX_aux*NY_aux] = x12 + x22
					# gty1[k][i22+k1][j]
					gty1[j + (i22+k1)*NY_aux + k*NX_aux*NY_aux] = u1 * (x12 - x22)
		#END if l == logd1_aux:
	#END for l in range(1, logd1_aux+1, 2):
#END cffts1_gpu_kernel_2()


# ----------------------------------------------------------------------
# y0[z][x][y] = x_in[z][y][x] 
#
# y0[y + x*NY + z*NX*NY] = x_in[x + y*NX + z*NX*NY] 
# ----------------------------------------------------------------------
@cuda.jit('void(complex128[:], complex128[:], int32, int32, int32)')
def cffts1_gpu_kernel_1(x_in, 
						y0,
						NX_aux, NY_aux, NZ_aux):
	x_y_z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	if x_y_z >= (NX_aux*NY_aux*NZ_aux):
		return

	x = x_y_z % NX_aux
	y = int(x_y_z / NX_aux) % NY_aux
	z = int(x_y_z / (NX_aux * NY_aux))
	y0[y+(x*NY_aux)+(z*NX_aux*NY_aux)] = x_in[x_y_z]
#END cffts1_gpu_kernel_1()


def cffts1_gpu(iss, 
			u, 
			x_in, 
			x_out, 
			y0, 
			y1):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTX_1)
	cffts1_gpu_kernel_1[blocks_per_grid_on_fftx_1,
		threads_per_block_on_fftx_1](x_in, 
									y0,
									NX, NY, NZ)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTX_1)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTX_2)
	logd1 = ilog2(NX)
	cffts1_gpu_kernel_2[blocks_per_grid_on_fftx_2,
		threads_per_block_on_fftx_2](iss, 
									y0, 
									y1, 
									u,
									NX, NY, NZ, logd1)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTX_2)

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_FFTX_3)
	cffts1_gpu_kernel_3[blocks_per_grid_on_fftx_3,
		threads_per_block_on_fftx_3](x_out, 
									y0,
									NX, NY, NZ)
	cuda.synchronize()
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_FFTX_3)
#END cffts1_gpu()


def fft_gpu(dirr,
			x1,
			x2):
	global u_device, y0_device, y1_device
	
	# ---------------------------------------------------------------------
	# note: args x1, x2 must be different arrays
	# note: args for cfftsx are (direction, layout, xin, xout, scratch)
	# xin/xout may be the same and it can be somewhat faster
	# if they are
	# ---------------------------------------------------------------------
	if dirr == 1:
		cffts1_gpu(1, u_device, x1, x1, y0_device, y1_device)
		cffts2_gpu(1, u_device, x1, x1, y0_device, y1_device)
		cffts3_gpu(1, u_device, x1, x2, y0_device, y1_device)
	else:
		cffts3_gpu(-1, u_device, x1, x1, y0_device, y1_device)
		cffts2_gpu(-1, u_device, x1, x1, y0_device, y1_device)
		cffts1_gpu(-1, u_device, x1, x2, y0_device, y1_device)
#END fft_gpu()


def fft_init_gpu(n, u):
	global u_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_INIT)
		
	# ---------------------------------------------------------------------
	# initialize the U array with sines and cosines in a manner that permits
	# stride one access at each FFT iteration.
	# ---------------------------------------------------------------------	
	m = ilog2(n)
	u[0] = complex(float(m), 0.0)
	ku = 2
	ln = 1

	m_cos, m_sin = math.cos, math.sin
	for j in range(1, m+1):
		t = PI / ln

		for i in range(ln):
			ti = i * t
			u[i+ku-1] = complex(m_cos(ti), m_sin(ti))

		ku = ku + ln
		ln = 2 * ln
		
	u_device = cuda.to_device(u)
	
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_INIT)
#END fft_init_gpu()


@cuda.jit('float64(int32, float64, float64, complex128[:])', device=True)
def vranlc_device(n, x_seed, a, y):
	t1 = c_randdp.r23 * a
	a1 = int(t1)
	a2 = a - c_randdp.t23 * a1
	x = x_seed
	
	idx = 0
	for i in range(n):
		t1 = c_randdp.r23 * x
		x1 = int(t1)
		x2 = x - c_randdp.t23 * x1
		t1 = a1 * x2 + a2 * x1
		t2 = int(c_randdp.r23 * t1)
		z = t1 - c_randdp.t23 * t2
		t3 = c_randdp.t23 * z + a2 * x2
		t4 = int(c_randdp.r46 * t3)
		x = t3 - c_randdp.t46 * t4
		
		if i % 2 == 0: #even: alter real part
			y[idx] = complex(c_randdp.r46 * x, y[idx].imag)
		else: #odd: alter imaginary part
			y[idx] = complex(y[idx].real, c_randdp.r46 * x)
			idx += 1
	
	x_seed = x
	return x_seed
#END vranlc_device()


@cuda.jit('void(complex128[:], float64[:], int32, int32, int32)')
def compute_initial_conditions_gpu_kernel(u0, 
									starts,
									NX_aux, NY_aux, NZ_aux):
	z = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if z >= NZ_aux:
		return

	x0 = starts[z]
	for y in range(NY_aux):
		#vranlc_device(2*NX_aux, &x0, A, (double*)&u0[ 0 + y*NX_aux + z*NX_aux*NY_aux ]);
		idx = 0 + y*NX_aux + z*NX_aux*NY_aux
		x0 = vranlc_device(2 * NX_aux, x0, A, u0[idx:])
#END compute_initial_conditions_gpu_kernel()


def compute_initial_conditions_gpu(u0):
	global starts_device
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_INITIAL_CONDITIONS)

	starts = numpy.empty(NZ, numpy.float64)
	start = SEED

	an = ipow46(A, 0)
	aux, start = randlc(start, an)
	an = ipow46(A, 2*NX*NY)

	starts[0] = start
	for z in range(1, NZ):
		aux, start = randlc(start, an)
		starts[z] = start

	starts_device = cuda.to_device(starts)

	compute_initial_conditions_gpu_kernel[blocks_per_grid_on_compute_initial_conditions,
		threads_per_block_on_compute_initial_conditions](u0, 
														starts_device,
														NX, NY, NZ)
		
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_INITIAL_CONDITIONS)
#END compute_initial_conditions_gpu()


@cuda.jit('void(float64[:], int32, int32, int32, int32)')
def compute_indexmap_gpu_kernel(twiddle,
							NTOTAL_aux, NX_aux, NY_aux, NZ_aux):
	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if thread_id >= NTOTAL_aux:
		return

	i = thread_id % NX_aux
	j = int( int(thread_id / NX_aux) % NY_aux )
	k = int( thread_id / (NX_aux * NY_aux) )

	kk = int( ((k+NZ_aux/2) % NZ_aux) - (NZ_aux/2) )
	kk2 = kk * kk
	jj = int( ((j+NY_aux/2) % NY_aux) - (NY_aux/2) )
	kj2 = jj * jj + kk2
	ii = int( ((i+NX_aux/2) % NX_aux) - (NX_aux/2) )

	twiddle[thread_id] = math.exp(AP * (ii*ii+kj2))
#END compute_indexmap_gpu_kernel()


def compute_indexmap_gpu(twiddle):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_INDEXMAP)

	compute_indexmap_gpu_kernel[blocks_per_grid_on_compute_indexmap,
		threads_per_block_on_compute_indexmap](twiddle,
											npbparams.NTOTAL, NX, NY, NZ)
		
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_INDEXMAP)
#END compute_indexmap_gpu()


@cuda.jit('void(complex128[:], complex128[:], float64[:], int32)')
def init_ui_gpu_kernel(u0,
					u1,
					twiddle,
					NTOTAL_aux):
	thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if thread_id >= NTOTAL_aux:
		return

	u0[thread_id] = complex(0.0, 0.0)
	u1[thread_id] = complex(0.0, 0.0)
	twiddle[thread_id] = 0.0
#END init_ui_gpu_kernel()


def init_ui_gpu(u0,
			u1,
			twiddle):
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_INIT_UI)

	init_ui_gpu_kernel[blocks_per_grid_on_init_ui,
		threads_per_block_on_init_ui](u0, 
									u1,
									twiddle,
									npbparams.NTOTAL)
	cuda.synchronize()

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_INIT_UI)
#END init_ui_gpu()


def main():
	global u
	global twiddle_device
	global u0_device, u1_device, sums_device
	
	if gpu_config.PROFILING:
		print(" PROFILING mode on")
	
	setup()
	setup_gpu()
	init_ui_gpu(u0_device, u1_device, twiddle_device)

	#Threads to launch kernels not implemented 
	compute_indexmap_gpu(twiddle_device)
	compute_initial_conditions_gpu(u1_device)
	fft_init_gpu(npbparams.MAXDIM, u)

	cuda.synchronize()
	
	fft_gpu(1, u1_device, u0_device)
	
	# ---------------------------------------------------------------------
	# start over from the beginning. note that all operations must
	# be timed, in contrast to other benchmarks. 
	# ---------------------------------------------------------------------
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		c_timers.timer_clear(PROFILING_INDEXMAP)
		c_timers.timer_clear(PROFILING_INITIAL_CONDITIONS)
		c_timers.timer_clear(PROFILING_INITIAL_CONDITIONS)
		c_timers.timer_clear(PROFILING_EVOLVE)
		c_timers.timer_clear(PROFILING_FFTX_1)
		c_timers.timer_clear(PROFILING_FFTX_2)
		c_timers.timer_clear(PROFILING_FFTX_3)
		c_timers.timer_clear(PROFILING_FFTY_1)
		c_timers.timer_clear(PROFILING_FFTY_2)
		c_timers.timer_clear(PROFILING_FFTY_3)
		c_timers.timer_clear(PROFILING_FFTZ_1)
		c_timers.timer_clear(PROFILING_FFTZ_2)
		c_timers.timer_clear(PROFILING_FFTZ_3)
		c_timers.timer_clear(PROFILING_CHECKSUM)
	
	c_timers.timer_start(PROFILING_TOTAL_TIME)

	#Threads to launch kernels not implemented 
	compute_indexmap_gpu(twiddle_device)
	compute_initial_conditions_gpu(u1_device)
	fft_init_gpu(npbparams.MAXDIM, u)
	
	cuda.synchronize()
	
	fft_gpu(1, u1_device, u0_device)
	
	for it in range(1, niter+1):
		evolve_gpu(u0_device, u1_device, twiddle_device)
		fft_gpu(-1, u1_device, u1_device)
		checksum_gpu(it, u1_device, sums_device)
	
	for it in range(1, niter+1):
		print("T = %5d     Checksum = %22.12e %22.12e" % (it, sums[it].real, sums[it].imag))

	verified = verify(NX, NY, NZ, niter)

	c_timers.timer_stop(PROFILING_TOTAL_TIME)
	total_time = c_timers.timer_read(PROFILING_TOTAL_TIME)

	mflops = 0.0
	if total_time != 0.0:
		mflops = ( 1.0e-6 * npbparams.NTOTAL *
			(14.8157 + 7.19641 * math.log(npbparams.NTOTAL)
			 + (5.23518 + 7.21113 * math.log(npbparams.NTOTAL)) * niter)
			/ total_time )
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
	
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_INDEXMAP)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" indexmap", threads_per_block_on_compute_indexmap, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_INITIAL_CONDITIONS)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" initial conditions", threads_per_block_on_compute_initial_conditions, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_INIT_UI)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" init ui", threads_per_block_on_init_ui, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_EVOLVE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" evolve", threads_per_block_on_evolve, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTX_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" fftx 1", threads_per_block_on_fftx_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTX_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" fftx 2", threads_per_block_on_fftx_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTX_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" fftx 3", threads_per_block_on_fftx_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTY_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" ffty 1", threads_per_block_on_ffty_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTY_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" ffty 2", threads_per_block_on_ffty_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTY_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" ffty 3", threads_per_block_on_ffty_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTZ_1)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" fftz 1", threads_per_block_on_fftz_1, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTZ_2)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" fftz 2", threads_per_block_on_fftz_2, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_FFTZ_3)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" fftz 3", threads_per_block_on_fftz_3, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_CHECKSUM)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%" % (" checksum", threads_per_block_on_checksum, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" indexmap", threads_per_block_on_compute_indexmap)
		gpu_config_string += "%29s\t%25d\n" % (" initial conditions", threads_per_block_on_compute_initial_conditions)
		gpu_config_string += "%29s\t%25d\n" % (" init ui", threads_per_block_on_init_ui)
		gpu_config_string += "%29s\t%25d\n" % (" evolve", threads_per_block_on_evolve)
		gpu_config_string += "%29s\t%25d\n" % (" fftx 1", threads_per_block_on_fftx_1)
		gpu_config_string += "%29s\t%25d\n" % (" fftx 2", threads_per_block_on_fftx_2)
		gpu_config_string += "%29s\t%25d\n" % (" fftx 3", threads_per_block_on_fftx_3)
		gpu_config_string += "%29s\t%25d\n" % (" ffty 1", threads_per_block_on_ffty_1)
		gpu_config_string += "%29s\t%25d\n" % (" ffty 2", threads_per_block_on_ffty_2)
		gpu_config_string += "%29s\t%25d\n" % (" ffty 3", threads_per_block_on_ffty_3)
		gpu_config_string += "%29s\t%25d\n" % (" fftz 1", threads_per_block_on_fftz_1)
		gpu_config_string += "%29s\t%25d\n" % (" fftz 2", threads_per_block_on_fftz_2)
		gpu_config_string += "%29s\t%25d\n" % (" fftz 3", threads_per_block_on_fftz_3)
		gpu_config_string += "%29s\t%25d" % (" checksum", threads_per_block_on_checksum)
	
	c_print_results.c_print_results("FT",
			npbparams.CLASS,
			npbparams.NX, 
			npbparams.NY,
			npbparams.NZ,
			niter,
			total_time,
			mflops,
			"          floating point",
			verified,
			device_prop.name,
			gpu_config_string)
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA FT')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_ft_info(args.CLASS)
	set_global_variables()
	
	main()
