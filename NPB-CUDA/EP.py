# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	P. O. Frederickson
#	D. H. Bailey
#	A. C. Woo
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
from numba import cuda
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
import c_randdp
import c_timers
import c_print_results

sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')))
import gpu_config

# Global variables
M = 0
MK = 16
MM = 0
NN = 0
NK = 0
NQ = 10
EPSILON = 1.0e-8
A = 1220703125.0
S = 271828183.0
NK_PLUS = 0
RECOMPUTATION = 128
RECOMPUTATION_SIZE = 2 * RECOMPUTATION
PROFILING_TOTAL_TIME = 0

# GPU variables
q_device = None
sx_device = None
sy_device = None
threads_per_block = 0
blocks_per_grid = 0
gpu_device_id = 0
total_devices = 0
device_prop = None

def set_global_variables():
	global M, MM, NN, NK, NK_PLUS
	M = npbparams.M
	MM = (M - MK)
	NN = (1 << MM)
	NK = (1 << MK)
	NK_PLUS = ((2*NK)+1)
#END set_global_variables()

@cuda.jit('float64(float64, float64)', device=True)
def randlc_device(x, a):
	t1 = c_randdp.r23 * a
	a1 = int(t1)
	a2 = a - c_randdp.t23 * a1
	t1 = c_randdp.r23 * x
	x1 = int(t1)
	x2 = x - c_randdp.t23 * x1
	t1 = a1 * x2 + a2 * x1
	t2 = int(c_randdp.r23 * t1)
	z = t1 - c_randdp.t23 * t2
	t3 = c_randdp.t23 * z + a2 * x2
	t4 = int(c_randdp.r46 * t3)
	x = t3 - c_randdp.t46 * t4
	return x
#END randlc_device()

@cuda.jit('float64(int32, float64, float64, float64[:])', device=True)
def vranlc_device(n, x_seed, a, y):
	t1 = c_randdp.r23 * a
	a1 = int(t1)
	a2 = a - c_randdp.t23 * a1
	x = x_seed
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
		y[i] = c_randdp.r46 * x
	
	x_seed = x
	return x_seed
#END vranlc_device()

@cuda.jit('void(float64[:], float64[:], float64[:], float64, int64)')
def gpu_kernel(q_global, 
		sx_global, 
		sy_global,
		an,
		NK_aux):

	x_local = cuda.local.array(RECOMPUTATION_SIZE, numba.float64)
	q_local = cuda.local.array(NQ, numba.float64)
	for i in range(NQ):
		q_local[i] = 0.0
	sx_local = 0.0
	sy_local = 0.0

	kk = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	t1 = S
	t2 = an
	
	# find starting seed t1 for this kk 
	for i in range(1,100+1):
		ik = int(kk / 2)
		if (2 * ik) != kk:
			t1 = randlc_device(t1, t2)
		if ik == 0:
			break
		t2 = randlc_device(t2, t2)
		kk = int(ik)
		
	math_sqrt = math.sqrt
	math_log  = math.log
	seed = t1
	for ii in range(0, NK_aux, RECOMPUTATION):
		# compute uniform pseudorandom numbers
		seed = vranlc_device(RECOMPUTATION_SIZE, seed, A, x_local)

		# compute gaussian deviates by acceptance-rejection method and
		# tally counts in concentric square annuli. this loop is not
		# vectorizable.
		for i in range(RECOMPUTATION):
			x1 = 2.0 * x_local[2*i] - 1.0
			x2 = 2.0 * x_local[2*i+1] - 1.0
			t1 = x1 * x1 + x2 * x2
			if t1 <= 1.0:
				t2 = math_sqrt(-2.0 * math_log(t1) / t1)
				t3 = x1 * t2
				t4 = x2 * t2
				l = int(max(abs(t3), abs(t4)))
				q_local[l] += 1.0
				sx_local += t3
				sy_local += t4
	#END for ii in range(0, NK_aux, RECOMPUTATION)
	
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 0, q_local[0])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 1, q_local[1])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 2, q_local[2])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 3, q_local[3])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 4, q_local[4])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 5, q_local[5])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 6, q_local[6])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 7, q_local[7])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 8, q_local[8])
	cuda.atomic.add(q_global, cuda.blockIdx.x * NQ + 9, q_local[9])
	cuda.atomic.add(sx_global, cuda.blockIdx.x, sx_local)
	cuda.atomic.add(sy_global, cuda.blockIdx.x, sy_local)
#END gpu_kernel()

def setup_gpu():
	global threads_per_block, blocks_per_grid
	global gpu_device_id, total_devices
	global device_prop
	global q_device, sx_device, sy_device
	
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
	EP_THREADS_PER_BLOCK = gpu_config.EP_THREADS_PER_BLOCK
	if EP_THREADS_PER_BLOCK >= 1 and EP_THREADS_PER_BLOCK <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block = EP_THREADS_PER_BLOCK
	else:
		threads_per_block = device_prop.WARP_SIZE

	blocks_per_grid = int(NN / threads_per_block)
	
	n_float64 = numpy.float64
	q_device = cuda.device_array(blocks_per_grid * NQ, n_float64)
	sx_device = cuda.device_array(blocks_per_grid, n_float64)
	sy_device = cuda.device_array(blocks_per_grid, n_float64)
#END setup_gpu()

def main():
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - EP Benchmark\n")
	print(" Number of random numbers generated:", pow(2, M+1))
	
	verified = False
	
	t1 = A

	for i in range(MK+1): 
		t2, t1 = c_randdp.randlc(t1, t1)
		
	an = t1
	gc = 0.0
	sx = 0.0
	sy = 0.0

	q = numpy.repeat(0.0, NQ)
	
	setup_gpu()

	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	gpu_kernel[blocks_per_grid, threads_per_block](
		q_device,
		sx_device,
		sy_device,
		an,
		NK)
	
	c_timers.timer_stop(PROFILING_TOTAL_TIME)
	tm = c_timers.timer_read(PROFILING_TOTAL_TIME)
	
	q_host = q_device.copy_to_host()
	sx_host = sx_device.copy_to_host()
	sy_host = sy_device.copy_to_host()

	for block in range(blocks_per_grid):
		for i in range(NQ):
			q[i] += q_host[block*NQ+i]
		sx += sx_host[block]
		sy += sy_host[block]

	gc = sum(q)
	
	nit = 0
	verified = True
	if M == 24:
		sx_verify_value = -3.247834652034740e+3
		sy_verify_value = -6.958407078382297e+3
	elif M == 25:
		sx_verify_value = -2.863319731645753e+3
		sy_verify_value = -6.320053679109499e+3
	elif M == 28:
		sx_verify_value = -4.295875165629892e+3
		sy_verify_value = -1.580732573678431e+4
	elif M == 30:
		sx_verify_value =  4.033815542441498e+4
		sy_verify_value = -2.660669192809235e+4
	elif M == 32:
		sx_verify_value =  4.764367927995374e+4
		sy_verify_value = -8.084072988043731e+4
	elif M == 36:
		sx_verify_value =  1.982481200946593e+5
		sy_verify_value = -1.020596636361769e+5
	elif M == 40:
		sx_verify_value = -5.319717441530e+05
		sy_verify_value = -3.688834557731e+05
	else:
		verified = False

	if verified:
		sx_err = abs((sx - sx_verify_value) / sx_verify_value)
		sy_err = abs((sy - sy_verify_value) / sy_verify_value)
		verified = (sx_err <= EPSILON) and (sy_err <= EPSILON)

	Mops = pow(2.0, M+1) / tm / 1000000.0
	
	print("\n EP Benchmark Results:\n")
	print(" GPU Time = {0:10.4f}".format(tm))
	print(" N = 2^{0:5d}".format(M))
	print(" No. Gaussian Pairs = {0:15.0f}".format(gc))
	print(" Sums = {0:25.15e} {1:25.15e}".format(sx, sy))
	print(" Counts: ")
	for i in range(NQ):
		print("{0:3d}{1:15.0f}".format(i, q[i]))
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%" % (" ep", threads_per_block, tm, (tm*100/tm))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" ep", threads_per_block)
	
	c_print_results.c_print_results("EP",
			npbparams.CLASS,
			M+1, 
			0,
			0,
			nit,
			tm,
			Mops,
			"Random numbers generated",
			verified,
			device_prop.name,
			gpu_config_string)
#END main()

#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA EP')
	parser.add_argument("-c", "--CLASS", required=True, help="Workload Class")
	args = parser.parse_args()
	
	npbparams.set_ep_info(args.CLASS)
	set_global_variables()
	
	main()
