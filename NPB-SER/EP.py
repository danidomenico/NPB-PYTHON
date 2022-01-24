# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
#	P. O. Frederickson
#	D. H. Bailey
#  	A. C. Woo
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
# The serial Python version is a translation of the NPB serial C++ version
# Serial Python version: https://github.com/danidomenico/NPB-PYTHON/tree/master/NPB-SER
# 
# Authors of the Python code:
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
from numba import njit
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
from c_randdp import vranlc
from c_randdp import randlc
import c_timers
import c_print_results

# --------------------------------------------------------------------
# this is the serial version of the app benchmark 1,
# the "embarassingly parallel" benchmark.
# --------------------------------------------------------------------
# M is the Log_2 of the number of complex pairs of uniform (0, 1) random
# numbers. MK is the Log_2 of the size of each batch of uniform random
# numbers.  MK can be set for convenience on a given system, since it does
# not affect the results.
# --------------------------------------------------------------------

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

def set_global_variables():
	global M, MM, NN, NK, NK_PLUS
	M = npbparams.M
	MM = (M - MK)
	NN = (1 << MM)
	NK = (1 << MK)
	NK_PLUS = ((2*NK)+1)
#END set_global_variables()

@njit
def find_start_seed_t1(kk, t1, t2):
	t3 = 0.0
	for i in range(1, 100+1):
		ik = int(kk / 2)
		if (2 * ik) != kk:
			t3, t1 = randlc(t1, t2)
		if ik == 0:
			break
		t3, t2 = randlc(t2, t2)
		kk = int(ik)
	
	return t1
#END find_start_seed_t1()

@njit
def compute_gaussian_deviates(sx, sy, x, q):
	math_sqrt = math.sqrt
	math_log  = math.log
	for i in range(NK):
		x1 = 2.0 * x[2*i] - 1.0
		x2 = 2.0 * x[2*i+1] - 1.0
		t1 = x1 * x1 + x2 * x2
		if t1 <= 1.0:
			t2 = math_sqrt(-2.0 * math_log(t1) / t1)
			t3 = x1 * t2
			t4 = x2 * t2
			l = int(max(abs(t3), abs(t4)))
			q[l] += 1.0
			sx = sx + t3
			sy = sy + t4
	
	return sx, sy
#END compute_gaussian_deviates()

def main():
	dum = numpy.array([1.0, 1.0, 1.0])
	
	timers_enabled = os.path.isfile("timer.flag")
	
	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - EP Benchmark\n")
	print(" Number of random numbers generated:", pow(2, M+1))
	
	verified = False
	
	#--------------------------------------------------------------------
	# compute the number of "batches" of random number pairs generated 
	# per processor. Adjust if the number of processors does not evenly 
	# divide the total number
	# --------------------------------------------------------------------
	np = NN
	
	# call the random number generator functions and initialize
	# the x-array to reduce the effects of paging on the timings.
	# also, call all mathematical functions that are used. make
	# sure these initializations cannot be eliminated as dead code.
	dum[0] = vranlc(0, dum[0], dum[1], numpy.array([]))
	dum[0], dum[1] = randlc(dum[1], dum[2])
	x = numpy.repeat(-1.0e99, NK_PLUS)
	Mops = math.log(math.sqrt(abs(max(1.0, 1.0))))
	
	c_timers.timer_clear(0)
	c_timers.timer_clear(1)
	c_timers.timer_clear(2)
	c_timers.timer_start(0)

	t1 = A
	t1 = vranlc(0, t1, A, x)
	
	# compute AN = A ^ (2 * NK) (mod 2^46)

	t1 = A

	for i in range(MK+1): 
		t2, t1 = randlc(t1, t1)
		
	an = t1
	tt = S
	gc = 0.0
	sx = 0.0
	sy = 0.0

	q = numpy.repeat(0.0, NQ)
	
	# each instance of this loop may be performed independently. we compute
	# the k offsets separately to take into account the fact that some nodes
	# have more numbers to generate than others
	k_offset = -1
	
	#sqrt_func = math.sqrt
	for k in range(1, np+1):
		kk = k_offset + k
		t1 = S
		t2 = an
		
		# find starting seed t1 for this kk 
		t1 = find_start_seed_t1(kk, t1, t2)
			
		# compute uniform pseudorandom numbers 
		if timers_enabled: 
			c_timers.timer_start(2)
		t1 = vranlc(2*NK, t1, A, x)
		if timers_enabled:
			c_timers.timer_stop(2)
		
		# compute gaussian deviates by acceptance-rejection method and
		# tally counts in concentric square annuli. this loop is not
		# vectorizable.

		if timers_enabled:
			c_timers.timer_start(1)
		sx, sy = compute_gaussian_deviates(sx, sy, x, q)
		if timers_enabled:
			c_timers.timer_stop(1)
	#END for k in range(1, np+1):
	
	gc = sum(q) 

	c_timers.timer_stop(0)
	tm = c_timers.timer_read(0)
	
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
	print(" CPU Time = {0:10.4f}".format(tm))
	print(" N = 2^{0:5d}".format(M))
	print(" No. Gaussian Pairs = {0:15.0f}".format(gc))
	print(" Sums = {0:25.15e} {1:25.15e}".format(sx, sy))
	print(" Counts: ")
	for i in range(NQ):
		print("{0:3d}{1:15.0f}".format(i, q[i]))
		
	c_print_results.c_print_results("EP",
			npbparams.CLASS,
			M+1, 
			0,
			0,
			nit,
			tm,
			Mops,
			"Random numbers generated",
			verified)

	if timers_enabled:
		if tm <= 0.0:
			tm = 1.0
		tt = c_timers.timer_read(0)
		print("\nTotal time:     {0:9.3f} ({1:6.2f})".format(tt, tt*100.0/tm));
		tt = c_timers.timer_read(1)
		print("Gaussian pairs: {0:9.3f} ({1:6.2f})".format(tt, tt*100.0/tm));
		tt = c_timers.timer_read(2)
		print("Random numbers: {0:9.3f} ({1:6.2f})".format(tt, tt*100.0/tm));
#END main()

#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER EP')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_ep_info(args.CLASS)
	set_global_variables()
	
	main()
