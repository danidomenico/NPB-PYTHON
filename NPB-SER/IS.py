# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# Authors of the Fortran code:
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
# The serial Python version is a translation of the NPB serial C++ version
# Serial Python version: https://github.com/PYTHON
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
from numba import njit
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
from c_randdp import vranlc
from c_randdp import randlc
import c_timers
import c_print_results

# Global variables
T_BENCHMARKING = 0
T_INITIALIZATION = 1 
T_SORTING = 2
T_TOTAL_EXECUTION = 3

#***************************************************************#
# For serial IS, buckets are not really req'd to solve NPB1 IS  #
# spec, but their use on some machines improves performance, on #
# other machines the use of buckets compromises performance,    #
# probably because it is extra computation which is not req'd.  #
# (Note: Mechanism not understood, probably cache related)      #
# Example:  SP2-66MhzWN:  50% speedup with buckets              #
# Example:  SGI Indy5000: 50% slowdown with buckets             #
# Example:  SGI O2000:   400% slowdown with buckets (Wow!)      #
#***************************************************************#
# To disable the use of buckets, set False to de following line #
USE_BUCKETS = True

TOTAL_KEYS_LOG_2 = 16 # Value for S class
MAX_KEY_LOG_2 = 11 # Value for S class
NUM_BUCKETS_LOG_2 = 9 # Value for S class

TOTAL_KEYS = 0
MAX_KEY = 0
NUM_BUCKETS = 0
NUM_KEYS = 0
SIZE_OF_BUFFERS = 0

MAX_ITERATIONS = 10
TEST_ARRAY_SIZE = 5

key_buff_ptr_global = None #used by full_verify to get copies of rank info
passed_verification = 0

key_array = None
key_buff1 = None
key_buff2 = None
partial_verify_vals = None
#key_buff1_aptr = None

test_rank_array = None
test_index_array = None

bucket_size = None
bucket_ptrs = None

def set_global_variables():
	global TOTAL_KEYS_LOG_2, MAX_KEY_LOG_2, NUM_BUCKETS_LOG_2
	global TOTAL_KEYS, MAX_KEY, NUM_BUCKETS, NUM_KEYS, SIZE_OF_BUFFERS
	global key_array, key_buff1, key_buff2, partial_verify_vals
	global bucket_ptrs
	global key_buff_ptr_global
	
	if npbparams.CLASS == 'W':
		TOTAL_KEYS_LOG_2 = 20
		MAX_KEY_LOG_2 = 16
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == 'A':
		TOTAL_KEYS_LOG_2 = 23
		MAX_KEY_LOG_2 = 19
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == 'B':
		TOTAL_KEYS_LOG_2 = 25
		MAX_KEY_LOG_2 = 21
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == 'C':
		TOTAL_KEYS_LOG_2 = 27
		MAX_KEY_LOG_2 = 23
		NUM_BUCKETS_LOG_2 = 10
	elif npbparams.CLASS == 'D':
		TOTAL_KEYS_LOG_2 = 31
		MAX_KEY_LOG_2 = 27
		NUM_BUCKETS_LOG_2 = 10
	
	TOTAL_KEYS = (1 << TOTAL_KEYS_LOG_2)
	MAX_KEY = (1 << MAX_KEY_LOG_2)
	NUM_BUCKETS = (1 << NUM_BUCKETS_LOG_2)
	NUM_KEYS = TOTAL_KEYS
	SIZE_OF_BUFFERS = NUM_KEYS  
	
	key_array = numpy.repeat(0, SIZE_OF_BUFFERS)
	key_buff1 = numpy.repeat(0, MAX_KEY)
	key_buff2 = numpy.repeat(0, SIZE_OF_BUFFERS)
	partial_verify_vals = numpy.repeat(0, TEST_ARRAY_SIZE)
	
	bucket_ptrs = numpy.repeat(0, NUM_BUCKETS)
	
	#Innitilize to set variable as an array
	key_buff_ptr_global = numpy.repeat(0, MAX_KEY)
#END set_global_variables()

def create_verification_arrays():
	global test_index_array, test_rank_array
	
	if npbparams.CLASS == 'S':
		test_index_array = numpy.array([48427,17148,23627,62548,4431])
		test_rank_array  = numpy.array([0,18,346,64917,65463])
	elif npbparams.CLASS == 'W':
		test_index_array = numpy.array([357773,934767,875723,898999,404505])
		test_rank_array  = numpy.array([1249,11698,1039987,1043896,1048018])
	elif npbparams.CLASS == 'A':
		test_index_array = numpy.array([2112377,662041,5336171,3642833,4250760])
		test_rank_array  = numpy.array([104,17523,123928,8288932,8388264])
	elif npbparams.CLASS == 'B':
		test_index_array = numpy.array([41869,812306,5102857,18232239,26860214])
		test_rank_array = numpy.array([33422937,10244,59149,33135281,99])
	elif npbparams.CLASS == 'C':
		test_index_array = numpy.array([44172927,72999161,74326391,129606274,21736814])
		test_rank_array  = numpy.array([61147,882988,266290,133997595,133525895])
	elif npbparams.CLASS == 'D':
		test_index_array = numpy.array([1317351170,995930646,1157283250,1503301535,1453734525])
		test_rank_array  = numpy.array([1,36538729,1978098519,2145192618,2147425337])
#END create_verification_arrays()


#*****************************************************************
#*************             R  A  N  K             ****************
#*****************************************************************
#void rank(int iteration)
@njit
def rank(iteration,
		 p_key_array, p_key_buff1, p_key_buff2,
		 p_partial_verify_vals,
		 p_key_buff_ptr_global,
		 p_bucket_size, p_bucket_ptrs):

	shift = 0
	num_bucket_keys = 0
	if USE_BUCKETS:
		shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2
		num_bucket_keys = (1 << shift)

	p_key_array[iteration] = iteration
	p_key_array[iteration + MAX_ITERATIONS] = MAX_KEY - iteration

	# Determine where the partial verify test keys are, load into
	# top of array bucket_size
	for i in range(TEST_ARRAY_SIZE):
		p_partial_verify_vals[i] = p_key_array[test_index_array[i]]
		
	# Setup pointers to key buffers
	key_buff_ptr2 = None
	if USE_BUCKETS:
		key_buff_ptr2 = p_key_buff2
	else:
		key_buff_ptr2 = p_key_array
	key_buff_ptr = p_key_buff1
	
	myid = 0
	num_procs = 1

	# Bucket sort is known to improve cache performance on some
	# cache based systems.  But the actual performance may depend
	# on cache size, problem size.

	if USE_BUCKETS:
		work_buff = p_bucket_size[myid]

		# Initialize 
		for i in range(NUM_BUCKETS):
			work_buff[i] = 0

		# Determine the number of keys in each bucket 
		for i in range(NUM_KEYS):
			work_buff[p_key_array[i] >> shift] += 1

		# Accumulative bucket sizes are the bucket pointers. 
		# These are global sizes accumulated upon to each bucket 
		p_bucket_ptrs[0] = 0
		for k in range(myid):
			p_bucket_ptrs[0] += p_bucket_size[k][0]

		for i in range(1, NUM_BUCKETS):
			p_bucket_ptrs[i] = p_bucket_ptrs[i-1]
			for k in range(myid):
				p_bucket_ptrs[i] += p_bucket_size[k][i]
			for k in range(myid, num_procs):
				p_bucket_ptrs[i] += p_bucket_size[k][i-1]

		# Sort into appropriate bucket
		for i in range(NUM_KEYS): 
			k = p_key_array[i]
			#key_buff2[p_bucket_ptrs[k >> shift]++] = k;
			idx = k >> shift
			p_key_buff2[p_bucket_ptrs[idx]] = k
			p_bucket_ptrs[idx] += 1

		# The bucket pointers now point to the final accumulated sizes
		if myid < (num_procs - 1):
			for i in range(NUM_BUCKETS):
				for k in range(myid+1, num_procs):
					p_bucket_ptrs[i] += p_bucket_size[k][i]

		# Now, buckets are sorted.  We only need to sort keys inside 
		# each bucket, which can be done in parallel.  Because the distribution 
		# of the number of keys in the buckets is Gaussian, the use of 
		# a dynamic schedule should improve load balance, thus, performance
		for i in range(NUM_BUCKETS):
			# Clear the work array section associated with each bucket
			k1 = i * num_bucket_keys
			k2 = k1 + num_bucket_keys
			for k in range(k1, k2):
				key_buff_ptr[k] = 0
			# Ranking of all keys occurs in this section:
			# In this section, the keys themselves are used as their
			# own indexes to determine how many of each there are: their
			# individual population */
			m = p_bucket_ptrs[i-1] if i > 0 else 0
			for k in range(m, p_bucket_ptrs[i]):
				key_buff_ptr[key_buff_ptr2[k]] += 1 # Now they have individual key population
			# To obtain ranks of each key, successively add the individual key 
			# population, not forgetting to add m, the total of lesser keys, 
			# to the first key population
			key_buff_ptr[k1] += m
			for k in range(k1+1, k2):
				key_buff_ptr[k] += key_buff_ptr[k-1]
	
	else: #USE_BUCKETS
		#work_buff = key_buff1_aptr[myid] #Replaced by the following line
		work_buff = key_buff_ptr
		# Clear the work array
		for i in range(MAX_KEY):
			work_buff[i] = 0
		# Ranking of all keys occurs in this section: 
		# In this section, the keys themselves are used as their 
		# own indexes to determine how many of each there are: their 
		# individual population 
		for i in range(NUM_KEYS):
			work_buff[key_buff_ptr2[i]] += 1 # Now they have individual key population
		# To obtain ranks of each key, successively add the individual key population
		for i in range(MAX_KEY-1):
			work_buff[i+1] += work_buff[i]
		# Accumulate the global key population
		#for k in range(1, num_procs): #Not executed (num_procs is always 1)
		#	for i in range(MAX_KEY):
		#		key_buff_ptr[i] += key_buff1_aptr[k][i]
	#END if USE_BUCKETS:
	
	# This is the partial verify test section
	# Observe that test_rank_array vals are 
	# shifted differently for different cases
	local_verification = passed_verification
	for i in range(TEST_ARRAY_SIZE):
		k = p_partial_verify_vals[i] # test vals were put here
		if 0 < k and k <= NUM_KEYS-1:
			key_rank = key_buff_ptr[k-1]
			failed = False
			
			if npbparams.CLASS == 'S':
				if i <= 2:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						local_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						local_verification += 1
			elif npbparams.CLASS == 'W':
				if i < 2:
					if key_rank != (test_rank_array[i] + (iteration-2)):
						failed = True
					else:
						local_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						local_verification += 1
			elif npbparams.CLASS == 'A':
				if i <= 2:
					if key_rank != (test_rank_array[i] + (iteration-1)):
						failed = True
					else:
						local_verification += 1
				else:
					if key_rank != (test_rank_array[i] - (iteration-1)):
						failed = True
					else:
						local_verification += 1
			elif npbparams.CLASS == 'B':
				if i == 1 or i == 2 or i == 4:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						local_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						local_verification += 1
			elif npbparams.CLASS == 'C':
				if i <= 2:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						local_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						local_verification += 1
			elif npbparams.CLASS == 'D':
				if i < 2:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						local_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						local_verification += 1
			
			if failed:
				print("Failed partial verification: iteration, ", iteration, ", test key ", i)
		#END if 0 < k and k <= NUM_KEYS-1
	#END for i in range(TEST_ARRAY_SIZE)

	# Make copies of rank info for use by full_verify: these variables 
	# in rank are local; making them global slows down the code, probably 
	# since they cannot be made register by compiler 
	if iteration == MAX_ITERATIONS:
		for i in range(MAX_KEY):
			p_key_buff_ptr_global[i] = key_buff_ptr[i]
		
	return local_verification
#END rank()


#*****************************************************************
#*************    F  U  L  L  _  V  E  R  I  F  Y     ************
#*****************************************************************
@njit
def full_verify(p_key_buff_ptr_global, p_key_array, p_key_buff2):
	myid = 0
	num_procs = 1

	# Now, finally, sort the keys:
	# Copy keys into work array; keys in key_array will be reassigned.

	if USE_BUCKETS:
		# Buckets are already sorted. Sorting keys within each bucket
		for j in range(NUM_BUCKETS):
			k1 = bucket_ptrs[j-1] if j > 0 else 0
			for i in range(k1, bucket_ptrs[j]):
				p_key_buff_ptr_global[p_key_buff2[i]] -= 1 #k = --key_buff_ptr_global[key_buff2[i]]
				k = p_key_buff_ptr_global[p_key_buff2[i]]
				p_key_array[k] = p_key_buff2[i]
	else:
		for i in range(NUM_KEYS):
			p_key_buff2[i] = p_key_array[i]
		# This is actual sorting. Each thread is responsible for a subset of key values
		j = num_procs
		j = int((MAX_KEY + j - 1) / j)
		k1 = j * myid
		k2 = k1 + j
		if k2 > MAX_KEY:
			k2 = MAX_KEY
		for i in range(NUM_KEYS):
			if p_key_buff2[i] >= k1 and p_key_buff2[i] < k2:
				p_key_buff_ptr_global[p_key_buff2[i]] -= 1 #k = --key_buff_ptr_global[key_buff2[i]]
				k = p_key_buff_ptr_global[p_key_buff2[i]]
				p_key_array[k] = p_key_buff2[i]
	#END USE_BUCKETS

	# Confirm keys correctly sorted: count incorrectly sorted keys, if any
	j = 0
	local_verification = passed_verification
	for i in range(1, NUM_KEYS):
		if p_key_array[i-1] > p_key_array[i]:
			j += 1
	if j != 0:
		print( "Full_verify: number of keys out of sort: ", j)
	else:
		local_verification += 1
	
	return local_verification
#END full_verify()


#*****************************************************************
#************   F  I  N  D  _  M  Y  _  S  E  E  D    ************
#************                                         ************
#************ returns parallel random number seq seed ************
#*****************************************************************
#double find_my_seed(int kn, int np, long nn, double s, double a)
@njit
def find_my_seed(kn, # my processor rank, 0<=kn<=num procs
			np, # np = num procs
			nn, # total num of ran numbers, all procs
			s, # Ran num seed, for ex.: 314159265.00
			a): # Ran num gen mult, try 1220703125.00
	# Create a random number sequence of total length nn residing
	# on np number of processors.  Each processor will therefore have a
	# subsequence of length nn/np.  This routine returns that random
	# number which is the first random number for the subsequence belonging
	# to processor rank kn, and which is used as seed for proc kn ran # gen.
	if kn == 0:
		return s

	mq = int((nn / 4 + np - 1) / np)
	nq = int(mq * 4 * kn) # number of rans to be skipped

	t1 = s
	t2 = a
	kk = nq
	while kk > 1:
		ik = int(kk / 2)
		if (2 * ik) ==  kk:
			aux, t2 = randlc(t2, t2)
			kk = ik
		else:
			aux, t1 = randlc(t1, t2)
			kk = kk - 1
	
	aux, t1 = randlc(t1, t2)

	return t1
#END find_my_seed()

#*****************************************************************
#*************      C  R  E  A  T  E  _  S  E  Q      ************
#*****************************************************************
# void create_seq(double seed, double a)
@njit
def create_seq(seed, a, p_key_array):
	an = a
	
	myid = 0
	num_procs = 1

	mq = int((NUM_KEYS + num_procs - 1) / num_procs)
	k1 = mq * myid
	k2 = k1 + mq
	if k2 > NUM_KEYS:
		k2 = NUM_KEYS

	s = find_my_seed(myid, 
				  num_procs, 
				  4 * NUM_KEYS, 
				  seed, 
				  an)

	k = int(MAX_KEY / 4)

	for i in range(k1, k2):
		x, s = randlc(s, an)
		x_aux, s = randlc(s, an)
		x += x_aux
		x_aux, s = randlc(s, an)
		x += x_aux
		x_aux, s = randlc(s, an)
		x += x_aux
		p_key_array[i] = int(k * x)
#END create_seq()

#*****************************************************************
#*************      A L L O C  _ K E Y _ B U F F      ************
#*****************************************************************
def alloc_key_buff():
	global bucket_size
	num_procs = 1

	if USE_BUCKETS:
		bucket_size = numpy.zeros(shape=(num_procs, NUM_BUCKETS))

		#for i in range(NUM_KEYS): #Already innitialized with zeros 
		#	key_buff2[i] = 0 
	#else: #Not executed due to altered adaptation
		#key_buff1_aptr = numpy.empty(shape=num_procs, dtype=numpy.ndarray)
		#key_buff1_aptr[0] = key_buff1
		#for i in range(1, num_procs):
		#	key_buff1_aptr[i] = numpy.repeat(0, MAX_KEY)
#END alloc_key_buff()

def main():
	global key_array, key_buff1, key_buff2
	global partial_verify_vals
	global key_buff_ptr_global
	global bucket_size, bucket_ptrs
	global passed_verification
	
	timer_on = os.path.isfile("timer.flag")
	c_timers.timer_clear(T_BENCHMARKING)
	if timer_on:
		c_timers.timer_clear(T_INITIALIZATION)
		c_timers.timer_clear(T_SORTING)
		c_timers.timer_clear(T_TOTAL_EXECUTION)
		
	if timer_on: 
		c_timers.timer_start(T_TOTAL_EXECUTION)
		
	create_verification_arrays()
	
	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - IS Benchmark\n")
	print(" Size:  %ld  (class %s)  (%s)" % (TOTAL_KEYS, npbparams.CLASS, ("Using buckets" if USE_BUCKETS else "Not using buckets")))
	print(" Iterations:   %d\n" % (MAX_ITERATIONS))
	
	if timer_on: 
		c_timers.timer_start(T_INITIALIZATION)

	# Generate random number sequence and subsequent keys on all procs
	create_seq(314159265.00, # Random number gen seed
			1220703125.00, # Random number gen mult 
			key_array) 
	
	alloc_key_buff()
	if timer_on:
		c_timers.timer_stop(T_INITIALIZATION)
		
	# Do one interation for free (i.e., untimed) to guarantee initialization of
	# all data and code pages and respective tables 
	rank(1,
		key_array, key_buff1, key_buff2,
		partial_verify_vals,
		key_buff_ptr_global,
		bucket_size, bucket_ptrs)

	# Start verification counter
	passed_verification = 0
	
	if npbparams.CLASS != 'S': 
		print("\n   iteration")
		
	# Start timer
	c_timers.timer_start(T_BENCHMARKING)
	
	# This is the main iteration
	for iteration in range(1, MAX_ITERATIONS+1):
		if npbparams.CLASS != 'S': 
			print("        %d" % (iteration))
		passed_verification += rank(iteration, 
								key_array, key_buff1, key_buff2,
								partial_verify_vals,
								key_buff_ptr_global,
								bucket_size, bucket_ptrs)

	# End of timing, obtain maximum time of all processors
	c_timers.timer_stop(T_BENCHMARKING)
	timecounter = c_timers.timer_read(T_BENCHMARKING)
	
	# This tests that keys are in sequence: sorting of last ranked key seq
	# occurs here, but is an untimed operation
	if timer_on:
		c_timers.timer_start(T_SORTING)
	passed_verification = full_verify(key_buff_ptr_global, 
									key_array, key_buff2)
	if timer_on:
		c_timers.timer_stop(T_SORTING)

	if timer_on: 
		c_timers.timer_stop(T_TOTAL_EXECUTION)

	# The final printout
	if passed_verification != (5 * MAX_ITERATIONS + 1):
		passed_verification = 0
		
	c_print_results.c_print_results("IS",
			npbparams.CLASS,
			int(TOTAL_KEYS / 64), 
			64,
			0,
			MAX_ITERATIONS,
			timecounter,
			(MAX_ITERATIONS * TOTAL_KEYS) / timecounter / 1000000.0,
			"keys ranked",
			passed_verification > 0)

	if timer_on:
		t_total = c_timers.timer_read(T_TOTAL_EXECUTION)
		print("\nAdditional timers -")
		print(" Total execution: %8.3f" % (t_total))
		if t_total == 0.0:
			t_total = 1.0
		timecounter = c_timers.timer_read(T_INITIALIZATION)
		t_percent = timecounter / t_total * 100.0
		print(" Initialization : %8.3f (%5.2f%%)" % (timecounter, t_percent))
		timecounter = c_timers.timer_read(T_BENCHMARKING)
		t_percent = timecounter / t_total * 100.0
		print(" Benchmarking   : %8.3f (%5.2f%%)" % (timecounter, t_percent))
		timecounter = c_timers.timer_read(T_SORTING)
		t_percent = timecounter / t_total * 100.0
		print(" Sorting        : %8.3f (%5.2f%%)" % (timecounter, t_percent))
#END main()

#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER IS')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_is_info(args.CLASS)
	set_global_variables()
	
	main()
