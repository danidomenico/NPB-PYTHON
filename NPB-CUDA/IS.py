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
PROFILING_TOTAL_TIME = 0
PROFILING_CREATE = 1 
PROFILING_RANK = 2
PROFILING_VERIFY = 3

TOTAL_KEYS_LOG_2 = 16 # Value for S class
MAX_KEY_LOG_2 = 11 # Value for S class

TOTAL_KEYS = 0
MAX_KEY = 0
NUM_KEYS = 0
SIZE_OF_BUFFERS = 0

MAX_ITERATIONS = 10
TEST_ARRAY_SIZE = 5

test_rank_array = None
test_index_array = None

# GPU variables
passed_verification = 0

key_array_device = None
key_buff1_device = None 
key_buff2_device = None
index_array_device = None 
rank_array_device = None
partial_verify_vals_device = None
passed_verification_device = None
key_scan_device = None 
sum_device = None

threads_per_block_on_create_seq = 0
threads_per_block_on_rank = 0
threads_per_block_on_rank_1 = 0
#threads_per_block_on_rank_2 = 0
#threads_per_block_on_rank_3 = 0
#threads_per_block_on_rank_4 = 0
#threads_per_block_on_rank_5 = 0
#threads_per_block_on_rank_6 = 0
threads_per_block_on_rank_7 = 0
threads_per_block_on_full_verify = 0
#threads_per_block_on_full_verify_1 = 0
#threads_per_block_on_full_verify_2 = 0
#threads_per_block_on_full_verify_3 = 0

blocks_per_grid_on_create_seq = 0
blocks_per_grid_on_rank_1 = 0
blocks_per_grid_on_rank_2 = 0
blocks_per_grid_on_rank_3 = 0
blocks_per_grid_on_rank_4 = 0
blocks_per_grid_on_rank_5 = 0
blocks_per_grid_on_rank_6 = 0
blocks_per_grid_on_rank_7 = 0
blocks_per_grid_on_full_verify_1 = 0
blocks_per_grid_on_full_verify_2 = 0
blocks_per_grid_on_full_verify_3 = 0

amount_of_work_on_create_seq = 0
#amount_of_work_on_rank_1 = 0
amount_of_work_on_rank_2 = 0
amount_of_work_on_rank_3 = 0
amount_of_work_on_rank_4 = 0
amount_of_work_on_rank_5 = 0
amount_of_work_on_rank_6 = 0
#amount_of_work_on_rank_7 = 0
amount_of_work_on_full_verify_1 = 0
amount_of_work_on_full_verify_2 = 0
amount_of_work_on_full_verify_3 = 0

gpu_device_id = 0
total_devices = 0
device_prop = None

def set_global_variables():
	global TOTAL_KEYS_LOG_2, MAX_KEY_LOG_2
	global TOTAL_KEYS, MAX_KEY, NUM_KEYS, SIZE_OF_BUFFERS
	
	if npbparams.CLASS == 'W':
		TOTAL_KEYS_LOG_2 = 20
		MAX_KEY_LOG_2 = 16
	elif npbparams.CLASS == 'A':
		TOTAL_KEYS_LOG_2 = 23
		MAX_KEY_LOG_2 = 19
	elif npbparams.CLASS == 'B':
		TOTAL_KEYS_LOG_2 = 25
		MAX_KEY_LOG_2 = 21
	elif npbparams.CLASS == 'C':
		TOTAL_KEYS_LOG_2 = 27
		MAX_KEY_LOG_2 = 23
	elif npbparams.CLASS == 'D':
		TOTAL_KEYS_LOG_2 = 31
		MAX_KEY_LOG_2 = 27
	
	TOTAL_KEYS = (1 << TOTAL_KEYS_LOG_2)
	MAX_KEY = (1 << MAX_KEY_LOG_2)
	NUM_KEYS = TOTAL_KEYS
	SIZE_OF_BUFFERS = NUM_KEYS  
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


def class_to_number():
	class_ = 0 #S
	if npbparams.CLASS == 'W':
		class_ = 1
	elif npbparams.CLASS == 'A':
		class_ = 2
	elif npbparams.CLASS == 'B':
		class_ = 3
	elif npbparams.CLASS == 'C':
		class_ = 4
	elif npbparams.CLASS == 'D':
		class_ = 5
	
	return class_
#END def class_to_number()


def setup_gpu():
	global gpu_device_id, total_devices
	global device_prop
	
	#Threads per block
	global threads_per_block_on_create_seq, threads_per_block_on_rank, threads_per_block_on_full_verify
	global threads_per_block_on_rank_1, threads_per_block_on_rank_7
	
	#Amaount of work
	global amount_of_work_on_create_seq
	#global amount_of_work_on_rank_1, amount_of_work_on_rank_7
	global amount_of_work_on_rank_2, amount_of_work_on_rank_3, amount_of_work_on_rank_4
	global amount_of_work_on_rank_5, amount_of_work_on_rank_6 
	global amount_of_work_on_full_verify_1, amount_of_work_on_full_verify_2, amount_of_work_on_full_verify_3
	
	#Blocks per grid
	global blocks_per_grid_on_create_seq
	global blocks_per_grid_on_rank_1, blocks_per_grid_on_rank_2, blocks_per_grid_on_rank_3
	global blocks_per_grid_on_rank_4, blocks_per_grid_on_rank_5, blocks_per_grid_on_rank_6
	global blocks_per_grid_on_rank_7
	global blocks_per_grid_on_full_verify_1, blocks_per_grid_on_full_verify_2, blocks_per_grid_on_full_verify_3
	
	# Arrays
	global key_array_device, key_buff1_device, key_buff2_device 
	global index_array_device, rank_array_device, partial_verify_vals_device
	global passed_verification_device, key_scan_device, sum_device
	
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
	aux_threads_per_block = gpu_config.IS_THREADS_PER_BLOCK_ON_CREATE_SEQ
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_create_seq = aux_threads_per_block
	else: 
		threads_per_block_on_create_seq = device_prop.WARP_SIZE
	
	aux_threads_per_block = gpu_config.IS_THREADS_PER_BLOCK_ON_RANK
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_rank = aux_threads_per_block
	else:
		threads_per_block_on_rank = device_prop.WARP_SIZE

	aux_threads_per_block = gpu_config.IS_THREADS_PER_BLOCK_ON_FULL_VERIFY
	if aux_threads_per_block >=1 and aux_threads_per_block <= device_prop.MAX_THREADS_PER_BLOCK:
		threads_per_block_on_full_verify = aux_threads_per_block
	else:
		threads_per_block_on_full_verify = device_prop.WARP_SIZE
	
	threads_per_block_on_rank_1 = 1
	threads_per_block_on_rank_7 = 1
	
	amount_of_work_on_create_seq = threads_per_block_on_create_seq * threads_per_block_on_create_seq
	#amount_of_work_on_rank_1 = 1
	amount_of_work_on_rank_2 = MAX_KEY
	amount_of_work_on_rank_3 = NUM_KEYS
	amount_of_work_on_rank_4 = threads_per_block_on_rank * threads_per_block_on_rank
	amount_of_work_on_rank_5 = threads_per_block_on_rank 
	amount_of_work_on_rank_6 = threads_per_block_on_rank * threads_per_block_on_rank
	#amount_of_work_on_rank_7 = 1
	amount_of_work_on_full_verify_1 = NUM_KEYS
	amount_of_work_on_full_verify_2 = NUM_KEYS
	amount_of_work_on_full_verify_3 = NUM_KEYS
	
	m_ceil = math.ceil
	blocks_per_grid_on_create_seq = m_ceil(amount_of_work_on_create_seq / threads_per_block_on_create_seq)
	blocks_per_grid_on_rank_1 = 1
	blocks_per_grid_on_rank_2 = m_ceil(amount_of_work_on_rank_2 / threads_per_block_on_rank)
	blocks_per_grid_on_rank_3 = m_ceil(amount_of_work_on_rank_3 / threads_per_block_on_rank)
	if amount_of_work_on_rank_4 > MAX_KEY:
		amount_of_work_on_rank_4 = MAX_KEY
	blocks_per_grid_on_rank_4 = m_ceil(amount_of_work_on_rank_4 / threads_per_block_on_rank)
	blocks_per_grid_on_rank_5 = 1
	if amount_of_work_on_rank_6 > MAX_KEY:
		amount_of_work_on_rank_6 = MAX_KEY
	blocks_per_grid_on_rank_6 = m_ceil(amount_of_work_on_rank_6 / threads_per_block_on_rank)
	blocks_per_grid_on_rank_7 = 1
	blocks_per_grid_on_full_verify_1 = m_ceil(amount_of_work_on_full_verify_1 / threads_per_block_on_full_verify)
	blocks_per_grid_on_full_verify_2 = m_ceil(amount_of_work_on_full_verify_2 / threads_per_block_on_full_verify)
	blocks_per_grid_on_full_verify_3 = m_ceil(amount_of_work_on_full_verify_3 / threads_per_block_on_full_verify)
	
	n_int32 = numpy.int32
	key_array_device = cuda.device_array(SIZE_OF_BUFFERS, n_int32)
	key_buff1_device = cuda.device_array(MAX_KEY, n_int32)
	key_buff2_device = cuda.device_array(SIZE_OF_BUFFERS, n_int32)
	partial_verify_vals_device = cuda.device_array(TEST_ARRAY_SIZE, n_int32)
	key_scan_device = cuda.device_array(MAX_KEY, n_int32)
	sum_device = cuda.device_array(threads_per_block_on_rank, n_int32)
	
	index_array_device = cuda.to_device(test_index_array)
	rank_array_device = cuda.to_device(test_rank_array)
	
	passed_verification_aux = numpy.zeros(1, n_int32)
	passed_verification_device = cuda.to_device(passed_verification_aux)
#END def setup_gpu()


#*****************************************************************
#******************** GENERATE SEQ FUNCTIONS *********************
#*****************************************************************
#void create_seq_gpu(double seed, double a)
def create_seq_gpu(seed, a):  
	create_seq_gpu_kernel[blocks_per_grid_on_create_seq, 
		threads_per_block_on_create_seq](key_array_device,
				seed,
				a,
				blocks_per_grid_on_create_seq,
				amount_of_work_on_create_seq,
				NUM_KEYS, MAX_KEY)
	cuda.synchronize()
#END create_seq_gpu()

@cuda.jit('void(float64, float64, float64[:])', device=True)
def randlc_device(x, a, ret):
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
	
	ret[0] = (c_randdp.r46 * x)
	ret[1] = x
#END randlc_device()

@cuda.jit('float64(int32, int32, int64, float64, float64)', device=True)
def find_my_seed_device(kn,
					np,
					nn,
					s,
					a):
	if kn == 0:
		return s

	mq = int((nn / 4 + np - 1) / np)
	nq = int(mq * 4 * kn) # number of rans to be skipped

	t1 = s
	t2 = a
	kk = nq
	ret_randlc = cuda.local.array(2, numba.float64)
	while kk > 1:
		ik = int(kk / 2)
		if (2 * ik) ==  kk:
			randlc_device(t2, t2, ret_randlc)
			aux, t2 = ret_randlc[0], ret_randlc[1]
			kk = ik
		else:
			randlc_device(t1, t2, ret_randlc)
			aux, t1 = ret_randlc[0], ret_randlc[1]
			kk = kk - 1
	
	randlc_device(t1, t2, ret_randlc)
	return ret_randlc[1]
#END find_my_seed_device()

@cuda.jit('void(int32[:], float64, float64, int32, int32, int32, int32)')
def create_seq_gpu_kernel(key_array,
						seed,
						a,
						number_of_blocks,
						amount_of_work,
						NUM_KEYS_aux, MAX_KEY_aux):
	an = a
	
	myid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	num_procs = amount_of_work

	mq = int((NUM_KEYS_aux + num_procs - 1) / num_procs)
	k1 = mq * myid
	k2 = k1 + mq
	if k2 > NUM_KEYS_aux:
		k2 = NUM_KEYS_aux

	s = find_my_seed_device(myid, 
				  num_procs, 
				  4 * NUM_KEYS_aux, 
				  seed, 
				  an)

	k = int(MAX_KEY_aux / 4)
	
	ret_randlc = cuda.local.array(2, numba.float64)
	for i in range(k1, k2):
		randlc_device(s, an, ret_randlc)
		x, s = ret_randlc[0], ret_randlc[1]
		randlc_device(s, an, ret_randlc)
		x += ret_randlc[0]
		s = ret_randlc[1]
		randlc_device(s, an, ret_randlc)
		x += ret_randlc[0]
		s = ret_randlc[1]
		randlc_device(s, an, ret_randlc)
		x += ret_randlc[0]
		s = ret_randlc[1]
		key_array[i] = int(k * x)
#END create_seq_gpu_kernel()

#*****************************************************************
#************************* RANK FUNCTIONS ************************
#*****************************************************************
def rank_gpu(iteration):
	# rank_gpu_kernel_1
	rank_gpu_kernel_1[blocks_per_grid_on_rank_1, 
		threads_per_block_on_rank_1](key_array_device,
							partial_verify_vals_device,
							index_array_device,
							iteration,
							MAX_KEY)
	
	# rank_gpu_kernel_2
	rank_gpu_kernel_2[blocks_per_grid_on_rank_2, 
		threads_per_block_on_rank](key_buff1_device,
								blocks_per_grid_on_rank_2,
								amount_of_work_on_rank_2)
		
	# rank_gpu_kernel_3
	rank_gpu_kernel_3[blocks_per_grid_on_rank_3, 
		threads_per_block_on_rank](key_buff1_device,
								key_array_device)
		
	# rank_gpu_kernel_4
	stream = 0
	size_shared_data_on_rank4 = (2 * threads_per_block_on_rank) * key_buff1_device.dtype.itemsize
	rank_gpu_kernel_4[blocks_per_grid_on_rank_4, 
		threads_per_block_on_rank, 
		stream, 
		size_shared_data_on_rank4](key_buff1_device,
								key_buff1_device,
								sum_device,
								blocks_per_grid_on_rank_4,
								MAX_KEY)

	# rank_gpu_kernel_5
	stream = 0
	size_shared_data_on_rank_5 = (2 * threads_per_block_on_rank) * sum_device.dtype.itemsize
	rank_gpu_kernel_5[blocks_per_grid_on_rank_5, 
		threads_per_block_on_rank,
		stream,
		size_shared_data_on_rank_5](sum_device,
								sum_device)
		
	# rank_gpu_kernel_6
	rank_gpu_kernel_6[blocks_per_grid_on_rank_6, 
		threads_per_block_on_rank](key_buff1_device,
								key_buff1_device,
								sum_device,
								blocks_per_grid_on_rank_6,
								MAX_KEY)
		
	# rank_gpu_kernel_7
	rank_gpu_kernel_7[blocks_per_grid_on_rank_7, 
		threads_per_block_on_rank_7](partial_verify_vals_device,
								key_buff1_device,
								rank_array_device,
								passed_verification_device,
								iteration,
								NUM_KEYS, class_to_number())
#END rank_gpu()

@cuda.jit('void(int32[:], int32[:], int32[:], int32, int32)')
def rank_gpu_kernel_1(key_array,
					partial_verify_vals,
					test_index_array,
					iteration,
					MAX_KEY_aux):
	
	key_array[iteration] = iteration
	key_array[iteration + MAX_ITERATIONS] = MAX_KEY_aux - iteration

	# --------------------------------------------------------------------
	# determine where the partial verify test keys are, 
	# --------------------------------------------------------------------
	# load into top of array bucket_size  
	# --------------------------------------------------------------------
	for i in range(TEST_ARRAY_SIZE):
		partial_verify_vals[i] = key_array[test_index_array[i]]
#END rank_gpu_kernel_1()

@cuda.jit('void(int32[:], int32, int32)')
def rank_gpu_kernel_2(key_buff1,
					number_of_blocks,
					amount_of_work):
	key_buff1[cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x] = 0
#END rank_gpu_kernel_2()

@cuda.jit('void(int32[:], int32[:])')
def rank_gpu_kernel_3(key_buff_ptr,
					key_buff_ptr2):
	# --------------------------------------------------------------------
	# in this section, the keys themselves are used as their 
	# own indexes to determine how many of each there are: their
	# individual population  
	# --------------------------------------------------------------------
	#
	cuda.atomic.add(key_buff_ptr, key_buff_ptr2[cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x], 1)
#END rank_gpu_kernel_3()

@cuda.jit('void(int32[:], int32[:], int32[:], int32, int32)')
def rank_gpu_kernel_4(source,
					destiny,
					summ,
					number_of_blocks,
					MAX_KEY_aux):
	#int* shared_data = (int*)(extern_share_data);
	shared_data = cuda.shared.array(shape=0, dtype=numba.int32)

	shared_data[cuda.threadIdx.x] = 0
	position = cuda.blockDim.x + cuda.threadIdx.x

	factor = int(MAX_KEY_aux / number_of_blocks)
	start = factor * cuda.blockIdx.x
	end = start + factor

	for i in range(start, end, cuda.blockDim.x):
		shared_data[position] = source[i + cuda.threadIdx.x]
		
		offset = 1 #for(uint offset=1; offset<blockDim.x; offset<<=1)
		for aux in range(1000):
			cuda.syncthreads()
			t = shared_data[position] + shared_data[position - offset]
			cuda.syncthreads()
			shared_data[position] = t
			
			offset <<= 1
			if offset >= cuda.blockDim.x:
				break

		prv_val = 0 if (i == start) else destiny[i - 1]
		destiny[i + cuda.threadIdx.x] = shared_data[position] + prv_val

	cuda.syncthreads()
	if cuda.threadIdx.x == 0:
		summ[cuda.blockIdx.x] = destiny[end-1]
#END rank_gpu_kernel_4()

@cuda.jit('void(int32[:], int32[:])')
def rank_gpu_kernel_5(source,
					destiny):
	#int* shared_data = (int*)(extern_share_data);
	shared_data = cuda.shared.array(shape=0, dtype=numba.int32)

	shared_data[cuda.threadIdx.x] = 0
	position = cuda.blockDim.x + cuda.threadIdx.x
	shared_data[position] = source[cuda.threadIdx.x]

	offset = 1 #for(uint offset=1; offset<blockDim.x; offset<<=1)
	for aux in range(1000):
		cuda.syncthreads()
		t = shared_data[position] + shared_data[position - offset]
		cuda.syncthreads()
		shared_data[position] = t
		
		offset <<= 1
		if offset >= cuda.blockDim.x:
			break

	cuda.syncthreads()

	destiny[cuda.threadIdx.x] = shared_data[position - 1]
#END rank_gpu_kernel_5()

@cuda.jit('void(int32[:], int32[:], int32[:], int32, int32)')
def rank_gpu_kernel_6(source,
					destiny,
					offset,
					number_of_blocks,
					MAX_KEY_aux):
	factor = int(MAX_KEY_aux / number_of_blocks)
	start = factor * cuda.blockIdx.x
	end = start + factor
	sum = offset[cuda.blockIdx.x]
	for i in range(start, end, cuda.blockDim.x):
		destiny[i + cuda.threadIdx.x] = source[i + cuda.threadIdx.x] + sum
#END rank_gpu_kernel_6()

@cuda.jit('void(int32[:], int32[:], int32[:], int32[:], int32, int32, int32)')
def rank_gpu_kernel_7(partial_verify_vals,
					key_buff_ptr,
					test_rank_array,
					passed_verification_device,
					iteration,
					NUM_KEYS_aux, CLASS_aux):
	# --------------------------------------------------------------------
	# this is the partial verify test section 
	# observe that test_rank_array vals are
	# shifted differently for different cases
	# --------------------------------------------------------------------
	passed_verification = 0 
	for i in range(TEST_ARRAY_SIZE):
		k = partial_verify_vals[i] # test vals were put here on partial_verify_vals
		if 0 < k and k <= NUM_KEYS_aux-1:
			key_rank = key_buff_ptr[k-1]
			failed = False
			
			if CLASS_aux == 0: #S see class_to_number()
				if i <= 2:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						passed_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						passed_verification += 1
			elif CLASS_aux == 1: #W
				if i < 2:
					if key_rank != (test_rank_array[i] + (iteration-2)):
						failed = True
					else:
						passed_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						passed_verification += 1
			elif CLASS_aux == 2: #A
				if i <= 2:
					if key_rank != (test_rank_array[i] + (iteration-1)):
						failed = True
					else:
						passed_verification += 1
				else:
					if key_rank != (test_rank_array[i] - (iteration-1)):
						failed = True
					else:
						passed_verification += 1
			elif CLASS_aux == 3: #B
				if i == 1 or i == 2 or i == 4:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						passed_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						passed_verification += 1
			elif CLASS_aux == 4: #C
				if i <= 2:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						passed_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						passed_verification += 1
			elif CLASS_aux == 5: #D
				if i < 2:
					if key_rank != (test_rank_array[i] + iteration):
						failed = True
					else:
						passed_verification += 1
				else:
					if key_rank != (test_rank_array[i] - iteration):
						failed = True
					else:
						passed_verification += 1
			
			if failed:
				print("Failed partial verification: iteration, ", iteration, ", test key ", i)
		#END if 0 < k and k <= NUM_KEYS_aux-1
	#END for i in range(TEST_ARRAY_SIZE)
	
	passed_verification_device[0] += passed_verification
#END rank_gpu_kernel_7()

#*****************************************************************
#********************* FULL VERIFY FUNCTIONS *********************
#*****************************************************************
def full_verify_gpu():
	global passed_verification
	
	size_aux_device = int(amount_of_work_on_full_verify_3 / threads_per_block_on_full_verify)
	memory_aux_device = cuda.device_array(size_aux_device, numpy.int32)
	
	# full_verify_gpu_kernel_1
	full_verify_gpu_kernel_1[blocks_per_grid_on_full_verify_1, 
		threads_per_block_on_full_verify](key_array_device,
										key_buff2_device)
	cuda.synchronize()
	
	# full_verify_gpu_kernel_2
	full_verify_gpu_kernel_2[blocks_per_grid_on_full_verify_2, 
		threads_per_block_on_full_verify](key_buff2_device,
										key_buff1_device,
										key_array_device)
	cuda.synchronize()
	
	# full_verify_gpu_kernel_3
	stream = 0
	size_shared_data_on_full_verify_3 = threads_per_block_on_full_verify * key_array_device.dtype.itemsize
	full_verify_gpu_kernel_3[blocks_per_grid_on_full_verify_3, 
		threads_per_block_on_full_verify,
		stream,
		size_shared_data_on_full_verify_3](key_array_device,
										memory_aux_device,
										NUM_KEYS)
	cuda.synchronize()
	
	# reduce on cpu
	memory_aux_host = memory_aux_device.copy_to_host()
	j = sum(memory_aux_host)
	
	if j != 0:
		print("Full_verify: number of keys out of sort: ", j)
	else:
		passed_verification += 1
#END full_verify_gpu()

@cuda.jit('void(int32[:], int32[:])')
def full_verify_gpu_kernel_1(key_array,
							key_buff2):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	key_buff2[i] = key_array[i]
#END full_verify_gpu_kernel_1()

@cuda.jit('void(int32[:], int32[:], int32[:])')
def full_verify_gpu_kernel_2(key_buff2,
							key_buff_ptr_global,
							key_array):
	value = key_buff2[cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x]
	index = cuda.atomic.add(key_buff_ptr_global, value, -1) - 1
	key_array[index] = value
#END full_verify_gpu_kernel_2()

@cuda.jit('void(int32[:], int32[:], int32)')
def full_verify_gpu_kernel_3(key_array,
							global_aux,
							NUM_KEYS_aux):
	#int* shared_aux = (int*)(extern_share_data);
	shared_aux = cuda.shared.array(shape=0, dtype=numba.int32)

	i = (cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x) + 1

	if i < NUM_KEYS_aux:
		if key_array[i-1] > key_array[i]:
			shared_aux[cuda.threadIdx.x] = 1
		else:
			shared_aux[cuda.threadIdx.x] = 0
	else:
		shared_aux[cuda.threadIdx.x] = 0

	cuda.syncthreads()
	
	i = int(cuda.blockDim.x / 2) #for(i=blockDim.x/2; i>0; i>>=1)
	for aux in range(1000):
		if cuda.threadIdx.x < i:
			shared_aux[cuda.threadIdx.x] += shared_aux[cuda.threadIdx.x + i]
		cuda.syncthreads()
		
		i >>= 1
		if i <= 0:
			break

	if cuda.threadIdx.x == 0:
		global_aux[cuda.blockIdx.x] = shared_aux[0]
#END full_verify_gpu_kernel_3()


def main():
	global passed_verification, passed_verification_device
	
	if gpu_config.PROFILING:
		print(" PROFILING mode on")
		
	c_timers.timer_clear(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		c_timers.timer_clear(PROFILING_CREATE)
		c_timers.timer_clear(PROFILING_RANK)
		c_timers.timer_clear(PROFILING_VERIFY)
		
	if gpu_config.PROFILING: 
		c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	create_verification_arrays()
	
	print("\n\n NAS Parallel Benchmarks 4.1 CUDA Python version - IS Benchmark\n")
	print(" Size:  %ld  (class %s)" % (TOTAL_KEYS, npbparams.CLASS))
	print(" Iterations:   %d\n" % (MAX_ITERATIONS))
	
	setup_gpu()

	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_CREATE)

	# generate random number sequence and subsequent keys on all procs
	create_seq_gpu(314159265.00, # random number gen seed
			1220703125.00) # random number gen mult

	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_CREATE)
	
	# Do one interation for free (i.e., untimed) to guarantee initialization of
	# all data and code pages and respective tables 
	rank_gpu(1)
	
	# start verification counter
	passed_verification_aux = numpy.zeros(1, numpy.int32)
	passed_verification_device = cuda.to_device(passed_verification_aux)
	
	if npbparams.CLASS != 'S':
		print("\n   iteration")
	
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_RANK)

	c_timers.timer_start(PROFILING_TOTAL_TIME)
	
	# this is the main iteration
	for iteration in range(1, MAX_ITERATIONS+1):
		if npbparams.CLASS != 'S':
			print("        %d" % (iteration))
		rank_gpu(iteration)
	
	c_timers.timer_stop(PROFILING_TOTAL_TIME)
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_RANK)
	
	passed_verification = passed_verification_device.copy_to_host()
	
	# this tests that keys are in sequence: sorting of last ranked key seq
	# occurs here, but is an untimed operation                             
	if gpu_config.PROFILING:
		c_timers.timer_start(PROFILING_VERIFY)
	
	full_verify_gpu()

	timecounter = 0.0
	if gpu_config.PROFILING:
		c_timers.timer_stop(PROFILING_VERIFY)
		c_timers.timer_stop(PROFILING_TOTAL_TIME)
		timecounter = c_timers.timer_read(PROFILING_RANK)
	else:
		timecounter = c_timers.timer_read(PROFILING_TOTAL_TIME)
	
	gpu_config_string = ""
	if gpu_config.PROFILING:
		gpu_config_string = "%5s\t%25s\t%25s\t%25s\n" % ("GPU Kernel", "Threads Per Block", "Time in Seconds", "Time in Percentage")
		
		tt = c_timers.timer_read(PROFILING_TOTAL_TIME)
		t1 = c_timers.timer_read(PROFILING_CREATE)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" create", threads_per_block_on_create_seq, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_RANK)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" rank", threads_per_block_on_rank, t1, (t1 * 100 / tt))
		t1 = c_timers.timer_read(PROFILING_VERIFY)
		gpu_config_string += "%29s\t%25d\t%25f\t%24.2f%%\n" % (" verify", threads_per_block_on_full_verify, t1, (t1 * 100 / tt))
	else:
		gpu_config_string = "%5s\t%25s\n" % ("GPU Kernel", "Threads Per Block")
		gpu_config_string += "%29s\t%25d\n" % (" create", threads_per_block_on_create_seq)
		gpu_config_string += "%29s\t%25d\n" % (" rank", threads_per_block_on_rank)
		gpu_config_string += "%29s\t%25d\n" % (" verify", threads_per_block_on_full_verify)
	
	# the final printout
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
			passed_verification > 0,
			device_prop.name,
			gpu_config_string)
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-CUDA IS')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_is_info(args.CLASS)
	set_global_variables()
	
	main()
