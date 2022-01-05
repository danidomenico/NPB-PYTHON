# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
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
# 	Dalvan Griebler <dalvangriebler@gmail.com>
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

import time
import numpy
from numba import cuda

# Global variables
start = numpy.repeat(0.0, 64)
elapsed = numpy.repeat(0.0, 64)

#*****************************************************************
#******            T  I  M  E  R  _  C  L  E  A  R          ******
#*****************************************************************
def timer_clear(n):
	global elapsed
	elapsed[n] = 0.0

#*****************************************************************
#******            T  I  M  E  R  _  S  T  A  R  T          ******
#*****************************************************************
def timer_start(n):
	global start
	start[n] = time.time()

#*****************************************************************
#******            T  I  M  E  R  _  S  T  O  P             ******
#*****************************************************************
def timer_stop(n):
	global elapsed
	cuda.synchronize()
	t = (time.time() - start[n])
	elapsed[n] += t
	
def timer_stop_assync(n):
	global elapsed
	t = (time.time() - start[n])
	elapsed[n] += t

#*****************************************************************
#******            T  I  M  E  R  _  R  E  A  D             ******
#*****************************************************************
def timer_read(n):
	return elapsed[n]
