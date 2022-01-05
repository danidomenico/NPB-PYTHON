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
import math
from numba import njit
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
import npbparams
from c_randdp import randlc
import c_timers
import c_print_results


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
T_INIT = 0
T_BENCH = 1
T_CONJ_GRAD = 2
T_LAST = 3

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
	
	for i in range(T_LAST):
		c_timers.timer_clear(i)
	
	t_names = numpy.empty(T_LAST, dtype=object)
	
	timeron = os.path.isfile("timer.flag")
	if timeron:
		t_names[T_INIT] = "init"
		t_names[T_BENCH] = "benchmk"
		t_names[T_CONJ_GRAD] = "conjgd"
		
	c_timers.timer_start(T_INIT)
		
	firstrow = 0
	lastrow  = npbparams.NA-1
	firstcol = 0
	lastcol  = npbparams.NA-1
	
	zeta_verify_value = create_zeta_verify_value()
		
	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - CG Benchmark\n")
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

	c_timers.timer_stop(T_INIT)

	print(" Initialization time = %15.3f seconds" % (c_timers.timer_read(T_INIT)))

	c_timers.timer_start(T_BENCH)
	
	# --------------------------------------------------------------------
	# ---->
	# main iteration for inverse power method
	# ---->
	# --------------------------------------------------------------------
	for it in range(1, npbparams.NITER+1):
		# the call to the conjugate gradient routine
		if timeron:
			c_timers.timer_start(T_CONJ_GRAD)
		rnorm = conj_grad(colidx, rowstr, x, z, a, p, q, r)
		if timeron:
			c_timers.timer_stop(T_CONJ_GRAD)

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
		zeta = npbparams.SHIFT + 1.0 / norm_temp1
		if it == 1:
			print("\n   iteration           ||r||                 zeta")
		print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

		# normalize z to obtain x
		for j in range(0, end):
			x[j] = norm_temp2 * z[j]
	# end of main iter inv pow meth

	c_timers.timer_stop(T_BENCH)

	# --------------------------------------------------------------------
	# end of timed section
	# --------------------------------------------------------------------

	t = c_timers.timer_read(T_BENCH)

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
	
	c_print_results.c_print_results("CG",
			npbparams.CLASS,
			npbparams.NA, 
			0,
			0,
			npbparams.NITER,
			t,
			mflops,
			"          floating point",
			verified)

	if timeron:
		tmax = c_timers.timer_read(T_BENCH)
		if tmax == 0.0:
			tmax = 1.0
		print("  SECTION   Time (secs)")
		for i in range(T_LAST):
			t = c_timers.timer_read(i)
			if i == T_INIT:
				print("  %8s:%9.3f" % (t_names[i], t))
			else:
				print("  %8s:%9.3f  (%6.2f%%)" % (t_names[i], t, t * 100.0 / tmax))
				if i == T_CONJ_GRAD:
					t = tmax - t
					print("    --> %8s:%9.3f  (%6.2f%%)" % ("rest", t, t * 100.0 / tmax))
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER CG')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_cg_info(args.CLASS)
	set_global_variables()
	
	main()
