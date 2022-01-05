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
from c_randdp import vranlc_complex
from c_randdp import randlc
import c_timers
import c_print_results

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
FFTBLOCK_DEFAULT = 0
FFTBLOCKPAD_DEFAULT = 0
FFTBLOCK = 0
FFTBLOCKPAD = 0
SEED = 314159265.0
A = 1220703125.0
PI = 3.141592653589793238
ALPHA = 1.0e-6
T_TOTAL = 1
T_SETUP = 2
T_FFT = 3
T_EVOLVE = 4
T_CHECKSUM = 5
T_FFTX = 6
T_FFTY = 7
T_FFTZ = 8
T_MAX = 8

NX = 0
NY = 0
NZ = 0

sums = None
twiddle = None
u = None
u0 = None
u1 = None

dims = numpy.empty(3, dtype=numpy.int32)

niter = 0
timers_enabled = False


def set_global_variables():
	global FFTBLOCK_DEFAULT, FFTBLOCKPAD_DEFAULT, FFTBLOCK, FFTBLOCKPAD
	global NX, NY, NZ
	global sums, twiddle, u, u0, u1
	
	FFTBLOCK_DEFAULT = npbparams.DEFAULT_BEHAVIOR
	FFTBLOCKPAD_DEFAULT = npbparams.DEFAULT_BEHAVIOR
	FFTBLOCK = FFTBLOCK_DEFAULT
	FFTBLOCKPAD = FFTBLOCKPAD_DEFAULT
	
	NX = npbparams.NX
	NY = npbparams.NY
	NZ = npbparams.NZ
	
	sums = numpy.empty(npbparams.NITER_DEFAULT+1, dtype=numpy.complex128)
	twiddle = numpy.repeat(0.0, npbparams.NTOTAL)
	u = numpy.empty(npbparams.MAXDIM, dtype=numpy.complex128)
	u0 = numpy.repeat(complex(0.0, 0.0), npbparams.NTOTAL)
	u1 = numpy.repeat(complex(0.0, 0.0), npbparams.NTOTAL)
#END set_global_variables()


def print_timers():
	tstrings = numpy.empty(T_MAX+1, dtype=object)
	tstrings[1] = "          total " 
	tstrings[2] = "          setup " 
	tstrings[3] = "            fft " 
	tstrings[4] = "         evolve " 
	tstrings[5] = "       checksum " 
	tstrings[6] = "          fftx* " 
	tstrings[7] = "          ffty* " 
	tstrings[8] = "          fftz* "

	t_m = c_timers.timer_read(T_TOTAL)
	if t_m <= 0.0:
		t_m = 1.00
	for i in range(1, T_MAX+1):
		t = c_timers.timer_read(i)
		print(" timer %2d(%16s) :%9.4f (%6.2f%%)" % (i, tstrings[i], t, t*100.0/t_m))
	print("  (* Time hasn't gauged: operation is not supported by @njit)")
#END print_timers()


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


@njit
def checksum(i,
		pointer_u1,
		d1,
		d2,
		d3,
		sums):
	#dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	u1 = numpy.reshape(pointer_u1, (NZ, NY, NX))
	chk = complex(0.0, 0.0)
	for j in range(1, 1024+1):
		q = j % NX
		r = (3*j) % NY
		s = (5*j) % NZ
		chk = chk + u1[s][r][q]

	chk = chk / npbparams.NTOTAL
	print(" T =", i, "  Checksum =   ", chk.real, "  ", chk.imag)
	sums[i] = chk
#END checksum()


# ---------------------------------------------------------------------
# evolve u0 -> u1 (t time steps) in fourier space
# ---------------------------------------------------------------------
@njit
def evolve(pointer_u0,
		pointer_u1,
		pointer_twiddle,
		d1,
		d2,
		d3):
	#dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	#dcomplex (*u1)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u1;
	#double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;
	u0 = numpy.reshape(pointer_u0, (NZ, NY, NX))
	u1 = numpy.reshape(pointer_u1, (NZ, NY, NX))
	twiddle = numpy.reshape(pointer_twiddle, (NZ, NY, NX))
	
	for k in range(d3):
		for j in range(d2):
			for i in range(d1):
				u0[k][j][i] = u0[k][j][i] * twiddle[k][j][i]
				u1[k][j][i] = u0[k][j][i]
#END evolve()


# ---------------------------------------------------------------------
# performs the l-th iteration of the second variant of the stockham FFT
# ---------------------------------------------------------------------
@njit
def fftz2(iss,
		l,
		m,
		n,
		ny,
		ny1,
		u, #dcomplex u[],
		x, #dcomplex x[][FFTBLOCKPAD]
		y): #dcomplex y[][FFTBLOCKPAD]

	# ---------------------------------------------------------------------
	# set initial parameters.
	# ---------------------------------------------------------------------
	n1 = int(n / 2)
	lk = 1 << (l - 1)
	li = 1 << (m - l)
	lj = 2 * lk
	ku = li

	cplx_conj = numpy.conj
	for i in range(li):
		i11 = i * lk
		i12 = i11 + n1
		i21 = i * lj
		i22 = i21 + lk
		u1 = complex(0.0, 0.0)
		if iss >= 1:
			u1 = u[ku+i]
		else:
			u1 = cplx_conj(u[ku+i])

		# ---------------------------------------------------------------------
		# this loop is vectorizable.
		# ---------------------------------------------------------------------
		for k in range(lk):
			for j in range(ny): 
				x11 = x[i11+k][j]
				x21 = x[i12+k][j]
				y[i21+k][j] = x11 + x21
				y[i22+k][j] = u1 * (x11 - x21)
	#END for i in range(li):
#END fftz2()


# ---------------------------------------------------------------------
# computes NY N-point complex-to-complex FFTs of X using an algorithm due
# to swarztrauber. X is both the input and the output array, while Y is a 
# scratch array. it is assumed that N = 2^M. before calling CFFTZ to 
# perform FFTs, the array U must be initialized by calling CFFTZ with is 
# set to 0 and M set to MX, where MX is the maximum value of M for any 
# subsequent call.
# ---------------------------------------------------------------------
@njit
def cfftz(iss,
		m,
		n,
		x, #dcomplex x[][FFTBLOCKPAD]
		y): #dcomplex y[][FFTBLOCKPAD]
	# ---------------------------------------------------------------------
	# check if input parameters are invalid.
	# ---------------------------------------------------------------------
	mx = int(u[0].real)
	if (iss != 1 and iss != -1) or m < 1 or m > mx:
		print("CFFTZ: Either U has not been initialized, or else\n"    
				"one of the input parameters is invalid", iss, m, mx)
		#sys.exit() #TODO: Search for an exit function supported by numba jit

	# ---------------------------------------------------------------------
	# perform one variant of the Stockham FFT.
	# ---------------------------------------------------------------------
	for l in range(1, m+1, 2):
		fftz2(iss, l, m, n, FFTBLOCK, FFTBLOCKPAD, u, x, y)
		if l == m:
			# ---------------------------------------------------------------------
			# copy Y to X.
			# ---------------------------------------------------------------------
			for j in range(n):
				for i in range(FFTBLOCK):
					x[j][i] = y[j][i]

			break

		fftz2(iss, l + 1, m, n, FFTBLOCK, FFTBLOCKPAD, u, y, x)
#END cfftz()


@njit
def cffts3(iss,
		d1,
		d2,
		d3,
		pointer_x,
		pointer_xout,
		y1, #dcomplex y1[][FFTBLOCKPAD],
		y2): #dcomplex y2[][FFTBLOCKPAD]){
	#dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	#dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;
	x = numpy.reshape(pointer_x, (NZ, NY, NX))
	xout = numpy.reshape(pointer_xout, (NZ, NY, NX))

	logd3 = ilog2(d3)

	#if timers_enabled: Not supported by @njit
	#	c_timers.timer_start(T_FFTZ)
	for j in range(d2):
		for ii in range(0, d1-FFTBLOCK+1, FFTBLOCK):
			for k in range(d3):
				for i in range(FFTBLOCK):
					y1[k][i] = x[k][j][i+ii]

			cfftz(iss, logd3, d3, y1, y2)
			for k in range(d3):
				for i in range(FFTBLOCK):
					xout[k][j][i+ii] = y1[k][i]
	#if timers_enabled: Not supported by @njit
	#	c_timers.timer_stop(T_FFTZ)
#END cffts3()


@njit
def cffts2(iss,
		d1,
		d2,
		d3,
		pointer_x,
		pointer_xout,
		y1, #dcomplex y1[][FFTBLOCKPAD],
		y2): #dcomplex y2[][FFTBLOCKPAD]
	#dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	#dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;
	x = numpy.reshape(pointer_x, (NZ, NY, NX))
	xout = numpy.reshape(pointer_xout, (NZ, NY, NX))

	logd2 = ilog2(d2)

	#if timers_enabled: Not supported by @njit
	#	c_timers.timer_start(T_FFTY)
	for k in range(d3):
		for ii in range(0, d1-FFTBLOCK+1, FFTBLOCK):
			for j in range(d2):
				for i in range(FFTBLOCK):
					y1[j][i] = x[k][j][i+ii]

			cfftz(iss, logd2, d2, y1, y2)
			for j in range(d2):
				for i in range(FFTBLOCK):
					xout[k][j][i+ii] = y1[j][i]
	#if timers_enabled: Not supported by @njit
	#	c_timers.timer_stop(T_FFTY)
#END cffts2()


@njit
def cffts1(iss,
		d1,
		d2,
		d3,
		pointer_x,
		pointer_xout,
		y1, #dcomplex y1[][FFTBLOCKPAD]
		y2): #dcomplex y2[][FFTBLOCKPAD]
	#dcomplex (*x)[NY][NX] = (dcomplex(*)[NY][NX])pointer_x;
	#dcomplex (*xout)[NY][NX] = (dcomplex(*)[NY][NX])pointer_xout;
	x = numpy.reshape(pointer_x, (NZ, NY, NX))
	xout = numpy.reshape(pointer_xout, (NZ, NY, NX))

	logd1 = ilog2(d1)

	#if timers_enabled: Not supported by @njit
	#	c_timers.timer_start(T_FFTX)
	for k in range(d3):
		for jj in range(0, d2-FFTBLOCK+1, FFTBLOCK):
			for j in range(FFTBLOCK):
				for i in range(d1):
					y1[i][j] = x[k][j+jj][i]

			cfftz(iss, logd1, d1, y1, y2)
			for j in range(FFTBLOCK):
				for i in range(d1):
					xout[k][j+jj][i] = y1[i][j]

	#if timers_enabled: Not supported by @njit 
	#	c_timers.timer_stop(T_FFTX)
#END cffts1()


@njit
def fft(dirr,
		pointer_x1,
		pointer_x2):
	y1 = numpy.empty(shape=(npbparams.MAXDIM, FFTBLOCKPAD), dtype=numpy.complex128)
	y2 = numpy.empty(shape=(npbparams.MAXDIM, FFTBLOCKPAD), dtype=numpy.complex128)

	# ---------------------------------------------------------------------
	# note: args x1, x2 must be different arrays
	# note: args for cfftsx are (direction, layout, xin, xout, scratch)
	# xin/xout may be the same and it can be somewhat faster
	# if they are
	# ---------------------------------------------------------------------
	if dirr == 1:
		cffts1(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				y1, y2)
		cffts2(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				y1, y2)
		cffts3(1, dims[0], dims[1], dims[2], pointer_x1, pointer_x2,
				y1, y2)
	else:
		cffts3(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				y1, y2)
		cffts2(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x1,
				y1, y2)
		cffts1(-1, dims[0], dims[1], dims[2], pointer_x1, pointer_x2,
				y1, y2)
	#END if dirr == 1:
#END fft()


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


# ---------------------------------------------------------------------
# compute the roots-of-unity array that will be used for subsequent FFTs. 
# ---------------------------------------------------------------------
@njit
def fft_init(n, u):
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
#END fft_init()


# ---------------------------------------------------------------------
# compute a^exponent mod 2^46
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# fill in array u0 with initial conditions from 
# random number generator 
# ---------------------------------------------------------------------
@njit
def compute_initial_conditions(pointer_u0,
		d1,
		d2, #dims[1]
		d3): #dims[2]
	#dcomplex (*u0)[NY][NX] = (dcomplex(*)[NY][NX])pointer_u0;
	u0 = numpy.reshape(pointer_u0, (NZ, NY, NX))

	starts = numpy.empty(NZ, numpy.float64)
	start = SEED

	# ---------------------------------------------------------------------
	# jump to the starting element for our first plane.
	# ---------------------------------------------------------------------
	an = ipow46(A, 0)
	aux, start = randlc(start, an)
	an = ipow46(A, 2*NX*NY)

	starts[0] = start
	for k in range(1, dims[2]):
		aux, start = randlc(start, an)
		starts[k] = start
		
	# ---------------------------------------------------------------------
	# go through by z planes filling in one square at a time.
	# ---------------------------------------------------------------------
	for k in range(dims[2]):
		x0 = starts[k]
		for j in range(dims[1]):
			#vranlc(2*NX, &x0, A, (double*)&u0[k][j][0]);
			idx = (k*NY+j)*NX + 0 
			x0 = vranlc_complex(2 * NX, x0, A, pointer_u0[idx:])
#END compute_initial_conditions()


# compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
# for time evolution exponent.
@njit
def compute_indexmap(pointer_twiddle,
					d1, #NX
					d2, #NY
					d3): #NZ
	#double (*twiddle)[NY][NX] = (double(*)[NY][NX])pointer_twiddle;
	twiddle = numpy.reshape(pointer_twiddle, (NZ, NY, NX))

	# ---------------------------------------------------------------------
	# basically we want to convert the fortran indices 
	# 1 2 3 4 5 6 7 8 
	# to 
	# 0 1 2 3 -4 -3 -2 -1
	# the following magic formula does the trick:
	# mod(i-1+n/2, n) - n/2
	# ---------------------------------------------------------------------
	m_exp = math.exp
	ap = - 4.0 * ALPHA * PI * PI
	for k in range(d3):
		kk = int( ((k+NZ/2) % NZ) - NZ/2 )
		kk2 = kk * kk
		for j in range(d2):
			jj = int( ((j+NY/2) % NY) - NY/2 )
			kj2 = jj * jj + kk2
			for i in range(d1):
				ii = int( ((i+NX/2) % NX) - NX/2 )
				twiddle[k][j][i] = m_exp(ap * (ii*ii+kj2))
#END compute_indexmap()


def setup(): 
	global timers_enabled
	global niter
	global dims
	
	timers_enabled = os.path.isfile("timer.flag")

	niter = npbparams.NITER_DEFAULT

	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - FT Benchmark\n")
	print(" Size                : %4dx%4dx%4d" % (NX, NY, NZ))
	print(" Iterations                  :%7d" % (niter))
	print()

	dims[0] = NX
	dims[1] = NY
	dims[2] = NZ
#END setup()


def main():
	global sums, twiddle, u, u0, u1
	
	# ---------------------------------------------------------------------
	# run the entire problem once to make sure all data is touched. 
	# this reduces variable startup costs, which is important for such a 
	# short benchmark. the other NPB 2 implementations are similar. 
	# ---------------------------------------------------------------------
	for i in range(T_MAX):
		c_timers.timer_clear(i)
	
	setup()
	#init_ui(u0, u1, twiddle, dims[0], dims[1], dims[2]) Already innitialized with zeros
	compute_indexmap(twiddle, dims[0], dims[1], dims[2])
	compute_initial_conditions(u1, dims[0], dims[1], dims[2])
	fft_init(npbparams.MAXDIM, u)
	fft(1, u1, u0)
	
	# ---------------------------------------------------------------------
	# start over from the beginning. note that all operations must
	# be timed, in contrast to other benchmarks. 
	# ---------------------------------------------------------------------
	for i in range(T_MAX):
		c_timers.timer_clear(i)
	
	c_timers.timer_start(T_TOTAL)
	if timers_enabled:
		c_timers.timer_start(T_SETUP)

	compute_indexmap(twiddle, dims[0], dims[1], dims[2])

	compute_initial_conditions(u1, dims[0], dims[1], dims[2])

	fft_init(npbparams.MAXDIM, u)

	if timers_enabled:
		c_timers.timer_stop(T_SETUP)
	if timers_enabled:
		c_timers.timer_start(T_FFT)
	fft(1, u1, u0)
	if timers_enabled:
		c_timers.timer_stop(T_FFT)

	for it in range(1, niter+1):
		if timers_enabled:
			c_timers.timer_start(T_EVOLVE)
		evolve(u0, u1, twiddle, dims[0], dims[1], dims[2])
		if timers_enabled:
			c_timers.timer_stop(T_EVOLVE)
		if timers_enabled:
			c_timers.timer_start(T_FFT)
		fft(-1, u1, u1)
		if timers_enabled:
			c_timers.timer_stop(T_FFT)
		if timers_enabled:
			c_timers.timer_start(T_CHECKSUM)
		checksum(it, u1, dims[0], dims[1], dims[2], sums)
		if timers_enabled:
			c_timers.timer_stop(T_CHECKSUM)

	verified = verify(NX, NY, NZ, niter)

	c_timers.timer_stop(T_TOTAL)
	total_time = c_timers.timer_read(T_TOTAL)

	mflops = 0.0
	if total_time != 0.0:
		mflops = ( 1.0e-6 * npbparams.NTOTAL *
			(14.8157 + 7.19641 * math.log(npbparams.NTOTAL)
			 + (5.23518 + 7.21113 * math.log(npbparams.NTOTAL)) * niter)
			/ total_time )
	
	c_print_results.c_print_results("FT",
			npbparams.CLASS,
			npbparams.NX, 
			npbparams.NY,
			npbparams.NZ,
			niter,
			total_time,
			mflops,
			"          floating point",
			verified)

	if timers_enabled:
		print_timers()
#END main()

#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER FT')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_ft_info(args.CLASS)
	set_global_variables()
	
	main()
