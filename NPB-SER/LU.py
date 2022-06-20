# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
#Authors of the Fortran code:
#	S. Weeratunga
#	V. Venkatakrishnan
#	E. Barszcz
#	M. Yarrow
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
import c_timers
import c_print_results

# ---------------------------------------------------------------------
# driver for the performance evaluation of the solver for
# five coupled parabolic/elliptic partial differential equations
# ---------------------------------------------------------------------
# parameters which can be overridden in runtime config file
# isiz1,isiz2,isiz3 give the maximum size
# ipr = 1 to print out verbose information
# omega = 2.0 is correct for all classes
# tolrsd is tolerance levels for steady state residuals
# ---------------------------------------------------------------------
# field variables and residuals
# to improve cache performance, second two dimensions padded by 1 
# for even number sizes only.
# note: corresponding array (called "v") in routines blts, buts, 
# and l2norm are similarly padded
# ---------------------------------------------------------------------

# Global variables
IPR_DEFAULT = 1
OMEGA_DEFAULT = 1.2
TOLRSD1_DEF = 1.0e-08
TOLRSD2_DEF = 1.0e-08
TOLRSD3_DEF = 1.0e-08
TOLRSD4_DEF = 1.0e-08
TOLRSD5_DEF = 1.0e-08
C1 = 1.40e+00
C2 = 0.40e+00
C3 = 1.00e-01
C4 = 1.00e+00
C5 = 1.40e+00
T_TOTAL = 1
T_RHSX = 2
T_RHSY = 3
T_RHSZ = 4
T_RHS = 5
T_JACLD = 6
T_BLTS = 7
T_JACU = 8
T_BUTS = 9
T_ADD = 10
T_L2NORM = 11
T_LAST = 11

u = None
rsd = None
frct = None
flux = None
qs = None
rho_i = None
a = None
b = None
c = None
d = None
ce = numpy.empty((13, 5), dtype=numpy.float64())

# grid 
dxi, deta, dzeta = 0.0, 0.0, 0.0
tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
nx, ny, nz = 0, 0, 0
nx0, ny0, nz0 = 0, 0, 0
ist, iend = 0, 0
jst, jend = 0, 0
ii1, ii2 = 0, 0
ji1, ji2 = 0, 0
ki1, ki2 = 0, 0
# dissipation
dx1, dx2, dx3, dx4, dx5 = 0.0, 0.0, 0.0, 0.0, 0.0
dy1, dy2, dy3, dy4, dy5 = 0.0, 0.0, 0.0, 0.0, 0.0
dz1, dz2, dz3, dz4, dz5 = 0.0, 0.0, 0.0, 0.0, 0.0
dssp = 0.0
# output control parameters */
ipr, inorm = 0, 0
# newton-raphson iteration control parameters
dt, omega = 0.0, 0.0 
tolrsd = numpy.empty(5, dtype=numpy.float64) 
rsdnm = numpy.empty(5, dtype=numpy.float64) 
errnm = numpy.empty(5, dtype=numpy.float64) 
frc, ttotal = 0.0, 0.0 
itmax, invert = 0, 0

# timer
maxtime = 1.0
timeron = False


def set_global_variables():
	global us, vs, ws, qs, rho_i, square
	global forcing, u, rhs
	global cuf, q, ue, buf
	global fjac, njac
	global lhs
	
	global u, rsd, frct
	global flux
	global qs, rho_i
	global a, b, c, d
	
	u = numpy.zeros((npbparams.ISIZ3, int(npbparams.ISIZ2/2*2+1), int(npbparams.ISIZ1/2*2+1), 5), dtype=numpy.float64())
	rsd = numpy.zeros((npbparams.ISIZ3, int(npbparams.ISIZ2/2*2+1), int(npbparams.ISIZ1/2*2+1), 5), dtype=numpy.float64())
	frct = numpy.zeros((npbparams.ISIZ3, int(npbparams.ISIZ2/2*2+1), int(npbparams.ISIZ1/2*2+1), 5), dtype=numpy.float64())
	
	flux  = numpy.zeros((npbparams.ISIZ1, 5), dtype=numpy.float64())
	
	qs = numpy.zeros((npbparams.ISIZ3, int(npbparams.ISIZ2/2*2+1), int(npbparams.ISIZ1/2*2+1)), dtype=numpy.float64())
	rho_i = numpy.zeros((npbparams.ISIZ3, int(npbparams.ISIZ2/2*2+1), int(npbparams.ISIZ1/2*2+1)), dtype=numpy.float64())
	
	a = numpy.zeros((npbparams.ISIZ2, int(npbparams.ISIZ1/2*2+1), 5, 5), dtype=numpy.float64())
	b = numpy.zeros((npbparams.ISIZ2, int(npbparams.ISIZ1/2*2+1), 5, 5), dtype=numpy.float64())
	c = numpy.zeros((npbparams.ISIZ2, int(npbparams.ISIZ1/2*2+1), 5, 5), dtype=numpy.float64())
	d = numpy.zeros((npbparams.ISIZ2, int(npbparams.ISIZ1/2*2+1), 5, 5), dtype=numpy.float64())
#END set_global_variables()


# ---------------------------------------------------------------------
# verification routine                         
# ---------------------------------------------------------------------
def verify(xcr, 
		xce,
		xci):
	dtref = 0.0
	# ---------------------------------------------------------------------
	# tolerance level
	# ---------------------------------------------------------------------
	epsilon = 1.0e-08
	verified = True
	xcrref = numpy.repeat(1.0, 5)
	xceref = numpy.repeat(1.0, 5)
	xciref = 1.0
	
	if npbparams.CLASS == 'S':
		dtref = 5.0e-1
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (12X12X12) grid,
		# after 50 time steps, with DT = 5.0d-01
		# ---------------------------------------------------------------------
		xcrref[0] = 1.6196343210976702e-02
		xcrref[1] = 2.1976745164821318e-03
		xcrref[2] = 1.5179927653399185e-03
		xcrref[3] = 1.5029584435994323e-03
		xcrref[4] = 3.4264073155896461e-02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (12X12X12) grid,
		# after 50 time steps, with DT = 5.0d-01
		# ---------------------------------------------------------------------
		xceref[0] = 6.4223319957960924e-04
		xceref[1] = 8.4144342047347926e-05
		xceref[2] = 5.8588269616485186e-05
		xceref[3] = 5.8474222595157350e-05
		xceref[4] = 1.3103347914111294e-03
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (12X12X12) grid,
		# after 50 time steps, with DT = 5.0d-01
		# ---------------------------------------------------------------------
		xciref = 7.8418928865937083e+00
	elif npbparams.CLASS == 'W':
		dtref = 1.5e-3
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (33x33x33) grid,
		# after 300 time steps, with DT = 1.5d-3
		# ---------------------------------------------------------------------
		xcrref[0] = 0.1236511638192e+02
		xcrref[1] = 0.1317228477799e+01
		xcrref[2] = 0.2550120713095e+01
		xcrref[3] = 0.2326187750252e+01
		xcrref[4] = 0.2826799444189e+02
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (33X33X33) grid,
		# ---------------------------------------------------------------------
		xceref[0] = 0.4867877144216e+00
		xceref[1] = 0.5064652880982e-01
		xceref[2] = 0.9281818101960e-01
		xceref[3] = 0.8570126542733e-01
		xceref[4] = 0.1084277417792e+01
		# ---------------------------------------------------------------------
		# rReference value of surface integral, for the (33X33X33) grid,
		# after 300 time steps, with DT = 1.5d-3
		# ---------------------------------------------------------------------
		xciref = 0.1161399311023e+02
	elif npbparams.CLASS == 'A':
		dtref = 2.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (64X64X64) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 7.7902107606689367e+02
		xcrref[1] = 6.3402765259692870e+01
		xcrref[2] = 1.9499249727292479e+02
		xcrref[3] = 1.7845301160418537e+02
		xcrref[4] = 1.8384760349464247e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (64X64X64) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 2.9964085685471943e+01
		xceref[1] = 2.8194576365003349e+00
		xceref[2] = 7.3473412698774742e+00
		xceref[3] = 6.7139225687777051e+00
		xceref[4] = 7.0715315688392578e+01
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (64X64X64) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xciref = 2.6030925604886277e+01
	elif npbparams.CLASS == 'B':
		dtref = 2.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (102X102X102) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 3.5532672969982736e+03
		xcrref[1] = 2.6214750795310692e+02
		xcrref[2] = 8.8333721850952190e+02
		xcrref[3] = 7.7812774739425265e+02
		xcrref[4] = 7.3087969592545314e+03
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (102X102X102) 
		# grid, after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 1.1401176380212709e+02
		xceref[1] = 8.1098963655421574e+00
		xceref[2] = 2.8480597317698308e+01
		xceref[3] = 2.5905394567832939e+01
		xceref[4] = 2.6054907504857413e+02
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (102X102X102) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xciref = 4.7887162703308227e+01
	elif npbparams.CLASS == 'C':
		dtref = 2.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (162X162X162) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 1.03766980323537846e+04
		xcrref[1] = 8.92212458801008552e+02
		xcrref[2] = 2.56238814582660871e+03
		xcrref[3] = 2.19194343857831427e+03
		xcrref[4] = 1.78078057261061185e+04
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (162X162X162) 
		# grid, after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 2.15986399716949279e+02
		xceref[1] = 1.55789559239863600e+01
		xceref[2] = 5.41318863077207766e+01
		xceref[3] = 4.82262643154045421e+01
		xceref[4] = 4.55902910043250358e+02
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (162X162X162) grid,
		# after 250 time steps, with DT = 2.0d+00
		# ---------------------------------------------------------------------
		xciref = 6.66404553572181300e+01
	elif npbparams.CLASS == 'D':
		dtref = 1.0e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (408X408X408) grid,
		# after 300 time steps, with DT = 1.0d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 0.4868417937025e+05
		xcrref[1] = 0.4696371050071e+04
		xcrref[2] = 0.1218114549776e+05
		xcrref[3] = 0.1033801493461e+05
		xcrref[4] = 0.7142398413817e+05
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (408X408X408) 
		# grid, after 300 time steps, with DT = 1.0d+00
		# ---------------------------------------------------------------------
		xceref[0] = 0.3752393004482e+03
		xceref[1] = 0.3084128893659e+02
		xceref[2] = 0.9434276905469e+02
		xceref[3] = 0.8230686681928e+02
		xceref[4] = 0.7002620636210e+03
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (408X408X408) grid,
		# after 300 time steps, with DT = 1.0d+00
		# ---------------------------------------------------------------------
		xciref = 0.8334101392503e+02
	elif npbparams.CLASS == 'E':
		dtref = 0.5e+0
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of residual, for the (1020X1020X1020) grid,
		# after 300 time steps, with DT = 0.5d+00
		# ---------------------------------------------------------------------
		xcrref[0] = 0.2099641687874e+06
		xcrref[1] = 0.2130403143165e+05
		xcrref[2] = 0.5319228789371e+05
		xcrref[3] = 0.4509761639833e+05
		xcrref[4] = 0.2932360006590e+06
		# ---------------------------------------------------------------------
		# reference values of RMS-norms of solution error, for the (1020X1020X1020) 
		# grid, after 300 time steps, with DT = 0.5d+00
		# ---------------------------------------------------------------------
		xceref[0] = 0.4800572578333e+03
		xceref[1] = 0.4221993400184e+02
		xceref[2] = 0.1210851906824e+03
		xceref[3] = 0.1047888986770e+03
		xceref[4] = 0.8363028257389e+03
		# ---------------------------------------------------------------------
		# reference value of surface integral, for the (1020X1020X1020) grid,
		# after 300 time steps, with DT = 0.5d+00
		# ---------------------------------------------------------------------
		xciref = 0.9512163272273e+02
	else:
		verified = False
		
	# ---------------------------------------------------------------------
	# verification test for residuals if gridsize is one of 
	# the defined grid sizes above (class .ne. 'U')
	# ---------------------------------------------------------------------
	# compute the difference of solution values and the known reference values.
	# ---------------------------------------------------------------------
	xcrdif = numpy.empty(5, dtype=numpy.float64)
	xcedif = numpy.empty(5, dtype=numpy.float64)
	for m in range(5):
		xcrdif[m] = abs((xcr[m]-xcrref[m]) / xcrref[m])
		xcedif[m] = abs((xce[m]-xceref[m]) / xceref[m])

	xcidif = abs((xci-xciref)/xciref)
	# ---------------------------------------------------------------------
	# output the comparison of computed results to known cases.
	# ---------------------------------------------------------------------
	print("\n Verification being performed for class_npb %c" % (npbparams.CLASS))
	print(" accuracy setting for epsilon = %20.13E" % (epsilon))
	verified = abs(dt-dtref) <= epsilon
	if not verified:
		print(" DT does not match the reference value of %15.8E" % (dtref))
	
	print(" Comparison of RMS-norms of residual")
	for m in range(5):
		if xcrdif[m] <= epsilon:
			print("          %2d  %20.13E%20.13E%20.13E" % (m+1, xcr[m], xcrref[m], xcrdif[m]))
		else:
			verified = False
			print(" FAILURE: %2d  %20.13E%20.13E%20.13E" % (m+1, xcr[m], xcrref[m], xcrdif[m]))
	
	print(" Comparison of RMS-norms of solution error")
	for m in range(5):
		if xcedif[m] <= epsilon:
			print("          %2d  %20.13E%20.13E%20.13E" % (m+1, xce[m], xceref[m], xcedif[m]))
		else:
			verified = False
			print(" FAILURE: %2d  %20.13E%20.13E%20.13E" % (m+1, xce[m], xceref[m], xcedif[m]))
	
	print(" Comparison of surface integral")
	if xcidif <= epsilon:
		print("              %20.13E%20.13E%20.13E" % (xci, xciref, xcidif))
	else:
		verified = False
		print(" FAILURE:     %20.13E%20.13E%20.13E" % (xci, xciref, xcidif))

	if verified:
		print(" Verification Successful")
	else:
		print(" Verification failed")
	
	return verified
#END verify()


@njit
def pintgr(u):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	phi1 = numpy.empty((npbparams.ISIZ3+2, npbparams.ISIZ2+2), dtype=numpy.float64)
	phi2 = numpy.empty((npbparams.ISIZ3+2, npbparams.ISIZ2+2), dtype=numpy.float64)
	# ---------------------------------------------------------------------
	# set up the sub-domains for integeration in each processor
	# ---------------------------------------------------------------------
	ibeg = ii1
	ifin = ii2
	jbeg = ji1
	jfin = ji2
	ifin1 = ifin-1
	jfin1 = jfin-1
	# ---------------------------------------------------------------------
	# initialize
	# ---------------------------------------------------------------------
	for i in range(npbparams.ISIZ2+2):
		for k in range(npbparams.ISIZ3+2):
			phi1[k][i] = 0.0
			phi2[k][i] = 0.0

	for j in range(jbeg, jfin): 
		for i in range(ibeg, ifin):
			k = ki1
			phi1[j][i] = ( C2*(u[k][j][i][4]
					-0.50*(u[k][j][i][1]*u[k][j][i][1]
						+u[k][j][i][2]*u[k][j][i][2]
						+u[k][j][i][3]*u[k][j][i][3])
					/u[k][j][i][0]) )
			k = ki2-1
			phi2[j][i] = ( C2*(u[k][j][i][4]
					-0.50*(u[k][j][i][1]*u[k][j][i][1]
						+u[k][j][i][2]*u[k][j][i][2]
						+u[k][j][i][3]*u[k][j][i][3])
					/u[k][j][i][0]) )

	frc1 = 0.0
	for j in range(jbeg, jfin1):
		for i in range(ibeg, ifin1):
			frc1 = ( frc1+(phi1[j][i]
					+phi1[j][i+1]
					+phi1[j+1][i]
					+phi1[j+1][i+1]
					+phi2[j][i]
					+phi2[j][i+1]
					+phi2[j+1][i]
					+phi2[j+1][i+1]) )

	frc1 = dxi*deta*frc1
	# ---------------------------------------------------------------------
	# initialize
	# ---------------------------------------------------------------------
	for i in range(npbparams.ISIZ2+2):
		for k in range(npbparams.ISIZ3+2):
			phi1[k][i] = 0.0
			phi2[k][i] = 0.0

	if jbeg == ji1:
		for k in range(ki1, ki2): 
			for i in range(ibeg, ifin):
				phi1[k][i] = ( C2*(u[k][jbeg][i][4]
						-0.50*(u[k][jbeg][i][1]*u[k][jbeg][i][1]
							+u[k][jbeg][i][2]*u[k][jbeg][i][2]
							+u[k][jbeg][i][3]*u[k][jbeg][i][3])
						/u[k][jbeg][i][0]) )

	if jfin == ji2:
		for k in range(ki1, ki2):
			for i in range(ibeg, ifin):
				phi2[k][i] = ( C2*(u[k][jfin-1][i][4]
						-0.50*(u[k][jfin-1][i][1]*u[k][jfin-1][i][1]
							+u[k][jfin-1][i][2]*u[k][jfin-1][i][2]
							+u[k][jfin-1][i][3]*u[k][jfin-1][i][3])
						/u[k][jfin-1][i][0]) )

	frc2 = 0.0
	for k in range(ki1, ki2-1):
		for i in range(ibeg, ifin1):
			frc2 = ( frc2+(phi1[k][i]
					+phi1[k][i+1]
					+phi1[k+1][i]
					+phi1[k+1][i+1]
					+phi2[k][i]
					+phi2[k][i+1]
					+phi2[k+1][i]
					+phi2[k+1][i+1]) )

	frc2 = dxi*dzeta*frc2
	# ---------------------------------------------------------------------
	# initialize
	# ---------------------------------------------------------------------
	for i in range(npbparams.ISIZ2+2):
		for k in range(npbparams.ISIZ3+2):
			phi1[k][i] = 0.0
			phi2[k][i] = 0.0

	if ibeg == ii1:
		for k in range(ki1, ki2):
			for j in range(jbeg, jfin):
				phi1[k][j] = ( C2*(u[k][j][ibeg][4]
						-0.50*(u[k][j][ibeg][1]*u[k][j][ibeg][1]
							+u[k][j][ibeg][2]*u[k][j][ibeg][2]
							+u[k][j][ibeg][3]*u[k][j][ibeg][3])
						/u[k][j][ibeg][0]) )

	if ifin == ii2:
		for k in range(ki1, ki2):
			for j in range(jbeg, jfin):
				phi2[k][j] = ( C2*(u[k][j][ifin-1][4]
						-0.50*(u[k][j][ifin-1][1]*u[k][j][ifin-1][1]
							+u[k][j][ifin-1][2]*u[k][j][ifin-1][2]
							+u[k][j][ifin-1][3]*u[k][j][ifin-1][3])
						/u[k][j][ifin-1][0]) )

	frc3 = 0.0
	for k in range(ki1, ki2-1):
		for j in range(jbeg, jfin1):
			frc3 = ( frc3+(phi1[k][j]
					+phi1[k][j+1]
					+phi1[k+1][j]
					+phi1[k+1][j+1]
					+phi2[k][j]
					+phi2[k][j+1]
					+phi2[k+1][j]
					+phi2[k+1][j+1]) )

	frc3 = deta*dzeta*frc3
	return ( 0.25*(frc1+frc2+frc3) ) #frc
#END pintgr()


# ---------------------------------------------------------------------
# compute the solution error
# ---------------------------------------------------------------------
def error(errnm, u):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	u000ijk = numpy.empty(5, dtype=numpy.float64)
	for m in range(5): 
		errnm[m] = 0.0
	for k in range(1, nz-1):
		for j in range(jst, jend):
			for i in range(ist, iend):
				exact(i, j, k, u000ijk)
				for m in range(5): 
					tmp = (u000ijk[m]-u[k][j][i][m])
					errnm[m] = errnm[m]+tmp*tmp

	m_sqrt = math.sqrt
	for m in range(5): 
		errnm[m] = m_sqrt(errnm[m]/((nx0-2)*(ny0-2)*(nz0-2)))
#END error()


# ---------------------------------------------------------------------
# compute the regular-sparse, block upper triangular solution:
# v <-- ( U-inv ) * v
# ---------------------------------------------------------------------
# to improve cache performance, second two dimensions padded by 1 
# for even number sizes only. only needed in v.
# ---------------------------------------------------------------------
@njit
def buts(nx,
		ny,
		nz,
		k,
		omega,
		v, #double v[][ISIZ2/2*2+1][ISIZ1/2*2+1][5],
		tv, #void* pointer_tv,
		d, #double d[][ISIZ1/2*2+1][5][5],
		udx, #double udx[][ISIZ1/2*2+1][5][5],
		udy, #double udy[][ISIZ1/2*2+1][5][5],
		udz, #double udz[][ISIZ1/2*2+1][5][5],
		ist,
		iend,
		jst,
		jend,
		nx0,
		ny0):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	tmat = numpy.empty((5, 5), dtype=numpy.float64)
	for j in range(jend-1, jst-1, -1):
		for i in range(iend-1, ist-1, -1):
			for m in range(5):
				tv[j][i][m] = ( 
					omega*(udz[j][i][0][m]*v[k+1][j][i][0]
							+udz[j][i][1][m]*v[k+1][j][i][1]
							+udz[j][i][2][m]*v[k+1][j][i][2]
							+udz[j][i][3][m]*v[k+1][j][i][3]
							+udz[j][i][4][m]*v[k+1][j][i][4]) )

	for j in range(jend-1, jst-1, -1):
		for i in range(iend-1, ist-1, -1):
			for m in range(5):
				tv[j][i][m] = ( tv[j][i][m]
					+omega*(udy[j][i][0][m]*v[k][j+1][i][0]
							+udx[j][i][0][m]*v[k][j][i+1][0]
							+udy[j][i][1][m]*v[k][j+1][i][1]
							+udx[j][i][1][m]*v[k][j][i+1][1]
							+udy[j][i][2][m]*v[k][j+1][i][2]
							+udx[j][i][2][m]*v[k][j][i+1][2]
							+udy[j][i][3][m]*v[k][j+1][i][3]
							+udx[j][i][3][m]*v[k][j][i+1][3]
							+udy[j][i][4][m]*v[k][j+1][i][4]
							+udx[j][i][4][m]*v[k][j][i+1][4]) )

			# ---------------------------------------------------------------------
			# diagonal block inversion
			# ---------------------------------------------------------------------
			for m in range(5):
				tmat[0][m] = d[j][i][0][m]
				tmat[1][m] = d[j][i][1][m]
				tmat[2][m] = d[j][i][2][m]
				tmat[3][m] = d[j][i][3][m]
				tmat[4][m] = d[j][i][4][m]

			#
			tmp1 = 1.0/tmat[0][0]
			tmp = tmp1*tmat[0][1]
			tmat[1][1] = tmat[1][1]-tmp*tmat[1][0]
			tmat[2][1] = tmat[2][1]-tmp*tmat[2][0]
			tmat[3][1] = tmat[3][1]-tmp*tmat[3][0]
			tmat[4][1] = tmat[4][1]-tmp*tmat[4][0]
			tv[j][i][1] = tv[j][i][1]-tv[j][i][0]*tmp
			#
			tmp = tmp1*tmat[0][2]
			tmat[1][2] = tmat[1][2]-tmp*tmat[1][0]
			tmat[2][2] = tmat[2][2]-tmp*tmat[2][0]
			tmat[3][2] = tmat[3][2]-tmp*tmat[3][0]
			tmat[4][2] = tmat[4][2]-tmp*tmat[4][0]
			tv[j][i][2] = tv[j][i][2]-tv[j][i][0]*tmp
			#
			tmp = tmp1*tmat[0][3]
			tmat[1][3] = tmat[1][3]-tmp*tmat[1][0]
			tmat[2][3] = tmat[2][3]-tmp*tmat[2][0]
			tmat[3][3] = tmat[3][3]-tmp*tmat[3][0]
			tmat[4][3] = tmat[4][3]-tmp*tmat[4][0]
			tv[j][i][3] = tv[j][i][3]-tv[j][i][0]*tmp
			#
			tmp = tmp1*tmat[0][4]
			tmat[1][4] = tmat[1][4]-tmp*tmat[1][0]
			tmat[2][4] = tmat[2][4]-tmp*tmat[2][0]
			tmat[3][4] = tmat[3][4]-tmp*tmat[3][0]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][0]
			tv[j][i][4] = tv[j][i][4]-tv[j][i][0]*tmp
			#
			tmp1 = 1.0/tmat[1][1]
			tmp = tmp1*tmat[1][2]
			tmat[2][2] = tmat[2][2]-tmp*tmat[2][1]
			tmat[3][2] = tmat[3][2]-tmp*tmat[3][1]
			tmat[4][2] = tmat[4][2]-tmp*tmat[4][1]
			tv[j][i][2] = tv[j][i][2]-tv[j][i][1]*tmp
			#
			tmp = tmp1*tmat[1][3]
			tmat[2][3] = tmat[2][3]-tmp*tmat[2][1]
			tmat[3][3] = tmat[3][3]-tmp*tmat[3][1]
			tmat[4][3] = tmat[4][3]-tmp*tmat[4][1]
			tv[j][i][3] = tv[j][i][3]-tv[j][i][1]*tmp
			#
			tmp = tmp1*tmat[1][4]
			tmat[2][4] = tmat[2][4]-tmp*tmat[2][1]
			tmat[3][4] = tmat[3][4]-tmp*tmat[3][1]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][1]
			tv[j][i][4] = tv[j][i][4]-tv[j][i][1]*tmp
			#
			tmp1 = 1.0/tmat[2][2]
			tmp = tmp1*tmat[2][3]
			tmat[3][3] = tmat[3][3]-tmp*tmat[3][2]
			tmat[4][3] = tmat[4][3]-tmp*tmat[4][2]
			tv[j][i][3] = tv[j][i][3]-tv[j][i][2]*tmp
			#
			tmp = tmp1*tmat[2][4]
			tmat[3][4] = tmat[3][4]-tmp*tmat[3][2]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][2]
			tv[j][i][4] = tv[j][i][4]-tv[j][i][2]*tmp
			#
			tmp1 = 1.0/tmat[3][3]
			tmp = tmp1*tmat[3][4]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][3]
			tv[j][i][4] = tv[j][i][4]-tv[j][i][3]*tmp
			# ---------------------------------------------------------------------
			# back substitution
			# ---------------------------------------------------------------------
			tv[j][i][4] = tv[j][i][4]/tmat[4][4]
			tv[j][i][3] = tv[j][i][3]-tmat[4][3]*tv[j][i][4]
			tv[j][i][3] = tv[j][i][3]/tmat[3][3]
			tv[j][i][2] = ( tv[j][i][2]
				-tmat[3][2]*tv[j][i][3]
				-tmat[4][2]*tv[j][i][4] )
			tv[j][i][2] = tv[j][i][2]/tmat[2][2]
			tv[j][i][1] = ( tv[j][i][1]
				-tmat[2][1]*tv[j][i][2]
				-tmat[3][1]*tv[j][i][3]
				-tmat[4][1]*tv[j][i][4] )
			tv[j][i][1] = tv[j][i][1]/tmat[1][1]
			tv[j][i][0] = ( tv[j][i][0]
				-tmat[1][0]*tv[j][i][1]
				-tmat[2][0]*tv[j][i][2]
				-tmat[3][0]*tv[j][i][3]
				-tmat[4][0]*tv[j][i][4] )
			tv[j][i][0] = tv[j][i][0]/tmat[0][0]
			v[k][j][i][0] = v[k][j][i][0]-tv[j][i][0]
			v[k][j][i][1] = v[k][j][i][1]-tv[j][i][1]
			v[k][j][i][2] = v[k][j][i][2]-tv[j][i][2]
			v[k][j][i][3] = v[k][j][i][3]-tv[j][i][3]
			v[k][j][i][4] = v[k][j][i][4]-tv[j][i][4]
		#END for i in range(iend-1, ist-1, -1):
	#END for j in range(jend-1, jst-1, -1):
#END buts()


# ---------------------------------------------------------------------
# compute the upper triangular part of the jacobian matrix
# ---------------------------------------------------------------------
@njit
def jacu(k,
		a, b, c, d,
		u, rho_i, qs):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	r43 = (4.0/3.0)
	c1345 = C1*C3*C4*C5
	c34 = C3*C4
	for j in range(jst, jend):
		for i in range(ist, iend):
			# ---------------------------------------------------------------------
			# form the block daigonal
			# ---------------------------------------------------------------------
			tmp1=rho_i[k][j][i]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			d[j][i][0][0] = 1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1)
			d[j][i][1][0] = 0.0
			d[j][i][2][0] = 0.0
			d[j][i][3][0] = 0.0
			d[j][i][4][0] = 0.0
			d[j][i][0][1] = ( dt*2.0
				*(-tx1*r43-ty1-tz1)
				*(c34*tmp2*u[k][j][i][1]) )
			d[j][i][1][1] = ( 1.0
				+dt*2.0*c34*tmp1 
				*(tx1*r43+ty1+tz1)
				+dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2) )
			d[j][i][2][1] = 0.0
			d[j][i][3][1] = 0.0
			d[j][i][4][1] = 0.0
			d[j][i][0][2] = ( dt*2.0
				*(-tx1-ty1*r43-tz1)
				*(c34*tmp2*u[k][j][i][2]) )
			d[j][i][1][2] = 0.0
			d[j][i][2][2] = ( 1.0
				+dt*2.0*c34*tmp1
				*(tx1+ty1*r43+tz1)
				+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3) )
			d[j][i][3][2] = 0.0
			d[j][i][4][2] = 0.0
			d[j][i][0][3] = ( dt*2.0
				*(-tx1-ty1-tz1*r43)
				*(c34*tmp2*u[k][j][i][3]) )
			d[j][i][1][3] = 0.0
			d[j][i][2][3] = 0.0
			d[j][i][3][3] = ( 1.0
				+dt*2.0*c34*tmp1
				*(tx1+ty1+tz1*r43)
				+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4) )
			d[j][i][4][3] = 0.0
			d[j][i][0][4] = ( -dt*2.0
				*(((tx1*(r43*c34-c1345)
								+ty1*(c34-c1345)
								+tz1*(c34-c1345))*(u[k][j][i][1]*u[k][j][i][1])
							+(tx1*(c34-c1345)
								+ty1*(r43*c34-c1345)
								+tz1*(c34-c1345))*(u[k][j][i][2]*u[k][j][i][2])
							+(tx1*(c34-c1345)
								+ty1*(c34-c1345)
								+tz1*(r43*c34-c1345))*(u[k][j][i][3]*u[k][j][i][3])
				  )*tmp3
						+(tx1+ty1+tz1)*c1345*tmp2*u[k][j][i][4]) )
			d[j][i][1][4] = ( dt*2.0
				*(tx1*(r43*c34-c1345)
						+ty1*(c34-c1345)
						+tz1*(c34-c1345))*tmp2*u[k][j][i][1] )
			d[j][i][2][4] = ( dt*2.0
				*(tx1*(c34-c1345)
						+ty1*(r43*c34-c1345)
						+tz1*(c34-c1345))*tmp2*u[k][j][i][2] )
			d[j][i][3][4] = ( dt*2.0
				*(tx1*(c34-c1345)
						+ty1*(c34-c1345)
						+tz1*(r43*c34-c1345))*tmp2*u[k][j][i][3] )
			d[j][i][4][4] = ( 1.0
				+dt*2.0*(tx1+ty1+tz1)*c1345*tmp1
				+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5) )
			# ---------------------------------------------------------------------
			# form the first block sub-diagonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k][j][i+1]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			a[j][i][0][0] = -dt*tx1*dx1
			a[j][i][1][0] = dt*tx2
			a[j][i][2][0] = 0.0
			a[j][i][3][0] = 0.0
			a[j][i][4][0] = 0.0
			a[j][i][0][1] = ( dt*tx2
				*(-(u[k][j][i+1][1]*tmp1)*(u[k][j][i+1][1]*tmp1)
						+C2*qs[k][j][i+1]*tmp1)
				-dt*tx1*(-r43*c34*tmp2*u[k][j][i+1][1]) )
			a[j][i][1][1] = ( dt*tx2
				*((2.0-C2)*(u[k][j][i+1][1]*tmp1))
				-dt*tx1*(r43*c34*tmp1)
				-dt*tx1*dx2 )
			a[j][i][2][1] = ( dt*tx2
				*(-C2*(u[k][j][i+1][2]*tmp1)) )
			a[j][i][3][1] = ( dt*tx2
				*(-C2*(u[k][j][i+1][3]*tmp1)) )
			a[j][i][4][1] = dt*tx2*C2
			a[j][i][0][2] = ( dt*tx2
				*(-(u[k][j][i+1][1]*u[k][j][i+1][2])*tmp2)
				-dt*tx1*(-c34*tmp2*u[k][j][i+1][2]) )
			a[j][i][1][2] = dt*tx2*(u[k][j][i+1][2]*tmp1)
			a[j][i][2][2] = ( dt*tx2*(u[k][j][i+1][1]*tmp1)
				-dt*tx1*(c34*tmp1)
				-dt*tx1*dx3 )
			a[j][i][3][2] = 0.0
			a[j][i][4][2] = 0.0
			a[j][i][0][3] = ( dt*tx2
				*(-(u[k][j][i+1][1]*u[k][j][i+1][3])*tmp2)
				-dt*tx1*(-c34*tmp2*u[k][j][i+1][3]) )
			a[j][i][1][3] = dt*tx2*(u[k][j][i+1][3]*tmp1)
			a[j][i][2][3] = 0.0
			a[j][i][3][3] = ( dt*tx2*(u[k][j][i+1][1]*tmp1)
				-dt*tx1*(c34*tmp1)
				-dt*tx1*dx4 )
			a[j][i][4][3] = 0.0
			a[j][i][0][4] = ( dt*tx2
				*((C2*2.0*qs[k][j][i+1]
							-C1*u[k][j][i+1][4])
						*(u[k][j][i+1][1]*tmp2))
				-dt*tx1
				*(-(r43*c34-c1345)*tmp3*(u[k][j][i+1][1]*u[k][j][i+1][1])
						-(c34-c1345)*tmp3*(u[k][j][i+1][2]*u[k][j][i+1][2])
						-(c34-c1345)*tmp3*( u[k][j][i+1][3]*u[k][j][i+1][3])
						-c1345*tmp2*u[k][j][i+1][4]) )
			a[j][i][1][4] = ( dt*tx2
				*(C1*(u[k][j][i+1][4]*tmp1)
						-C2
						*(u[k][j][i+1][1]*u[k][j][i+1][1]*tmp2
							+qs[k][j][i+1]*tmp1))
				-dt*tx1
				*(r43*c34-c1345)*tmp2*u[k][j][i+1][1] )
			a[j][i][2][4] = ( dt*tx2
				*(-C2*(u[k][j][i+1][2]*u[k][j][i+1][1])*tmp2)
				-dt*tx1
				*(c34-c1345)*tmp2*u[k][j][i+1][2] )
			a[j][i][3][4] = ( dt*tx2
				*(-C2*(u[k][j][i+1][3]*u[k][j][i+1][1])*tmp2)
				-dt*tx1
				*(c34-c1345)*tmp2*u[k][j][i+1][3] )
			a[j][i][4][4] = ( dt*tx2
				*(C1*(u[k][j][i+1][1]*tmp1))
				-dt*tx1*c1345*tmp1
				-dt*tx1*dx5 )
			# ---------------------------------------------------------------------
			# form the second block sub-diagonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k][j+1][i]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			b[j][i][0][0] = -dt*ty1*dy1
			b[j][i][1][0] = 0.0
			b[j][i][2][0] = dt*ty2
			b[j][i][3][0] = 0.0
			b[j][i][4][0] = 0.0
			b[j][i][0][1] = ( dt*ty2
				*(-(u[k][j+1][i][1]*u[k][j+1][i][2])*tmp2)
				-dt*ty1*(-c34*tmp2*u[k][j+1][i][1]) )
			b[j][i][1][1] = ( dt*ty2*(u[k][j+1][i][2]*tmp1)
				-dt*ty1*(c34*tmp1)
				-dt*ty1*dy2 )
			b[j][i][2][1] = dt*ty2*(u[k][j+1][i][1]*tmp1)
			b[j][i][3][1] = 0.0
			b[j][i][4][1] = 0.0
			b[j][i][0][2] = ( dt*ty2
				*(-(u[k][j+1][i][2]*tmp1)*(u[k][j+1][i][2]*tmp1)
						+C2*(qs[k][j+1][i]*tmp1))
				-dt*ty1*(-r43*c34*tmp2*u[k][j+1][i][2]) )
			b[j][i][1][2] = ( dt*ty2
				*(-C2*(u[k][j+1][i][1]*tmp1)) )
			b[j][i][2][2] = ( dt*ty2*((2.0-C2)
					*(u[k][j+1][i][2]*tmp1))
				-dt*ty1*(r43*c34*tmp1)
				-dt*ty1*dy3 )
			b[j][i][3][2] = ( dt*ty2
				*(-C2*(u[k][j+1][i][3]*tmp1)) )
			b[j][i][4][2] = dt*ty2*C2
			b[j][i][0][3] = ( dt*ty2
				*(-(u[k][j+1][i][2]*u[k][j+1][i][3])*tmp2)
				-dt*ty1*(-c34*tmp2*u[k][j+1][i][3]) )
			b[j][i][1][3] = 0.0
			b[j][i][2][3] = dt*ty2*(u[k][j+1][i][3]*tmp1)
			b[j][i][3][3] = ( dt*ty2*(u[k][j+1][i][2]*tmp1)
				-dt*ty1*(c34*tmp1)
				-dt*ty1*dy4 )
			b[j][i][4][3] = 0.0
			b[j][i][0][4] = ( dt*ty2
				*((C2*2.0*qs[k][j+1][i]
							-C1*u[k][j+1][i][4])
						*(u[k][j+1][i][2]*tmp2))
				-dt*ty1
				*(-(c34-c1345)*tmp3*(u[k][j+1][i][1]*u[k][j+1][i][1])
						-(r43*c34-c1345)*tmp3*(u[k][j+1][i][2]*u[k][j+1][i][2])
						-(c34-c1345)*tmp3*(u[k][j+1][i][3]*u[k][j+1][i][3])
						-c1345*tmp2*u[k][j+1][i][4]) )
			b[j][i][1][4] = ( dt*ty2
				*(-C2*(u[k][j+1][i][1]*u[k][j+1][i][2])*tmp2)
				-dt*ty1
				*(c34-c1345)*tmp2*u[k][j+1][i][1] )
			b[j][i][2][4] = ( dt*ty2
				*(C1*(u[k][j+1][i][4]*tmp1)
						-C2 
						*(qs[k][j+1][i]*tmp1
							+u[k][j+1][i][2]*u[k][j+1][i][2]*tmp2))
				-dt*ty1
				*(r43*c34-c1345)*tmp2*u[k][j+1][i][2] )
			b[j][i][3][4] = ( dt*ty2
				*(-C2*(u[k][j+1][i][2]*u[k][j+1][i][3])*tmp2)
				-dt*ty1*(c34-c1345)*tmp2*u[k][j+1][i][3] )
			b[j][i][4][4] = ( dt*ty2
				*(C1*(u[k][j+1][i][2]*tmp1))
				-dt*ty1*c1345*tmp1
				-dt*ty1*dy5 )
			# ---------------------------------------------------------------------
			# form the third block sub-diagonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k+1][j][i]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			c[j][i][0][0] = -dt*tz1*dz1
			c[j][i][1][0] = 0.0
			c[j][i][2][0] = 0.0
			c[j][i][3][0] = dt*tz2
			c[j][i][4][0] = 0.0
			c[j][i][0][1] = ( dt*tz2
				*(-(u[k+1][j][i][1]*u[k+1][j][i][3])*tmp2)
				-dt*tz1*(-c34*tmp2*u[k+1][j][i][1]) )
			c[j][i][1][1] = ( dt*tz2*(u[k+1][j][i][3]*tmp1)
				-dt*tz1*c34*tmp1
				-dt*tz1*dz2 )
			c[j][i][2][1] = 0.0
			c[j][i][3][1] = dt*tz2*(u[k+1][j][i][1]*tmp1)
			c[j][i][4][1] = 0.0
			c[j][i][0][2] = ( dt*tz2
				*(-(u[k+1][j][i][2]*u[k+1][j][i][3])*tmp2)
				-dt*tz1*(-c34*tmp2*u[k+1][j][i][2]) )
			c[j][i][1][2] = 0.0
			c[j][i][2][2] = ( dt*tz2*(u[k+1][j][i][3]*tmp1)
				-dt*tz1*(c34*tmp1)
				-dt*tz1*dz3 )
			c[j][i][3][2] = dt*tz2*(u[k+1][j][i][2]*tmp1)
			c[j][i][4][2] = 0.0
			c[j][i][0][3] = ( dt*tz2
				*(-(u[k+1][j][i][3]*tmp1)*(u[k+1][j][i][3]*tmp1)
						+C2*(qs[k+1][j][i]*tmp1))
				-dt*tz1*(-r43*c34*tmp2*u[k+1][j][i][3]) )
			c[j][i][1][3] = ( dt*tz2
				*(-C2*(u[k+1][j][i][1]*tmp1)) )
			c[j][i][2][3] = ( dt*tz2
				*(-C2*(u[k+1][j][i][2]*tmp1)) )
			c[j][i][3][3] = ( dt*tz2*(2.0-C2)
				*(u[k+1][j][i][3]*tmp1)
				-dt*tz1*(r43*c34*tmp1)
				-dt*tz1*dz4 )
			c[j][i][4][3] = dt*tz2*C2
			c[j][i][0][4] = ( dt*tz2
				*((C2*2.0*qs[k+1][j][i]
							-C1*u[k+1][j][i][4])
						*(u[k+1][j][i][3]*tmp2))
				-dt*tz1
				*(-(c34-c1345)*tmp3*(u[k+1][j][i][1]*u[k+1][j][i][1])
						-(c34-c1345)*tmp3*(u[k+1][j][i][2]*u[k+1][j][i][2])
						-(r43*c34-c1345)*tmp3*(u[k+1][j][i][3]*u[k+1][j][i][3])
						-c1345*tmp2*u[k+1][j][i][4]) )
			c[j][i][1][4] = ( dt*tz2
				*(-C2*(u[k+1][j][i][1]*u[k+1][j][i][3])*tmp2)
				-dt*tz1*(c34-c1345)*tmp2*u[k+1][j][i][1] )
			c[j][i][2][4] = ( dt*tz2
				*(-C2*(u[k+1][j][i][2]*u[k+1][j][i][3])*tmp2)
				-dt*tz1*(c34-c1345)*tmp2*u[k+1][j][i][2] )
			c[j][i][3][4] = ( dt*tz2
				*(C1*(u[k+1][j][i][4]*tmp1)
						-C2
						*(qs[k+1][j][i]*tmp1
							+u[k+1][j][i][3]*u[k+1][j][i][3]*tmp2))
				-dt*tz1*(r43*c34-c1345)*tmp2*u[k+1][j][i][3] )
			c[j][i][4][4] = ( dt*tz2
				*(C1*(u[k+1][j][i][3]*tmp1))
				-dt*tz1*c1345*tmp1
				-dt*tz1*dz5 )
		#END for i in range(ist, iend):
	#END for j in range(jst, jend):
#END jacu()


# ---------------------------------------------------------------------
# compute the regular-sparse, block lower triangular solution:
# v <-- ( L-inv ) * v
# ---------------------------------------------------------------------
# to improve cache performance, second two dimensions padded by 1 
# for even number sizes only. only needed in v.
# ---------------------------------------------------------------------
@njit
def blts(nx,
		ny,
		nz,
		k,
		omega,
		v, #double v[][ISIZ2/2*2+1][ISIZ1/2*2+1][5], 
		ldz, #double ldz[][ISIZ1/2*2+1][5][5],
		ldy, #double ldy[][ISIZ1/2*2+1][5][5],
		ldx, #double ldx[][ISIZ1/2*2+1][5][5],
		d, #double d[][ISIZ1/2*2+1][5][5],
		ist,
		iend,
		jst,
		jend,
		nx0,
		ny0):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	tmat = numpy.empty((5, 5), dtype=numpy.float64) 
	tv = numpy.empty(5, dtype=numpy.float64)
	for j in range(jst, jend):
		for i in range(ist, iend):
			for m in range(5):
				v[k][j][i][m] = ( v[k][j][i][m]
					-omega*(ldz[j][i][0][m]*v[k-1][j][i][0]
							+ldz[j][i][1][m]*v[k-1][j][i][1]
							+ldz[j][i][2][m]*v[k-1][j][i][2]
							+ldz[j][i][3][m]*v[k-1][j][i][3]
							+ldz[j][i][4][m]*v[k-1][j][i][4]) )

	for j in range(jst, jend):
		for i in range(ist, iend):
			for m in range(5):
				tv[m] = ( v[k][j][i][m]
					-omega*(ldy[j][i][0][m]*v[k][j-1][i][0]
							+ldx[j][i][0][m]*v[k][j][i-1][0]
							+ldy[j][i][1][m]*v[k][j-1][i][1]
							+ldx[j][i][1][m]*v[k][j][i-1][1]
							+ldy[j][i][2][m]*v[k][j-1][i][2]
							+ldx[j][i][2][m]*v[k][j][i-1][2]
							+ldy[j][i][3][m]*v[k][j-1][i][3]
							+ldx[j][i][3][m]*v[k][j][i-1][3]
							+ldy[j][i][4][m]*v[k][j-1][i][4]
							+ldx[j][i][4][m]*v[k][j][i-1][4]) )
			# ---------------------------------------------------------------------
			# diagonal block inversion
			# 
			# forward elimination
			# ---------------------------------------------------------------------
			for m in range(5):
				tmat[0][m] = d[j][i][0][m]
				tmat[1][m] = d[j][i][1][m]
				tmat[2][m] = d[j][i][2][m]
				tmat[3][m] = d[j][i][3][m]
				tmat[4][m] = d[j][i][4][m]
			#
			tmp1 = 1.0/tmat[0][0]
			tmp = tmp1*tmat[0][1]
			tmat[1][1] = tmat[1][1]-tmp*tmat[1][0]
			tmat[2][1] = tmat[2][1]-tmp*tmat[2][0]
			tmat[3][1] = tmat[3][1]-tmp*tmat[3][0]
			tmat[4][1] = tmat[4][1]-tmp*tmat[4][0]
			tv[1] = tv[1]-tv[0]*tmp
			#
			tmp = tmp1*tmat[0][2]
			tmat[1][2] = tmat[1][2]-tmp*tmat[1][0]
			tmat[2][2] = tmat[2][2]-tmp*tmat[2][0]
			tmat[3][2] = tmat[3][2]-tmp*tmat[3][0]
			tmat[4][2] = tmat[4][2]-tmp*tmat[4][0]
			tv[2] = tv[2]-tv[0]*tmp
			#
			tmp = tmp1*tmat[0][3]
			tmat[1][3] = tmat[1][3]-tmp*tmat[1][0]
			tmat[2][3] = tmat[2][3]-tmp*tmat[2][0]
			tmat[3][3] = tmat[3][3]-tmp*tmat[3][0]
			tmat[4][3] = tmat[4][3]-tmp*tmat[4][0]
			tv[3] = tv[3]-tv[0]*tmp
			#
			tmp = tmp1*tmat[0][4]
			tmat[1][4] = tmat[1][4]-tmp*tmat[1][0]
			tmat[2][4] = tmat[2][4]-tmp*tmat[2][0]
			tmat[3][4] = tmat[3][4]-tmp*tmat[3][0]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][0]
			tv[4] = tv[4]-tv[0]*tmp
			#
			tmp1 = 1.0/tmat[1][1]
			tmp = tmp1*tmat[1][2]
			tmat[2][2] = tmat[2][2]-tmp*tmat[2][1]
			tmat[3][2] = tmat[3][2]-tmp*tmat[3][1]
			tmat[4][2] = tmat[4][2]-tmp*tmat[4][1]
			tv[2] = tv[2]-tv[1]*tmp
			#
			tmp = tmp1*tmat[1][3]
			tmat[2][3] = tmat[2][3]-tmp*tmat[2][1]
			tmat[3][3] = tmat[3][3]-tmp*tmat[3][1]
			tmat[4][3] = tmat[4][3]-tmp*tmat[4][1]
			tv[3] = tv[3]-tv[1]*tmp
			#
			tmp = tmp1*tmat[1][4]
			tmat[2][4] = tmat[2][4]-tmp*tmat[2][1]
			tmat[3][4] = tmat[3][4]-tmp*tmat[3][1]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][1]
			tv[4] = tv[4]-tv[1]*tmp
			#
			tmp1 = 1.0/tmat[2][2]
			tmp = tmp1*tmat[2][3]
			tmat[3][3] = tmat[3][3]-tmp*tmat[3][2]
			tmat[4][3] = tmat[4][3]-tmp*tmat[4][2]
			tv[3] = tv[3]-tv[2]*tmp
			#
			tmp = tmp1*tmat[2][4]
			tmat[3][4] = tmat[3][4]-tmp*tmat[3][2]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][2]
			tv[4] = tv[4]-tv[2]*tmp
			#
			tmp1 = 1.0/tmat[3][3]
			tmp = tmp1*tmat[3][4]
			tmat[4][4] = tmat[4][4]-tmp*tmat[4][3]
			tv[4] = tv[4]-tv[3]*tmp
			# ---------------------------------------------------------------------
			# back substitution
			# ---------------------------------------------------------------------
			v[k][j][i][4] = tv[4]/tmat[4][4]
			tv[3] = ( tv[3] 
				-tmat[4][3]*v[k][j][i][4] )
			v[k][j][i][3] = tv[3]/tmat[3][3]
			tv[2] = ( tv[2]
				-tmat[3][2]*v[k][j][i][3]
				-tmat[4][2]*v[k][j][i][4] )
			v[k][j][i][2] = tv[2]/tmat[2][2]
			tv[1] = ( tv[1]
				-tmat[2][1]*v[k][j][i][2]
				-tmat[3][1]*v[k][j][i][3]
				-tmat[4][1]*v[k][j][i][4])
			v[k][j][i][1] = tv[1]/tmat[1][1]
			tv[0] = ( tv[0]
				-tmat[1][0]*v[k][j][i][1]
				-tmat[2][0]*v[k][j][i][2]
				-tmat[3][0]*v[k][j][i][3]
				-tmat[4][0]*v[k][j][i][4])
			v[k][j][i][0] = tv[0]/tmat[0][0]
		#for i in range(ist, iend):
	#for j in range(jst, jend):
#END blts()


# ---------------------------------------------------------------------
# compute the lower triangular part of the jacobian matrix
# ---------------------------------------------------------------------
@njit
def jacld(k,
		a, b, c, d,
		u, rho_i, qs):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	r43 = (4.0/3.0)
	c1345 = C1*C3*C4*C5
	c34 = C3*C4
	for j in range(jst, jend):
		for i in range(ist, iend): 
			# ---------------------------------------------------------------------
			# form the block daigonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k][j][i]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			d[j][i][0][0] = 1.0+dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1)
			d[j][i][1][0] = 0.0
			d[j][i][2][0] = 0.0
			d[j][i][3][0] = 0.0
			d[j][i][4][0] = 0.0
			d[j][i][0][1] = ( -dt*2.0
				*(tx1*r43+ty1+tz1)*c34*tmp2*u[k][j][i][1] )
			d[j][i][1][1] = ( 1.0
				+dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1)
				+dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2) )
			d[j][i][2][1] = 0.0
			d[j][i][3][1] = 0.0
			d[j][i][4][1] = 0.0
			d[j][i][0][2] = ( -dt*2.0 
				*(tx1+ty1*r43+tz1)*c34*tmp2*u[k][j][i][2] )
			d[j][i][1][2] = 0.0
			d[j][i][2][2] = ( 1.0
				+dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1)
				+dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3) )
			d[j][i][3][2] = 0.0
			d[j][i][4][2] = 0.0
			d[j][i][0][3] = ( -dt*2.0
				*(tx1+ty1+tz1*r43)*c34*tmp2*u[k][j][i][3] )
			d[j][i][1][3] = 0.0
			d[j][i][2][3] = 0.0
			d[j][i][3][3] = ( 1.0
				+dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43)
				+dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4) )
			d[j][i][4][3] = 0.0
			d[j][i][0][4] = ( -dt*2.0
				*(((tx1*(r43*c34-c1345)
								+ty1*(c34-c1345)
								+tz1*(c34-c1345))*(u[k][j][i][1]*u[k][j][i][1])
							+(tx1*(c34-c1345)
								+ty1*(r43*c34-c1345)
								+tz1*(c34-c1345))*(u[k][j][i][2]*u[k][j][i][2])
							+(tx1*(c34-c1345)
								+ty1*(c34-c1345)
								+tz1*(r43*c34-c1345))*(u[k][j][i][3]*u[k][j][i][3])
				  )*tmp3
						+(tx1+ty1+tz1)*c1345*tmp2*u[k][j][i][4]) )
			d[j][i][1][4] = (dt*2.0*tmp2*u[k][j][i][1]
				*(tx1*(r43*c34-c1345)
						+ty1*(c34-c1345)
						+tz1*(c34-c1345)) )
			d[j][i][2][4] = (dt*2.0*tmp2*u[k][j][i][2]
				*(tx1*(c34-c1345)
						+ty1*(r43*c34-c1345)
						+tz1*(c34-c1345)) )
			d[j][i][3][4] = (dt*2.0*tmp2*u[k][j][i][3]
				*(tx1*(c34-c1345)
						+ty1*(c34-c1345)
						+tz1*(r43*c34-c1345)) )
			d[j][i][4][4] = (1.0
				+dt*2.0*(tx1+ty1+tz1)*c1345*tmp1
				+dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5) )
			# ---------------------------------------------------------------------
			# form the first block sub-diagonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k-1][j][i]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			a[j][i][0][0] = -dt*tz1*dz1
			a[j][i][1][0] = 0.0
			a[j][i][2][0] = 0.0
			a[j][i][3][0] = -dt*tz2
			a[j][i][4][0] = 0.0
			a[j][i][0][1] = ( -dt*tz2
				*(-(u[k-1][j][i][1]*u[k-1][j][i][3])*tmp2)
				-dt*tz1*(-c34*tmp2*u[k-1][j][i][1]) )
			a[j][i][1][1] = ( -dt*tz2*(u[k-1][j][i][3]*tmp1)
				-dt*tz1*c34*tmp1
				-dt*tz1*dz2 )
			a[j][i][2][1] = 0.0
			a[j][i][3][1] = -dt*tz2*(u[k-1][j][i][1]*tmp1)
			a[j][i][4][1] = 0.0
			a[j][i][0][2] = ( -dt*tz2
				*(-(u[k-1][j][i][2]*u[k-1][j][i][3])*tmp2)
				-dt*tz1*(-c34*tmp2*u[k-1][j][i][2]) )
			a[j][i][1][2] = 0.0
			a[j][i][2][2] = ( -dt*tz2*(u[k-1][j][i][3]*tmp1)
				-dt*tz1*(c34*tmp1)
				-dt*tz1*dz3 )
			a[j][i][3][2] = -dt*tz2*(u[k-1][j][i][2]*tmp1)
			a[j][i][4][2] = 0.0
			a[j][i][0][3] = ( -dt*tz2
				*(-(u[k-1][j][i][3]*tmp1)*(u[k-1][j][i][3]*tmp1)
						+C2*qs[k-1][j][i]*tmp1)
				-dt*tz1*(-r43*c34*tmp2*u[k-1][j][i][3]) )
			a[j][i][1][3] = ( -dt*tz2
				*(-C2*(u[k-1][j][i][1]*tmp1)) )
			a[j][i][2][3] = ( -dt*tz2
				*(-C2*(u[k-1][j][i][2]*tmp1)) )
			a[j][i][3][3] = ( -dt*tz2*(2.0-C2)
				*(u[k-1][j][i][3]*tmp1)
				-dt*tz1*(r43*c34*tmp1)
				-dt*tz1*dz4 )
			a[j][i][4][3] = -dt*tz2*C2
			a[j][i][0][4] = ( -dt*tz2
				*((C2*2.0*qs[k-1][j][i]-C1*u[k-1][j][i][4])
						*u[k-1][j][i][3]*tmp2)
				-dt*tz1
				*(-(c34-c1345)*tmp3*(u[k-1][j][i][1]*u[k-1][j][i][1])
						-(c34-c1345)*tmp3*(u[k-1][j][i][2]*u[k-1][j][i][2])
						-(r43*c34-c1345)*tmp3*(u[k-1][j][i][3]*u[k-1][j][i][3])
						-c1345*tmp2*u[k-1][j][i][4]) )
			a[j][i][1][4] = ( -dt*tz2
				*(-C2*(u[k-1][j][i][1]*u[k-1][j][i][3])*tmp2)
				-dt*tz1*(c34-c1345)*tmp2*u[k-1][j][i][1] )
			a[j][i][2][4] = ( -dt*tz2
				*(-C2*(u[k-1][j][i][2]*u[k-1][j][i][3])*tmp2)
				-dt*tz1*(c34-c1345)*tmp2*u[k-1][j][i][2] )
			a[j][i][3][4] = ( -dt*tz2
				*(C1*(u[k-1][j][i][4]*tmp1)
						-C2*(qs[k-1][j][i]*tmp1
							+u[k-1][j][i][3]*u[k-1][j][i][3]*tmp2))
				-dt*tz1*(r43*c34-c1345)*tmp2*u[k-1][j][i][3] )
			a[j][i][4][4] = ( -dt*tz2
				*(C1*(u[k-1][j][i][3]*tmp1))
				-dt*tz1*c1345*tmp1
				-dt*tz1*dz5 )
			# ---------------------------------------------------------------------
			# form the second block sub-diagonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k][j-1][i]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			b[j][i][0][0] = -dt*ty1*dy1
			b[j][i][1][0] = 0.0
			b[j][i][2][0] = -dt*ty2
			b[j][i][3][0] = 0.0
			b[j][i][4][0] = 0.0
			b[j][i][0][1] = ( -dt*ty2
				*(-(u[k][j-1][i][1]*u[k][j-1][i][2])*tmp2)
				-dt*ty1*(-c34*tmp2*u[k][j-1][i][1]) )
			b[j][i][1][1] = ( -dt*ty2*(u[k][j-1][i][2]*tmp1)
				-dt*ty1*(c34*tmp1)
				-dt*ty1*dy2 )
			b[j][i][2][1] = -dt*ty2*(u[k][j-1][i][1]*tmp1)
			b[j][i][3][1] = 0.0
			b[j][i][4][1] = 0.0
			b[j][i][0][2] = ( -dt*ty2
				*(-(u[k][j-1][i][2]*tmp1)*(u[k][j-1][i][2]*tmp1)
						+C2*(qs[k][j-1][i]*tmp1))
				-dt*ty1*(-r43*c34*tmp2*u[k][j-1][i][2]) )
			b[j][i][1][2] = ( -dt*ty2
				*(-C2*(u[k][j-1][i][1]*tmp1)) )
			b[j][i][2][2] = ( -dt*ty2*((2.0-C2)*(u[k][j-1][i][2]*tmp1))
				-dt*ty1*(r43*c34*tmp1)
				-dt*ty1*dy3 )
			b[j][i][3][2] = -dt*ty2*(-C2*(u[k][j-1][i][3]*tmp1))
			b[j][i][4][2] = -dt*ty2*C2
			b[j][i][0][3] = ( -dt*ty2
				*(-(u[k][j-1][i][2]*u[k][j-1][i][3])*tmp2)
				-dt*ty1*(-c34*tmp2*u[k][j-1][i][3]) )
			b[j][i][1][3] = 0.0
			b[j][i][2][3] = -dt*ty2*(u[k][j-1][i][3]*tmp1)
			b[j][i][3][3] = ( -dt*ty2*(u[k][j-1][i][2]*tmp1)
				-dt*ty1*(c34*tmp1)
				-dt*ty1*dy4 )
			b[j][i][4][3] = 0.0
			b[j][i][0][4] = ( -dt*ty2
				*((C2*2.0*qs[k][j-1][i]-C1*u[k][j-1][i][4])
						*(u[k][j-1][i][2]*tmp2))
				-dt*ty1
				*(-(c34-c1345)*tmp3*(u[k][j-1][i][1]*u[k][j-1][i][1])
						-(r43*c34-c1345)*tmp3*(u[k][j-1][i][2]*u[k][j-1][i][2])
						-(c34-c1345)*tmp3*(u[k][j-1][i][3]*u[k][j-1][i][3])
						-c1345*tmp2*u[k][j-1][i][4]) )
			b[j][i][1][4] = ( -dt*ty2
				*(-C2*(u[k][j-1][i][1]*u[k][j-1][i][2])*tmp2)
				-dt*ty1*(c34-c1345)*tmp2*u[k][j-1][i][1] )
			b[j][i][2][4] = ( -dt*ty2
				*(C1*(u[k][j-1][i][4]*tmp1)
						-C2*(qs[k][j-1][i]*tmp1
							+u[k][j-1][i][2]*u[k][j-1][i][2]*tmp2))
				-dt*ty1*(r43*c34-c1345)*tmp2*u[k][j-1][i][2] )
			b[j][i][3][4] = ( -dt*ty2
				*(-C2*(u[k][j-1][i][2]*u[k][j-1][i][3])*tmp2)
				-dt*ty1*(c34-c1345)*tmp2*u[k][j-1][i][3] )
			b[j][i][4][4] = ( -dt*ty2
				*(C1*(u[k][j-1][i][2]*tmp1))
				-dt*ty1*c1345*tmp1
				-dt*ty1*dy5 )
			# ---------------------------------------------------------------------
			# form the third block sub-diagonal
			# ---------------------------------------------------------------------
			tmp1 = rho_i[k][j][i-1]
			tmp2 = tmp1*tmp1
			tmp3 = tmp1*tmp2
			c[j][i][0][0] = -dt*tx1*dx1
			c[j][i][1][0] = -dt*tx2
			c[j][i][2][0] = 0.0
			c[j][i][3][0] = 0.0
			c[j][i][4][0] = 0.0
			c[j][i][0][1] = ( -dt*tx2
				*(-(u[k][j][i-1][1]*tmp1)*(u[k][j][i-1][1]*tmp1)
						+C2*qs[k][j][i-1]*tmp1)
				-dt*tx1*(-r43*c34*tmp2*u[k][j][i-1][1]) )
			c[j][i][1][1] = ( -dt*tx2
				*((2.0-C2)*(u[k][j][i-1][1]*tmp1))
				-dt*tx1*(r43*c34*tmp1)
				-dt*tx1*dx2 )
			c[j][i][2][1] = ( -dt*tx2
				*(-C2*(u[k][j][i-1][2]*tmp1)) )
			c[j][i][3][1] = ( -dt*tx2
				*(-C2*(u[k][j][i-1][3]*tmp1)) )
			c[j][i][4][1] = -dt*tx2*C2
			c[j][i][0][2] = ( -dt*tx2
				*(-(u[k][j][i-1][1]*u[k][j][i-1][2])*tmp2)
				-dt*tx1*(-c34*tmp2*u[k][j][i-1][2]) )
			c[j][i][1][2] = -dt*tx2*(u[k][j][i-1][2]*tmp1)
			c[j][i][2][2] = ( -dt*tx2*(u[k][j][i-1][1]*tmp1)
				-dt*tx1*(c34*tmp1)
				-dt*tx1*dx3 )
			c[j][i][3][2] = 0.0
			c[j][i][4][2] = 0.0
			c[j][i][0][3] = ( -dt*tx2
				*(-(u[k][j][i-1][1]*u[k][j][i-1][3])*tmp2)
				-dt*tx1*(-c34*tmp2*u[k][j][i-1][3]) )
			c[j][i][1][3] = -dt*tx2*(u[k][j][i-1][3]*tmp1)
			c[j][i][2][3] = 0.0
			c[j][i][3][3] = ( -dt*tx2*(u[k][j][i-1][1]*tmp1)
				-dt*tx1*(c34*tmp1)-dt*tx1*dx4 )
			c[j][i][4][3] = 0.0
			c[j][i][0][4] = ( -dt*tx2
				*((C2*2.0*qs[k][j][i-1]-C1*u[k][j][i-1][4])
						*u[k][j][i-1][1]*tmp2)
				-dt*tx1
				*(-(r43*c34-c1345)*tmp3*(u[k][j][i-1][1]*u[k][j][i-1][1])
						-(c34-c1345)*tmp3*(u[k][j][i-1][2]*u[k][j][i-1][2])
						-(c34-c1345)*tmp3*(u[k][j][i-1][3]*u[k][j][i-1][3])
						-c1345*tmp2*u[k][j][i-1][4]) )
			c[j][i][1][4] = ( -dt*tx2
				*(C1*(u[k][j][i-1][4]*tmp1)
						-C2*(u[k][j][i-1][1]*u[k][j][i-1][1]*tmp2
							+qs[k][j][i-1]*tmp1))
				-dt*tx1*(r43*c34-c1345)*tmp2*u[k][j][i-1][1] )
			c[j][i][2][4] = ( -dt*tx2
				*(-C2*(u[k][j][i-1][2]*u[k][j][i-1][1])*tmp2)
				-dt*tx1*(c34-c1345)*tmp2*u[k][j][i-1][2] )
			c[j][i][3][4] = ( -dt*tx2
				*(-C2*(u[k][j][i-1][3]*u[k][j][i-1][1])*tmp2)
				-dt*tx1*(c34-c1345)*tmp2*u[k][j][i-1][3] )
			c[j][i][4][4] = ( -dt*tx2
				*(C1*(u[k][j][i-1][1]*tmp1))
				-dt*tx1*c1345*tmp1
				-dt*tx1*dx5 )
		#END for i in range(ist, iend): 
	#END for j in range(jst, jend):
#END jacld()


# ---------------------------------------------------------------------
# to compute the l2-norm of vector v.
# ---------------------------------------------------------------------
# to improve cache performance, second two dimensions padded by 1 
# for even number sizes only.  Only needed in v.
# ---------------------------------------------------------------------
@njit
def l2norm(nx0,
		ny0,
		nz0,
		ist,
		iend,
		jst,
		jend,
		v, #double v[][ISIZ2/2*2+1][ISIZ1/2*2+1][5],
		sum_v): #double sum[]){
	for m in range(5): 
		sum_v[m] = 0.0

	for k in range(1, nz0-1):
		for j in range(jst, jend):
			for i in range(ist, iend):
				for m in range(5): 
					sum_v[m] = sum_v[m] + v[k][j][i][m] * v[k][j][i][m]
	
	m_sqrt = math.sqrt
	for m in range(5): 
		sum_v[m] = m_sqrt(sum_v[m]/((nx0-2)*(ny0-2)*(nz0-2)))
#END l2norm()


# ---------------------------------------------------------------------
# compute the right hand sides
# ---------------------------------------------------------------------
@njit
def rhs(rsd, frct, rho_i, qs, u, flux):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	utmp = numpy.empty((npbparams.ISIZ3, 6), dtype=numpy.float64) 
	rtmp = numpy.empty((npbparams.ISIZ3, 5), dtype=numpy.float64)
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHS)
	for k in range(nz):
		for j in range(ny):
			for i in range(nx):
				for m in range(5): 
					rsd[k][j][i][m] = -frct[k][j][i][m]

				tmp=1.0/u[k][j][i][0]
				rho_i[k][j][i] = tmp
				qs[k][j][i] = ( 0.50*(u[k][j][i][1]*u[k][j][i][1]
						+u[k][j][i][2]*u[k][j][i][2]
						+u[k][j][i][3]*u[k][j][i][3])
					*tmp )

	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHSX)
	
	# ---------------------------------------------------------------------
	# xi-direction flux differences
	# ---------------------------------------------------------------------
	for k in range(1, nz-1): 
		for j in range(jst, jend):
			for i in range(nx):
				flux[i][0] = u[k][j][i][1]
				u21 = u[k][j][i][1]*rho_i[k][j][i]
				q = qs[k][j][i]
				flux[i][1] = u[k][j][i][1]*u21+C2*(u[k][j][i][4]-q)
				flux[i][2] = u[k][j][i][2]*u21
				flux[i][3] = u[k][j][i][3]*u21
				flux[i][4] = (C1*u[k][j][i][4]-C2*q)*u21
    
			for i in range(ist, iend): 
				for m in range(5):
					rsd[k][j][i][m] = ( rsd[k][j][i][m]
						-tx2*(flux[i+1][m]-flux[i-1][m]) )
    
			for i in range(ist, nx):
				tmp = rho_i[k][j][i]
				u21i = tmp*u[k][j][i][1]
				u31i = tmp*u[k][j][i][2]
				u41i = tmp*u[k][j][i][3]
				u51i = tmp*u[k][j][i][4]
				tmp = rho_i[k][j][i-1]
				u21im1 = tmp*u[k][j][i-1][1]
				u31im1 = tmp*u[k][j][i-1][2]
				u41im1 = tmp*u[k][j][i-1][3]
				u51im1 = tmp*u[k][j][i-1][4]
				flux[i][1] = (4.0/3.0)*tx3*(u21i-u21im1)
				flux[i][2] = tx3*(u31i-u31im1)
				flux[i][3] = tx3*(u41i-u41im1)
				flux[i][4] = ( 0.50*(1.0-C1*C5)
					*tx3*((u21i*u21i+u31i*u31i+u41i*u41i)
							-(u21im1*u21im1+u31im1*u31im1+u41im1*u41im1))
					+(1.0/6.0)
					*tx3*(u21i*u21i-u21im1*u21im1)
					+C1*C5*tx3*(u51i-u51im1) )
    
			for i in range(ist, iend):
				rsd[k][j][i][0] = (rsd[k][j][i][0]
					+dx1*tx1*(u[k][j][i-1][0]
							-2.0* u[k][j][i][0]
							+u[k][j][i+1][0]) )
				rsd[k][j][i][1] = (rsd[k][j][i][1]
					+tx3*C3*C4*(flux[i+1][1]-flux[i][1])
					+dx2*tx1*(u[k][j][i-1][1]
							-2.0*u[k][j][i][1]
							+u[k][j][i+1][1]) )
				rsd[k][j][i][2] = (rsd[k][j][i][2]
					+tx3*C3*C4*(flux[i+1][2]-flux[i][2])
					+dx3*tx1*(u[k][j][i-1][2]
							-2.0*u[k][j][i][2]
							+u[k][j][i+1][2]) )
				rsd[k][j][i][3] = (rsd[k][j][i][3]
					+tx3*C3*C4*(flux[i+1][3]-flux[i][3])
					+dx4*tx1*(u[k][j][i-1][3]
							-2.0*u[k][j][i][3]
							+u[k][j][i+1][3]) )
				rsd[k][j][i][4] = (rsd[k][j][i][4]
					+tx3*C3*C4*(flux[i+1][4]-flux[i][4])
					+dx5*tx1*(u[k][j][i-1][4]
							-2.0*u[k][j][i][4]
							+u[k][j][i+1][4]) )
    
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			for m in range(5):
				rsd[k][j][1][m] = (rsd[k][j][1][m]
					-dssp*(+5.0*u[k][j][1][m]
							-4.0*u[k][j][2][m]
							+u[k][j][3][m]) )
				rsd[k][j][2][m] = (rsd[k][j][2][m]
					-dssp*(-4.0*u[k][j][1][m]
							+6.0*u[k][j][2][m]
							-4.0*u[k][j][3][m]
							+u[k][j][4][m]) )
    
			for i in range(3, nx-3):
				for m in range(5):
					rsd[k][j][i][m] = (rsd[k][j][i][m]
						-dssp*(u[k][j][i-2][m]
								-4.0*u[k][j][i-1][m]
								+6.0*u[k][j][i][m]
								-4.0*u[k][j][i+1][m]
								+u[k][j][i+2][m]) )
    
			for m in range(5):
				rsd[k][j][nx-3][m] = (rsd[k][j][nx-3][m]
					-dssp*(u[k][j][nx-5][m]
							-4.0*u[k][j][nx-4][m]
							+6.0*u[k][j][nx-3][m]
							-4.0*u[k][j][nx-2][m]) )
				rsd[k][j][nx-2][m] = (rsd[k][j][nx-2][m]
					-dssp*(u[k][j][nx-4][m]
							-4.0*u[k][j][nx-3][m]
							+5.0*u[k][j][nx-2][m]) )
		#END for j in range(jst, jend):
	#END for k in range(1, nz-1): 
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHSX)
	#if timeron: Not supported by @njit
	#	c_timers.imer_start(T_RHSY)
	
	# ---------------------------------------------------------------------
	# eta-direction flux differences
	# ---------------------------------------------------------------------
	for k in range(1, nz-1):
		for i in range(ist, iend):
			for j in range(ny):
				flux[j][0] = u[k][j][i][2]
				u31 = u[k][j][i][2]*rho_i[k][j][i]
				q = qs[k][j][i]
				flux[j][1] = u[k][j][i][1]*u31
				flux[j][2] = u[k][j][i][2]*u31+C2*(u[k][j][i][4]-q)
				flux[j][3] = u[k][j][i][3]*u31
				flux[j][4] = (C1*u[k][j][i][4]-C2*q)*u31
    
			for j in range(jst, jend):
				for m in range(5):
					rsd[k][j][i][m] = ( rsd[k][j][i][m]
						-ty2*(flux[j+1][m]-flux[j-1][m]) )
    
			for j in range(jst, ny):
				tmp = rho_i[k][j][i]
				u21j = tmp*u[k][j][i][1]
				u31j = tmp*u[k][j][i][2]
				u41j = tmp*u[k][j][i][3]
				u51j = tmp*u[k][j][i][4]
				tmp = rho_i[k][j-1][i]
				u21jm1 = tmp*u[k][j-1][i][1]
				u31jm1 = tmp*u[k][j-1][i][2]
				u41jm1 = tmp*u[k][j-1][i][3]
				u51jm1 = tmp*u[k][j-1][i][4]
				flux[j][1] = ty3*(u21j-u21jm1)
				flux[j][2] = (4.0/3.0)*ty3*(u31j-u31jm1)
				flux[j][3] = ty3*(u41j-u41jm1)
				flux[j][4] = ( 0.50*(1.0-C1*C5)
					*ty3*((u21j*u21j+u31j*u31j+u41j*u41j)
							-(u21jm1*u21jm1+u31jm1*u31jm1+u41jm1*u41jm1))
					+(1.0/6.0)
					*ty3*(u31j*u31j-u31jm1*u31jm1)
					+C1*C5*ty3*(u51j-u51jm1) )
    
			for j in range(jst, jend):
				rsd[k][j][i][0] = ( rsd[k][j][i][0]
					+dy1*ty1*(u[k][j-1][i][0]
							-2.0*u[k][j][i][0]
							+u[k][j+1][i][0]) )
				rsd[k][j][i][1] = ( rsd[k][j][i][1]
					+ty3*C3*C4*(flux[j+1][1]-flux[j][1])
					+dy2*ty1*(u[k][j-1][i][1]
							-2.0*u[k][j][i][1]
							+u[k][j+1][i][1]) )
				rsd[k][j][i][2] = ( rsd[k][j][i][2]
					+ty3*C3*C4*(flux[j+1][2]-flux[j][2])
					+dy3*ty1*(u[k][j-1][i][2]
							-2.0*u[k][j][i][2]
							+u[k][j+1][i][2]) )
				rsd[k][j][i][3] = ( rsd[k][j][i][3]
					+ty3*C3*C4*(flux[j+1][3]-flux[j][3])
					+dy4*ty1*(u[k][j-1][i][3]
							-2.0*u[k][j][i][3]
							+u[k][j+1][i][3]) )
				rsd[k][j][i][4] = ( rsd[k][j][i][4]
					+ty3*C3*C4*(flux[j+1][4]-flux[j][4])
					+dy5*ty1*(u[k][j-1][i][4]
							-2.0*u[k][j][i][4]
							+u[k][j+1][i][4]) )
		#END for i in range(ist, iend):
		
		# ---------------------------------------------------------------------
		# fourth-order dissipation
		# ---------------------------------------------------------------------
		for i in range(ist, iend):
			for m in range(5):
				rsd[k][1][i][m] = ( rsd[k][1][i][m]
					-dssp*(+5.0*u[k][1][i][m]
							-4.0*u[k][2][i][m]
							+u[k][3][i][m]) )
				rsd[k][2][i][m] = ( rsd[k][2][i][m]
					-dssp*(-4.0*u[k][1][i][m]
							+6.0*u[k][2][i][m]
							-4.0*u[k][3][i][m]
							+u[k][4][i][m]) )
    
		for j in range(3, ny-3):
			for i in range(ist, iend):
				for m in range(5):
					rsd[k][j][i][m] = ( rsd[k][j][i][m]
						-dssp*(u[k][j-2][i][m]
								-4.0*u[k][j-1][i][m]
								+6.0*u[k][j][i][m]
								-4.0*u[k][j+1][i][m]
								+u[k][j+2][i][m]) )
    
		for i in range(ist, iend):
			for m in range(5):
				rsd[k][ny-3][i][m] = ( rsd[k][ny-3][i][m]
					-dssp*(u[k][ny-5][i][m]
							-4.0*u[k][ny-4][i][m]
							+6.0*u[k][ny-3][i][m]
							-4.0*u[k][ny-2][i][m]) )
				rsd[k][ny-2][i][m] = ( rsd[k][ny-2][i][m]
					-dssp*(u[k][ny-4][i][m]
							-4.0*u[k][ny-3][i][m]
							+5.0*u[k][ny-2][i][m]) )
	#END for k in range(1, nz-1):
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHSY)
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHSZ)
	
	# ---------------------------------------------------------------------
	# zeta-direction flux differences
	# ---------------------------------------------------------------------
	for j in range(jst, jend):
		for i in range(ist, iend):
			for k in range(nz):
				utmp[k][0] = u[k][j][i][0]
				utmp[k][1] = u[k][j][i][1]
				utmp[k][2] = u[k][j][i][2]
				utmp[k][3] = u[k][j][i][3]
				utmp[k][4] = u[k][j][i][4]
				utmp[k][5] = rho_i[k][j][i]
    
			for k in range(nz):
				flux[k][0] = utmp[k][3]
				u41 = utmp[k][3]*utmp[k][5]
				q = qs[k][j][i]
				flux[k][1] = utmp[k][1]*u41
				flux[k][2] = utmp[k][2]*u41
				flux[k][3] = utmp[k][3]*u41+C2*(utmp[k][4]-q)
				flux[k][4] = (C1*utmp[k][4]-C2*q)*u41
    
			for k in range(1, nz-1):
				for m in range(5):
					rtmp[k][m] = ( rsd[k][j][i][m]
						-tz2*(flux[k+1][m]-flux[k-1][m]) )
    
			for k in range(1, nz):
				tmp = utmp[k][5]
				u21k = tmp*utmp[k][1]
				u31k = tmp*utmp[k][2]
				u41k = tmp*utmp[k][3]
				u51k = tmp*utmp[k][4]
				tmp = utmp[k-1][5]
				u21km1 = tmp*utmp[k-1][1]
				u31km1 = tmp*utmp[k-1][2]
				u41km1 = tmp*utmp[k-1][3]
				u51km1 = tmp*utmp[k-1][4]
				flux[k][1] = tz3*(u21k-u21km1)
				flux[k][2] = tz3*(u31k-u31km1)
				flux[k][3] = (4.0/3.0)*tz3*(u41k-u41km1)
				flux[k][4] = ( 0.50*(1.0-C1*C5)
					*tz3*((u21k*u21k+u31k*u31k+u41k*u41k)
							-(u21km1*u21km1+u31km1*u31km1+u41km1*u41km1))
					+(1.0/6.0)
					*tz3*(u41k*u41k-u41km1*u41km1)
					+C1*C5*tz3*(u51k-u51km1) )
    
			for k in range(1, nz-1):
				rtmp[k][0] = ( rtmp[k][0]
					+dz1*tz1*(utmp[k-1][0]
							-2.0*utmp[k][0]
							+utmp[k+1][0]) )
				rtmp[k][1] = ( rtmp[k][1]
					+tz3*C3*C4*(flux[k+1][1]-flux[k][1])
					+dz2*tz1*(utmp[k-1][1]
							-2.0*utmp[k][1]
							+utmp[k+1][1]) )
				rtmp[k][2] = ( rtmp[k][2]
					+tz3*C3*C4*(flux[k+1][2]-flux[k][2])
					+dz3*tz1*(utmp[k-1][2]
							-2.0*utmp[k][2]
							+utmp[k+1][2]) )
				rtmp[k][3] = ( rtmp[k][3]
					+tz3*C3*C4*(flux[k+1][3]-flux[k][3])
					+dz4*tz1*(utmp[k-1][3]
							-2.0*utmp[k][3]
							+utmp[k+1][3]) )
				rtmp[k][4] = ( rtmp[k][4]
					+tz3*C3*C4*(flux[k+1][4]-flux[k][4])
					+dz5*tz1*(utmp[k-1][4]
							-2.0*utmp[k][4]
							+utmp[k+1][4]) )
    
			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			for m in range(5):
				rsd[1][j][i][m] = ( rtmp[1][m]
					-dssp*(+5.0*utmp[1][m]
							-4.0*utmp[2][m]
							+utmp[3][m]) )
				rsd[2][j][i][m] = ( rtmp[2][m]
					-dssp*(-4.0*utmp[1][m]
							+6.0*utmp[2][m]
							-4.0*utmp[3][m]
							+utmp[4][m] ) )
    
			for k in range(3, nz-3):
				for m in range(5):
					rsd[k][j][i][m] = ( rtmp[k][m]
						-dssp*(utmp[k-2][m]
								-4.0*utmp[k-1][m]
								+6.0*utmp[k][m]
								-4.0*utmp[k+1][m]
								+utmp[k+2][m]) )
    
			for m in range(5):
				rsd[nz-3][j][i][m] = ( rtmp[nz-3][m]
					-dssp*(utmp[nz-5][m]
							-4.0*utmp[nz-4][m]
							+6.0*utmp[nz-3][m]
							-4.0*utmp[nz-2][m]) )
				rsd[nz-2][j][i][m] = ( rtmp[nz-2][m]
					-dssp*(utmp[nz-4][m]
							-4.0*utmp[nz-3][m]
							+5.0*utmp[nz-2][m]) )
		#END for i in range(ist, iend): 
	#END for j in range(jst, jend):
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHSZ)
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHS)
#END rhs()


@njit
def ssor_exec(istep, rsd, rsdnm, tolrsd,
			a, b, c, d, u,
			frct, rho_i, qs, flux,
			tv, delunm, tmp):
	# ---------------------------------------------------------------------
	# perform SSOR iteration
	# ---------------------------------------------------------------------
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHS)
	for k in range(1, nz-1):
		for j in range(jst, jend):
			for i in range(ist, iend):
				for m in range(5):
					rsd[k][j][i][m] = dt*rsd[k][j][i][m]
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHS)
	for k in range(1, nz-1):
		# ---------------------------------------------------------------------
		# form the lower triangular part of the jacobian matrix
		# ---------------------------------------------------------------------
		#if timeron: Not supported by @njit
		#	c_timers.timer_start(T_JACLD)
		jacld(k, 
			a, b, c, d,
			u, rho_i, qs)
		#if timeron: Not supported by @njit
		#	c_timers.timer_stop(T_JACLD)
		# ---------------------------------------------------------------------
		# perform the lower triangular solution
		# ---------------------------------------------------------------------
		#if timeron: Not supported by @njit
		#	c_timers.timer_start(T_BLTS)
		blts(nx,
			ny,
			nz,
			k,
			omega,
			rsd,
			a,
			b,
			c,
			d,
			ist,
			iend,
			jst,
			jend,
			nx0,
			ny0)
		#if timeron: Not supported by @njit
		#	c_timers.timer_stop(T_BLTS)
	#END for k in range(1, nz-1):
	
	for k in range(nz-2, 0, -1):
		# ---------------------------------------------------------------------
		# form the strictly upper triangular part of the jacobian matrix
		# ---------------------------------------------------------------------
		#if timeron: Not supported by @njit
		#	c_timers.timer_start(T_JACU)
		jacu(k,
			a, b, c, d,
			u, rho_i, qs)
		#if timeron: Not supported by @njit
		#	c_timers.timer_stop(T_JACU)
		# ---------------------------------------------------------------------
		# perform the upper triangular solution
		# ---------------------------------------------------------------------
		#if timeron: Not supported by @njit
		#	c_timers.timer_start(T_BUTS)
		buts(nx,
			ny,
			nz,
			k,
			omega,
			rsd,
			tv,
			d,
			a,
			b,
			c,
			ist,
			iend,
			jst,
			jend,
			nx0,
			ny0)
		#if timeron: Not supported by @njit
		#	c_timers.timer_stop(T_BUTS)
	#END for k in range(nz-2, 0, -1)
	#
	# ---------------------------------------------------------------------
	# update the variables
	# ---------------------------------------------------------------------
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_ADD)
	for k in range(1, nz-1):
		for j in range(jst, jend):
			for i in range(ist, iend):
				for m in range(5):
					u[k][j][i][m] = u[k][j][i][m]+tmp*rsd[k][j][i][m]
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_ADD)
	# ---------------------------------------------------------------------
	# compute the max-norms of newton iteration corrections
	# ---------------------------------------------------------------------
	if (istep%inorm)==0:
		#if timeron: Not supported by @njit
		#	c_timers.timer_start(T_L2NORM)
		l2norm(nx0,
			ny0,
			nz0,
			ist,
			iend,
			jst,
			jend,
			rsd,
			delunm)
		#if timeron: Not supported by @njit
		#	c_timers.timer_stop(T_L2NORM)
    
	# ---------------------------------------------------------------------
	# compute the steady-state residuals
	# ---------------------------------------------------------------------
	rhs(rsd, frct, rho_i, qs, u, flux)
	# ---------------------------------------------------------------------
	# compute the max-norms of newton iteration residuals
	# ---------------------------------------------------------------------
	if (istep%inorm)==0 or istep==itmax:
		#if timeron: Not supported by @njit
		#	c_timers.timer_start(T_L2NORM)
		l2norm(nx0,
			ny0,
			nz0,
			ist,
			iend,
			jst,
			jend,
			rsd,
			rsdnm)
		#if timeron: Not supported by @njit
		#	c_timers.timer_stop(T_L2NORM)
    
	# ---------------------------------------------------------------------
	# check the newton-iteration residuals against the tolerance levels
	# ---------------------------------------------------------------------
	if ( (rsdnm[0]<tolrsd[0]) and
		(rsdnm[1]<tolrsd[1]) and
		(rsdnm[2]<tolrsd[2]) and
		(rsdnm[3]<tolrsd[3]) and
		(rsdnm[4]<tolrsd[4]) ):
		print(" \n convergence was achieved after", istep, " pseudo-time steps")
		return True
	
	return False
#END ssor_exec()


# ---------------------------------------------------------------------
# to perform pseudo-time stepping SSOR iterations
# for five nonlinear pde's.
# ---------------------------------------------------------------------
def ssor(niter, u, rsd, frct, flux):
	global maxtime
	global a, b, c, d
	global rho_i, qs
	global rsdnm, tolrsd
	
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	tv = numpy.empty((npbparams.ISIZ2, int(npbparams.ISIZ1/2*2+1), 5), dtype=numpy.float64)
	delunm = numpy.empty(5, dtype=numpy.float64)
	# ---------------------------------------------------------------------
	# begin pseudo-time stepping iterations
	# ---------------------------------------------------------------------
	tmp = 1.0/(omega*(2.0-omega))
	# ---------------------------------------------------------------------
	# initialize a,b,c,d to zero (guarantees that page tables have been
	# formed, if applicable on given architecture, before timestepping).
	# ---------------------------------------------------------------------
	for j in range(npbparams.ISIZ2):
		for i in range(npbparams.ISIZ1):
			for n in range(5):
				for m in range(5):
					a[j][i][n][m] = 0.0
					b[j][i][n][m] = 0.0
					c[j][i][n][m] = 0.0
					d[j][i][n][m] = 0.0

	for i in range(1, T_LAST+1):
		 c_timers.timer_clear(i)
	# ---------------------------------------------------------------------
	# compute the steady-state residuals
	# ---------------------------------------------------------------------
	rhs(rsd, frct, rho_i, qs, u, flux)
	# ---------------------------------------------------------------------
	# compute the L2 norms of newton iteration residuals
	# ---------------------------------------------------------------------
	l2norm(nx0,
		ny0,
		nz0,
		ist,
		iend,
		jst,
		jend,
		rsd,
		rsdnm)
	for i in range(1, T_LAST+1):
		 c_timers.timer_clear(i)
	c_timers.timer_start(1)
	# ---------------------------------------------------------------------
	# the timestep loop
	# ---------------------------------------------------------------------
	for istep in range(1, niter+1):
		if (istep%20)==0 or istep==itmax or istep==1:
			if niter > 1:
				print(" Time step %4d" % (istep))
		convergence = ssor_exec(istep, rsd, rsdnm, tolrsd,
								a, b, c, d, u,
								frct, rho_i, qs, flux,
								tv, delunm, tmp)
		
		if convergence:
			break
	#END for istep in range(1, niter+1):
	
	c_timers.timer_stop(1)
	maxtime = c_timers.timer_read(1)
#END ssor()

# ---------------------------------------------------------------------
# compute the right hand side based on exact solution
# ---------------------------------------------------------------------
@njit
def erhs(frct, rsd, flux):
	for k in range(nz): 
		for j in range(ny):
			for i in range(nx):
				for m in range(5):
					frct[k][j][i][m] = 0.0

	for k in range(nz):
		zeta = k / (nz-1)
		for j in range(ny):
			eta = j / (ny0-1 )
			for i in range(nx):
				xi = i / (nx0-1)
				for m in range(5):
					rsd[k][j][i][m] = ( ce[0][m]+
						(ce[1][m]+
						 (ce[4][m]+
						  (ce[7][m]+
						   ce[10][m]*xi)*xi)*xi)*xi+
						(ce[2][m]+
						 (ce[5][m]+
						  (ce[8][m]+
						   ce[11][m]*eta)*eta)*eta)*eta+
						(ce[3][m]+
						 (ce[6][m]+
						  (ce[9][m]+
						   ce[12][m]*zeta)*zeta)*zeta)*zeta )
	#END for k in range(nz)
	
	# ---------------------------------------------------------------------
	# xi-direction flux differences
	# ---------------------------------------------------------------------
	for k in range(1, nz-1): 
		for j in range(jst, jend):
			for i in range(nx):
				flux[i][0] = rsd[k][j][i][1]
				u21 = rsd[k][j][i][1]/rsd[k][j][i][0]
				q = ( 0.50*(rsd[k][j][i][1]*rsd[k][j][i][1]
						+rsd[k][j][i][2]*rsd[k][j][i][2]
						+rsd[k][j][i][3]*rsd[k][j][i][3])
					/rsd[k][j][i][0] )
				flux[i][1] = rsd[k][j][i][1]*u21+C2*(rsd[k][j][i][4]-q)
				flux[i][2] = rsd[k][j][i][2]*u21
				flux[i][3] = rsd[k][j][i][3]*u21
				flux[i][4] = (C1*rsd[k][j][i][4]-C2*q)*u21

			for i in range(ist, iend):
				for m in range(5):
					frct[k][j][i][m] = ( frct[k][j][i][m]
						-tx2*(flux[i+1][m]-flux[i-1][m]) )

			for i in range(ist, nx):
				tmp = 1.0/rsd[k][j][i][0]
				u21i = tmp*rsd[k][j][i][1]
				u31i = tmp*rsd[k][j][i][2]
				u41i = tmp*rsd[k][j][i][3]
				u51i = tmp*rsd[k][j][i][4]
				tmp = 1.0/rsd[k][j][i-1][0]
				u21im1 = tmp*rsd[k][j][i-1][1]
				u31im1 = tmp*rsd[k][j][i-1][2]
				u41im1 = tmp*rsd[k][j][i-1][3]
				u51im1 = tmp*rsd[k][j][i-1][4]
				flux[i][1] = (4.0/3.0)*tx3*(u21i-u21im1)
				flux[i][2] = tx3*(u31i-u31im1)
				flux[i][3] = tx3*(u41i-u41im1)
				flux[i][4] = ( 0.50*(1.0-C1*C5)
					*tx3*((u21i*u21i+u31i*u31i+u41i*u41i)
							-(u21im1*u21im1+u31im1*u31im1+u41im1*u41im1))
					+(1.0/6.0)
					*tx3*(u21i*u21i-u21im1*u21im1)
					+C1*C5*tx3*(u51i-u51im1) )

			for i in range(ist, iend):
				frct[k][j][i][0] = ( frct[k][j][i][0]
					+dx1*tx1*(rsd[k][j][i-1][0]
							-2.0*rsd[k][j][i][0]
							+rsd[k][j][i+1][0]) )
				frct[k][j][i][1] = ( frct[k][j][i][1]
					+tx3*C3*C4*(flux[i+1][1]-flux[i][1])
					+dx2*tx1*(rsd[k][j][i-1][1]
							-2.0*rsd[k][j][i][1]
							+rsd[k][j][i+1][1]) )
				frct[k][j][i][2] = ( frct[k][j][i][2]
					+tx3*C3*C4*(flux[i+1][2]-flux[i][2])
					+dx3*tx1*(rsd[k][j][i-1][2]
							-2.0*rsd[k][j][i][2]
							+rsd[k][j][i+1][2]) )
				frct[k][j][i][3] = ( frct[k][j][i][3]
					+tx3*C3*C4*(flux[i+1][3]-flux[i][3])
					+dx4*tx1*(rsd[k][j][i-1][3]
							-2.0*rsd[k][j][i][3]
							+rsd[k][j][i+1][3]) )
				frct[k][j][i][4] = ( frct[k][j][i][4]
					+tx3*C3*C4*(flux[i+1][4]-flux[i][4])
					+dx5*tx1*(rsd[k][j][i-1][4]
							-2.0*rsd[k][j][i][4]
							+rsd[k][j][i+1][4]) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			for m in range(5):
				frct[k][j][1][m] = ( frct[k][j][1][m]
					-dssp*(+5.0*rsd[k][j][1][m]
							-4.0*rsd[k][j][2][m]
							+rsd[k][j][3][m]) )
				frct[k][j][2][m] = ( frct[k][j][2][m]
					-dssp*(-4.0*rsd[k][j][1][m]
							+6.0*rsd[k][j][2][m]
							-4.0*rsd[k][j][3][m]
							+rsd[k][j][4][m]) )

			for i in range(3, nx-3):
				for m in range(5):
					frct[k][j][i][m] = ( frct[k][j][i][m]
						-dssp*(rsd[k][j][i-2][m]
								-4.0*rsd[k][j][i-1][m]
								+6.0*rsd[k][j][i][m]
								-4.0*rsd[k][j][i+1][m]
								+rsd[k][j][i+2][m] ) )

			for m in range(5):
				frct[k][j][nx-3][m] = ( frct[k][j][nx-3][m]
					-dssp*(rsd[k][j][nx-5][m]
							-4.0*rsd[k][j][nx-4][m]
							+6.0*rsd[k][j][nx-3][m]
							-4.0*rsd[k][j][nx-2][m] ) )
				frct[k][j][nx-2][m] = ( frct[k][j][nx-2][m]
					-dssp*(rsd[k][j][nx-4][m]
							-4.0*rsd[k][j][nx-3][m]
							+5.0*rsd[k][j][nx-2][m] ) )
		#END for j in range(jst, jend):
	#END for k in range(1, nz-1):
	
	# ---------------------------------------------------------------------
	# eta-direction flux differences
	# ---------------------------------------------------------------------
	for k in range(1, nz-1):
		for i in range(ist, iend):
			for j in range(ny): 
				flux[j][0] = rsd[k][j][i][2]
				u31 = rsd[k][j][i][2]/rsd[k][j][i][0]
				q = ( 0.50*(rsd[k][j][i][1]*rsd[k][j][i][1]
						+rsd[k][j][i][2]*rsd[k][j][i][2]
						+rsd[k][j][i][3]*rsd[k][j][i][3])
					/rsd[k][j][i][0] )
				flux[j][1] = rsd[k][j][i][1]*u31
				flux[j][2] = rsd[k][j][i][2]*u31+C2*(rsd[k][j][i][4]-q)
				flux[j][3] = rsd[k][j][i][3]*u31
				flux[j][4] = (C1*rsd[k][j][i][4]-C2*q)*u31

			for j in range(jst, jend): 
				for m in range(5):
					frct[k][j][i][m] = ( frct[k][j][i][m]
						-ty2*(flux[j+1][m]-flux[j-1][m]) )

			for j in range(jst, ny):
				tmp = 1.0/rsd[k][j][i][0]
				u21j = tmp*rsd[k][j][i][1]
				u31j = tmp*rsd[k][j][i][2]
				u41j = tmp*rsd[k][j][i][3]
				u51j = tmp*rsd[k][j][i][4]
				tmp = 1.0/rsd[k][j-1][i][0]
				u21jm1 = tmp*rsd[k][j-1][i][1]
				u31jm1 = tmp*rsd[k][j-1][i][2]
				u41jm1 = tmp*rsd[k][j-1][i][3]
				u51jm1 = tmp*rsd[k][j-1][i][4]
				flux[j][1] = ty3*(u21j-u21jm1)
				flux[j][2] = (4.0/3.0)*ty3*(u31j-u31jm1)
				flux[j][3] = ty3*(u41j-u41jm1)
				flux[j][4] = ( 0.50*(1.0-C1*C5)
					*ty3*((u21j*u21j+u31j*u31j+u41j*u41j)
							-(u21jm1*u21jm1+u31jm1*u31jm1+u41jm1*u41jm1))
					+(1.0/6.0)
					*ty3*(u31j*u31j-u31jm1*u31jm1)
					+C1*C5*ty3*(u51j-u51jm1) )

			for j in range(jst, jend):
				frct[k][j][i][0] = ( frct[k][j][i][0]
					+dy1*ty1*(rsd[k][j-1][i][0]
							-2.0*rsd[k][j][i][0]
							+rsd[k][j+1][i][0]) )
				frct[k][j][i][1] = ( frct[k][j][i][1]
					+ty3*C3*C4*(flux[j+1][1]-flux[j][1])
					+dy2*ty1*(rsd[k][j-1][i][1]
							-2.0*rsd[k][j][i][1]
							+rsd[k][j+1][i][1]) )
				frct[k][j][i][2] = ( frct[k][j][i][2]
					+ty3*C3*C4*(flux[j+1][2]-flux[j][2])
					+dy3*ty1*(rsd[k][j-1][i][2]
							-2.0*rsd[k][j][i][2]
							+rsd[k][j+1][i][2]) )
				frct[k][j][i][3] = ( frct[k][j][i][3]
					+ty3*C3*C4*(flux[j+1][3]-flux[j][3])
					+dy4*ty1*(rsd[k][j-1][i][3]
							-2.0*rsd[k][j][i][3]
							+rsd[k][j+1][i][3]) )
				frct[k][j][i][4] = ( frct[k][j][i][4]
					+ty3*C3*C4*(flux[j+1][4]-flux[j][4])
					+dy5*ty1*(rsd[k][j-1][i][4]
							-2.0*rsd[k][j][i][4]
							+rsd[k][j+1][i][4] ) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			for m in range(5):
				frct[k][1][i][m] = ( frct[k][1][i][m]
					-dssp*(+5.0*rsd[k][1][i][m]
							-4.0*rsd[k][2][i][m]
							+rsd[k][3][i][m]) )
				frct[k][2][i][m] = ( frct[k][2][i][m]
					-dssp*(-4.0*rsd[k][1][i][m]
							+6.0*rsd[k][2][i][m]
							-4.0*rsd[k][3][i][m]
							+rsd[k][4][i][m]) )

			for j in range(3, ny-3):
				for m in range(5):
					frct[k][j][i][m] = ( frct[k][j][i][m]
						-dssp*(rsd[k][j-2][i][m]
								-4.0*rsd[k][j-1][i][m]
								+6.0*rsd[k][j][i][m]
								-4.0*rsd[k][j+1][i][m]
								+rsd[k][j+2][i][m]) )

			for m in range(5):
				frct[k][ny-3][i][m] = ( frct[k][ny-3][i][m]
					-dssp*(rsd[k][ny-5][i][m]
							-4.0*rsd[k][ny-4][i][m]
							+6.0*rsd[k][ny-3][i][m]
							-4.0*rsd[k][ny-2][i][m]) )
				frct[k][ny-2][i][m] = ( frct[k][ny-2][i][m]
					-dssp*(rsd[k][ny-4][i][m]
							-4.0*rsd[k][ny-3][i][m]
							+5.0*rsd[k][ny-2][i][m]) )
		#END for i in range(ist, iend):
	#END for k in range(1, nz-1):
	# ---------------------------------------------------------------------
	# zeta-direction flux differences
	# ---------------------------------------------------------------------
	for j in range(jst, jend):
		for i in range(ist, iend):
			for k in range(nz):
				flux[k][0] = rsd[k][j][i][3]
				u41 = rsd[k][j][i][3]/rsd[k][j][i][0]
				q = ( 0.50*(rsd[k][j][i][1]*rsd[k][j][i][1]
						+rsd[k][j][i][2]*rsd[k][j][i][2]
						+rsd[k][j][i][3]*rsd[k][j][i][3])
					/rsd[k][j][i][0] )
				flux[k][1] = rsd[k][j][i][1]*u41
				flux[k][2] = rsd[k][j][i][2]*u41 
				flux[k][3] = rsd[k][j][i][3]*u41+C2*(rsd[k][j][i][4]-q)
				flux[k][4] = (C1*rsd[k][j][i][4]-C2*q)*u41

			for k in range(1, nz-1):
				for m in range(5):
					frct[k][j][i][m] = ( frct[k][j][i][m]
						-tz2*(flux[k+1][m]-flux[k-1][m]) )

			for k in range(1, nz):
				tmp = 1.0/rsd[k][j][i][0]
				u21k = tmp*rsd[k][j][i][1]
				u31k = tmp*rsd[k][j][i][2]
				u41k = tmp*rsd[k][j][i][3]
				u51k = tmp*rsd[k][j][i][4]
				tmp = 1.0/rsd[k-1][j][i][0]
				u21km1 = tmp*rsd[k-1][j][i][1]
				u31km1 = tmp*rsd[k-1][j][i][2]
				u41km1 = tmp*rsd[k-1][j][i][3]
				u51km1 = tmp*rsd[k-1][j][i][4]
				flux[k][1] = tz3*(u21k-u21km1)
				flux[k][2] = tz3*(u31k-u31km1)
				flux[k][3] = (4.0/3.0)*tz3*(u41k-u41km1)
				flux[k][4] = ( 0.50*(1.0-C1*C5)
					*tz3*((u21k*u21k+u31k*u31k+u41k*u41k)
							-(u21km1*u21km1+u31km1*u31km1+u41km1*u41km1))
					+(1.0/6.0)
					*tz3*(u41k*u41k-u41km1*u41km1)
					+C1*C5*tz3*(u51k-u51km1) )

			for k in range(1, nz-1):
				frct[k][j][i][0] = ( frct[k][j][i][0]
					+dz1*tz1*(rsd[k+1][j][i][0]
							-2.0*rsd[k][j][i][0]
							+rsd[k-1][j][i][0]) )
				frct[k][j][i][1] = ( frct[k][j][i][1]
					+tz3*C3*C4*(flux[k+1][1]-flux[k][1])
					+dz2*tz1*(rsd[k+1][j][i][1]
							-2.0*rsd[k][j][i][1]
							+rsd[k-1][j][i][1]) )
				frct[k][j][i][2] = ( frct[k][j][i][2]
					+tz3*C3*C4*(flux[k+1][2]-flux[k][2])
					+dz3*tz1*(rsd[k+1][j][i][2]
							-2.0*rsd[k][j][i][2]
							+rsd[k-1][j][i][2]) )
				frct[k][j][i][3] = ( frct[k][j][i][3]
					+tz3*C3*C4*(flux[k+1][3]-flux[k][3])
					+dz4*tz1*(rsd[k+1][j][i][3]
							-2.0*rsd[k][j][i][3]
							+rsd[k-1][j][i][3]) )
				frct[k][j][i][4] = ( frct[k][j][i][4]
					+tz3*C3*C4*(flux[k+1][4]-flux[k][4])
					+dz5*tz1*(rsd[k+1][j][i][4]
							-2.0*rsd[k][j][i][4]
							+rsd[k-1][j][i][4]) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation
			# ---------------------------------------------------------------------
			for m in range(5):
				frct[1][j][i][m] = ( frct[1][j][i][m]
					-dssp*(+5.0*rsd[1][j][i][m]
							-4.0*rsd[2][j][i][m]
							+rsd[3][j][i][m]) )
				frct[2][j][i][m] = ( frct[2][j][i][m]
					-dssp*(-4.0*rsd[1][j][i][m]
							+6.0*rsd[2][j][i][m]
							-4.0*rsd[3][j][i][m]
							+rsd[4][j][i][m]) )

			for k in range(3, nz-3):
				for m in range(5):
					frct[k][j][i][m] = ( frct[k][j][i][m]
						-dssp*(rsd[k-2][j][i][m]
								-4.0*rsd[k-1][j][i][m]
								+6.0*rsd[k][j][i][m]
								-4.0*rsd[k+1][j][i][m]
								+rsd[k+2][j][i][m]) )

			for m in range(5):
				frct[nz-3][j][i][m] = ( frct[nz-3][j][i][m]
					-dssp*(rsd[nz-5][j][i][m]
							-4.0*rsd[nz-4][j][i][m]
							+6.0*rsd[nz-3][j][i][m]
							-4.0*rsd[nz-2][j][i][m]) )
				frct[nz-2][j][i][m] = ( frct[nz-2][j][i][m]
					-dssp*(rsd[nz-4][j][i][m]
							-4.0*rsd[nz-3][j][i][m]
							+5.0*rsd[nz-2][j][i][m]) )
		#END for i in range(ist, iend):
	#END for j in range(jst, jend):
#END erhs()


# ---------------------------------------------------------------------
# compute the exact solution at (i,j,k)
# ---------------------------------------------------------------------
@njit
def exact(i, j, k, u000ijk):
	xi = i / (nx0-1)
	eta = j / (ny0-1)
	zeta = k / (nz-1)
	for m in range(5):
		u000ijk[m] = ( ce[0][m]+
			(ce[1][m]+
			 (ce[4][m]+
			  (ce[7][m]+
			   ce[10][m]*xi)*xi)*xi)*xi+
			(ce[2][m]+
			 (ce[5][m]+
			  (ce[8][m]+
			   ce[11][m]*eta)*eta)*eta)*eta+
			(ce[3][m]+
			 (ce[6][m]+
			  (ce[9][m]+
			   ce[12][m]*zeta)*zeta)*zeta)*zeta )
#END exact()

# ---------------------------------------------------------------------
# set the initial values of independent variables based on tri-linear
# interpolation of boundary values in the computational space.
# ---------------------------------------------------------------------
@njit
def setiv(u):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	ue_1jk = numpy.empty(5, dtype=numpy.float64)
	ue_nx0jk = numpy.empty(5, dtype=numpy.float64)
	ue_i1k = numpy.empty(5, dtype=numpy.float64)
	ue_iny0k = numpy.empty(5, dtype=numpy.float64)
	ue_ij1 = numpy.empty(5, dtype=numpy.float64)
	ue_ijnz = numpy.empty(5, dtype=numpy.float64)
	for k in range(1, nz-1):
		zeta = k / (nz-1)
		for j in range(1, ny-1):
			eta = j / (ny0-1)
			for i in range(1, nx-1):
				xi = i / (nx0-1)
				exact(0, j, k, ue_1jk)
				exact(nx0-1, j, k, ue_nx0jk)
				exact(i, 0, k, ue_i1k)
				exact(i, ny0-1, k, ue_iny0k)
				exact(i, j, 0, ue_ij1)
				exact(i, j, nz-1, ue_ijnz)
				for m in range(5):
					pxi = ( (1.0-xi)*ue_1jk[m]
						+xi*ue_nx0jk[m] )
					peta = ( (1.0-eta)*ue_i1k[m]
						+eta*ue_iny0k[m] )
					pzeta = ( (1.0-zeta)*ue_ij1[m]
						+zeta*ue_ijnz[m] )
					u[k][j][i][m] = ( pxi+peta+pzeta
						-pxi*peta-peta*pzeta-pzeta*pxi
						+pxi*peta*pzeta )
#END setiv()


# ---------------------------------------------------------------------
# set the boundary values of dependent variables
# ---------------------------------------------------------------------
@njit
def setbv(u):
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	temp1 = numpy.empty(5, dtype=numpy.float64)
	temp2 = numpy.empty(5, dtype=numpy.float64)
	# ---------------------------------------------------------------------
	# set the dependent variable values along the top and bottom faces
	# ---------------------------------------------------------------------
	for j in range(ny):
		for i in range(nx):
			exact(i, j, 0, temp1)
			exact(i, j, nz-1, temp2)
			for m in range(5):
				u[0][j][i][m] = temp1[m]
				u[nz-1][j][i][m] = temp2[m]

	# ---------------------------------------------------------------------
	# set the dependent variable values along north and south faces
	# ---------------------------------------------------------------------
	for k in range(nz):
		for i in range(nx):
			exact(i, 0, k, temp1)
			exact(i, ny-1, k, temp2)
			for m in range(5):
				u[k][0][i][m] = temp1[m]
				u[k][ny-1][i][m] = temp2[m]

	# ---------------------------------------------------------------------
	# set the dependent variable values along east and west faces
	# ---------------------------------------------------------------------
	for k in range(nz):
		for j in range(ny):
			exact(0, j, k, temp1)
			exact(nx-1, j, k, temp2)
			for m in range(5):
				u[k][j][0][m] = temp1[m]
				u[k][j][nx-1][m] = temp2[m]
#END setbv()


def setcoeff():
	global dxi, deta, dzeta
	global tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3
	global dx1, dx2, dx3, dx4, dx5
	global dy1, dy2, dy3, dy4, dy5
	global dz1, dz2, dz3, dz4, dz5
	global dssp
	global ce
	
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	# set up coefficients
	# ---------------------------------------------------------------------
	dxi = 1.0/(nx0-1)
	deta = 1.0/(ny0-1)
	dzeta = 1.0/(nz0-1)
	tx1 = 1.0/(dxi*dxi)
	tx2 = 1.0/(2.0*dxi)
	tx3 = 1.0/dxi
	ty1 = 1.0/(deta*deta)
	ty2 = 1.0/(2.0*deta)
	ty3 = 1.0/deta
	tz1 = 1.0/(dzeta*dzeta)
	tz2 = 1.0/(2.0*dzeta)
	tz3 = 1.0/dzeta
	# ---------------------------------------------------------------------
	# diffusion coefficients
	# ---------------------------------------------------------------------
	dx1 = 0.75
	dx2 = dx1
	dx3 = dx1
	dx4 = dx1
	dx5 = dx1
	dy1 = 0.75
	dy2 = dy1
	dy3 = dy1
	dy4 = dy1
	dy5 = dy1
	dz1 = 1.00
	dz2 = dz1
	dz3 = dz1
	dz4 = dz1
	dz5 = dz1
	# ---------------------------------------------------------------------
	# fourth difference dissipation
	# ---------------------------------------------------------------------
	dssp = (max(max(dx1,dy1),dz1))/4.0
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the first pde
	# ---------------------------------------------------------------------
	ce[0][0] = 2.0
	ce[1][0] = 0.0
	ce[2][0] = 0.0
	ce[3][0] = 4.0
	ce[4][0] = 5.0
	ce[5][0] = 3.0
	ce[6][0] = 5.0e-01
	ce[7][0] = 2.0e-02
	ce[8][0] = 1.0e-02
	ce[9][0] = 3.0e-02
	ce[10][0] = 5.0e-01
	ce[11][0] = 4.0e-01
	ce[12][0] = 3.0e-01
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the second pde
	# ---------------------------------------------------------------------
	ce[0][1] = 1.0
	ce[1][1] = 0.0
	ce[2][1] = 0.0
	ce[3][1] = 0.0
	ce[4][1] = 1.0
	ce[5][1] = 2.0
	ce[6][1] = 3.0
	ce[7][1] = 1.0e-02
	ce[8][1] = 3.0e-02
	ce[9][1] = 2.0e-02
	ce[10][1] = 4.0e-01
	ce[11][1] = 3.0e-01
	ce[12][1] = 5.0e-01
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the third pde
	# ---------------------------------------------------------------------
	ce[0][2] = 2.0
	ce[1][2] = 2.0
	ce[2][2] = 0.0
	ce[3][2] = 0.0
	ce[4][2] = 0.0
	ce[5][2] = 2.0
	ce[6][2] = 3.0
	ce[7][2] = 4.0e-02
	ce[8][2] = 3.0e-02
	ce[9][2] = 5.0e-02
	ce[10][2] = 3.0e-01
	ce[11][2] = 5.0e-01
	ce[12][2] = 4.0e-01
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the fourth pde
	# ---------------------------------------------------------------------
	ce[0][3] = 2.0
	ce[1][3] = 2.0
	ce[2][3] = 0.0
	ce[3][3] = 0.0
	ce[4][3] = 0.0
	ce[5][3] = 2.0
	ce[6][3] = 3.0
	ce[7][3] = 3.0e-02
	ce[8][3] = 5.0e-02
	ce[9][3] = 4.0e-02
	ce[10][3] = 2.0e-01
	ce[11][3] = 1.0e-01
	ce[12][3] = 3.0e-01
	# ---------------------------------------------------------------------
	# coefficients of the exact solution to the fifth pde
	# ---------------------------------------------------------------------
	ce[0][4] = 5.0
	ce[1][4] = 4.0
	ce[2][4] = 3.0
	ce[3][4] = 2.0
	ce[4][4] = 1.0e-01
	ce[5][4] = 4.0e-01
	ce[6][4] = 3.0e-01
	ce[7][4] = 5.0e-02
	ce[8][4] = 4.0e-02
	ce[9][4] = 3.0e-02
	ce[10][4] = 1.0e-01
	ce[11][4] = 3.0e-01
	ce[12][4] = 2.0e-01
#END setcoeff()


def domain():
	global nx, ny, nz
	global ist, iend, jst, jend
	global ii1, ii2, ji1, ji2, ki1, ki2
	
	# ---------------------------------------------------------------------
	# local variables
	# ---------------------------------------------------------------------
	nx = nx0
	ny = ny0
	nz = nz0
	# ---------------------------------------------------------------------
	# check the sub-domain size
	# ---------------------------------------------------------------------
	if (nx<4) or (ny<4) or (nz<4):
		print("     SUBDOMAIN SIZE IS TOO SMALL - \n"
				"     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
				"     SO THAT NX, NY AND NZ ARE GREATER THAN OR EQUAL\n"
				"     TO 4 THEY ARE CURRENTLY%3d%3d%3d", nx, ny, nz)
		sys.exit()

	if (nx>npbparams.ISIZ1) or (ny>npbparams.ISIZ2) or (nz>npbparams.ISIZ3):
		print("     SUBDOMAIN SIZE IS TOO LARGE - \n"
				"     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
				"     SO THAT NX, NY AND NZ ARE LESS THAN OR EQUAL TO \n"
				"     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY.  THEY ARE\n"
				"     CURRENTLYi%4d%4d%4d", nx, ny, nz)
		sys.exit()

	# ---------------------------------------------------------------------
	# set up the start and end in i and j extents for all processors
	# ---------------------------------------------------------------------
	ist = 1
	iend = nx-1
	jst = 1
	jend = ny-1
	ii1 = 1
	ii2 = nx0-1
	ji1 = 1
	ji2 = ny0-2
	ki1 = 2
	ki2 = nz0-1
#END domain()


def read_input():
	# ---------------------------------------------------------------------
	# if input file does not exist, it uses defaults
	# ipr = 1 for detailed progress output
	# inorm = how often the norm is printed (once every inorm iterations)
	# itmax = number of pseudo time steps
	# dt = time step
	# omega 1 over-relaxation factor for SSOR
	# tolrsd = steady state residual tolerance levels
	# nx, ny, nz = number of grid points in x, y, z directions
	# ---------------------------------------------------------------------
	global ipr, inorm, itmax
	global dt, omega
	global tolrsd
	global nx0, ny0, nz0
	
	fp = os.path.isfile("inputlu.data")
	if fp:
		print(" Reading from input file inputlu.data") 
		print(" ERROR - Not implemented") 
		sys.exit()
	else:
		ipr = IPR_DEFAULT
		inorm = npbparams.INORM_DEFAULT
		itmax = npbparams.ITMAX_DEFAULT
		dt = npbparams.DT_DEFAULT
		omega = OMEGA_DEFAULT
		tolrsd[0] = TOLRSD1_DEF
		tolrsd[1] = TOLRSD2_DEF
		tolrsd[2] = TOLRSD3_DEF
		tolrsd[3] = TOLRSD4_DEF
		tolrsd[4] = TOLRSD5_DEF
		nx0 = npbparams.ISIZ1
		ny0 = npbparams.ISIZ2
		nz0 = npbparams.ISIZ3
		
	# ---------------------------------------------------------------------
	# check problem size
	# ---------------------------------------------------------------------
	if (nx0<4) or (ny0<4) or (nz0<4):
		print("     PROBLEM SIZE IS TOO SMALL - \n"
				"     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5")
		sys.exit()
	
	if (nx0>npbparams.ISIZ1) or (ny0>npbparams.ISIZ2) or (nz0>npbparams.ISIZ3):
		print("     PROBLEM SIZE IS TOO LARGE - \n"
				"     NX, NY AND NZ SHOULD BE EQUAL TO \n"
				"     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY")
		sys.exit()
		
	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - LU Benchmark\n")
	print(" Size: %4dx%4dx%4d" % (nx0, ny0, nz0))
	print(" Iterations: %4d" % (itmax))
	print()
#END read_input()


def main():
	global u, rsd
	global frct, flux
	global rsdnm, errnm
	global frc

	# ---------------------------------------------------------------------
	# setup info for timers
	# ---------------------------------------------------------------------
	t_names = numpy.empty(T_LAST+1, dtype=object)
	timeron = os.path.isfile("timer.flag")
	if timeron:
		t_names[T_TOTAL] = "total"
		t_names[T_RHSX] = "rhsx*"
		t_names[T_RHSY] = "rhsy*"
		t_names[T_RHSZ] = "rhsz*"
		t_names[T_RHS] = "rhs*"
		t_names[T_JACLD] = "jacld*"
		t_names[T_BLTS] = "blts*"
		t_names[T_JACU] = "jacu*"
		t_names[T_BUTS] = "buts*"
		t_names[T_ADD] = "add*"
		t_names[T_L2NORM] = "l2norm*"
	
	# ---------------------------------------------------------------------
	# read input data
	# ---------------------------------------------------------------------
	read_input()
	# ---------------------------------------------------------------------
	# set up domain sizes
	# ---------------------------------------------------------------------
	domain()
	# ---------------------------------------------------------------------
	# set up coefficients
	# ---------------------------------------------------------------------
	setcoeff()
	# ---------------------------------------------------------------------
	# set the boundary values for dependent variables
	# ---------------------------------------------------------------------
	setbv(u)
	# ---------------------------------------------------------------------
	# set the initial values for dependent variables
	# ---------------------------------------------------------------------
	setiv(u)
	# ---------------------------------------------------------------------
	# compute the forcing term based on prescribed exact solution
	# ---------------------------------------------------------------------
	erhs(frct, rsd, flux)
	# ---------------------------------------------------------------------
	# perform one SSOR iteration to touch all pages
	# ---------------------------------------------------------------------
	ssor(1, u, rsd, frct, flux)
	# ---------------------------------------------------------------------
	# reset the boundary and initial values
	# ---------------------------------------------------------------------
	setbv(u)
	setiv(u)
	# ---------------------------------------------------------------------
	# perform the SSOR iterations
	# ---------------------------------------------------------------------
	ssor(itmax, u, rsd, frct, flux)
	# ---------------------------------------------------------------------
	# compute the solution error
	# ---------------------------------------------------------------------
	error(errnm, u)
	# ---------------------------------------------------------------------
	# compute the surface integral
	# ---------------------------------------------------------------------
	frc = pintgr(u)
	# ---------------------------------------------------------------------
	# verification test
	# ---------------------------------------------------------------------
	verified = verify(rsdnm, errnm, frc)
	mflops = ( itmax*(1984.77 * nx0
			* ny0
			* nz0
			- 10923.3 * pow(((nx0+ny0+nz0)/3.0),2.0) 
			+ 27770.9 * (nx0+ny0+nz0)/3.0
			- 144010.0) / (maxtime*1000000.0) )
	
	c_print_results.c_print_results("LU",
			npbparams.CLASS,
			nx0, 
			ny0,
			nz0,
			itmax,
			maxtime,
			mflops,
			"          floating point",
			verified)

	# ---------------------------------------------------------------------
	# more timers
	# ---------------------------------------------------------------------
	if timeron:
		trecs = numpy.empty(T_LAST+1, dtype=numpy.float64)
		for i in range(1, T_LAST+1):
			trecs[i] = c_timers.timer_read(i)
		tmax = maxtime
		if tmax == 0.0:
			tmax = 1.0
		print("  SECTION   Time (secs)")
		for i in range(1, T_LAST+1):
			print("  %-8s:%9.3f  (%6.2f%%)" % (t_names[i], trecs[i], trecs[i]*100/tmax))
			if i == T_RHS:
				t = trecs[T_RHSX] + trecs[T_RHSY] + trecs[T_RHSZ]
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("sub-rhs", t, t*100/tmax))
				t = trecs[i] - t
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("rest-rhs", t, t*100/tmax))
		print("  (* Time hasn't gauged: operation is not supported by @njit)")
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER LU')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_lu_info(args.CLASS)
	set_global_variables()
	
	main()
