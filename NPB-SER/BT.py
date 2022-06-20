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


# Global variables
IMAX = 0
JMAX = 0
KMAX = 0
IMAXP = 0
JMAXP = 0
AA = 0
BB = 1
CC = 2
BLOCK_SIZE = 5
T_TOTAL = 1
T_RHSX = 2
T_RHSY = 3
T_RHSZ = 4
T_RHS = 5
T_XSOLVE = 6
T_YSOLVE = 7
T_ZSOLVE = 8
T_RDIS1 = 9
T_RDIS2 = 10
T_ADD = 11
T_LAST = 11

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
ce = numpy.empty((13, 5), dtype=numpy.float64())

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
dnxm1, dnym1, dnzm1 = 0.0, 0.0, 0.0
c1c2, c1c5, c3c4, c1345, conz1 = 0.0, 0.0, 0.0, 0.0, 0.0
c1, c2, c3, c4, c5 = 0.0, 0.0, 0.0, 0.0, 0.0
c4dssp, c5dssp, dtdssp = 0.0, 0.0, 0.0
dttx1, dttx2, dtty1, dtty2, dttz1, dttz2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
c2dttx1, c2dtty1, c2dttz1 = 0.0, 0.0, 0.0
comz1, comz4, comz5, comz6 = 0.0, 0.0, 0.0, 0.0
c3c4tx3, c3c4ty3, c3c4tz3 = 0.0, 0.0, 0.0
c2iv, con43, con16 = 0.0, 0.0, 0.0
tmp1, tmp2, tmp3 = 0.0, 0.0, 0.0

grid_points = numpy.empty(3, dtype=numpy.int32)

timeron = False

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
	ue = numpy.empty((5, npbparams.PROBLEM_SIZE+1), dtype=numpy.float64())
	buf = numpy.empty((5, npbparams.PROBLEM_SIZE+1), dtype=numpy.float64())
	
	fjac = numpy.zeros((npbparams.PROBLEM_SIZE+1, 5, 5), dtype=numpy.float64())
	njac = numpy.zeros((npbparams.PROBLEM_SIZE+1, 5, 5), dtype=numpy.float64())
	
	lhs = numpy.zeros((npbparams.PROBLEM_SIZE+1, 3, 5, 5), dtype=numpy.float64())
#END set_global_variables()


def set_constants():
	global ce
	global c1, c2, c3, c4, c5
	global dnxm1, dnym1, dnzm1
	global c1c2, c1c5, c3c4, c1345, conz1
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
	
	ce[0][0] = 2.0
	ce[1][0] = 0.0
	ce[2][0] = 0.0
	ce[3][0] = 4.0
	ce[4][0] = 5.0
	ce[5][0] = 3.0
	ce[6][0] = 0.5
	ce[7][0] = 0.02
	ce[8][0] = 0.01
	ce[9][0] = 0.03
	ce[10][0] = 0.5
	ce[11][0] = 0.4
	ce[12][0] = 0.3
	# -------
	ce[0][1] = 1.0
	ce[1][1] = 0.0
	ce[2][1] = 0.0
	ce[3][1] = 0.0
	ce[4][1] = 1.0
	ce[5][1] = 2.0
	ce[6][1] = 3.0
	ce[7][1] = 0.01
	ce[8][1] = 0.03
	ce[9][1] = 0.02
	ce[10][1] = 0.4
	ce[11][1] = 0.3
	ce[12][1] = 0.5
	# -------
	ce[0][2] = 2.0
	ce[1][2] = 2.0
	ce[2][2] = 0.0
	ce[3][2] = 0.0
	ce[4][2] = 0.0
	ce[5][2] = 2.0
	ce[6][2] = 3.0
	ce[7][2] = 0.04
	ce[8][2] = 0.03
	ce[9][2] = 0.05
	ce[10][2] = 0.3
	ce[11][2] = 0.5
	ce[12][2] = 0.4
	# -------
	ce[0][3] = 2.0
	ce[1][3] = 2.0
	ce[2][3] = 0.0
	ce[3][3] = 0.0
	ce[4][3] = 0.0
	ce[5][3] = 2.0
	ce[6][3] = 3.0
	ce[7][3] = 0.03
	ce[8][3] = 0.05
	ce[9][3] = 0.04
	ce[10][3] = 0.2
	ce[11][3] = 0.1
	ce[12][3] = 0.3
	# -------
	ce[0][4] = 5.0
	ce[1][4] = 4.0
	ce[2][4] = 3.0
	ce[3][4] = 2.0
	ce[4][4] = 0.1
	ce[5][4] = 0.4
	ce[6][4] = 0.3
	ce[7][4] = 0.05
	ce[8][4] = 0.04
	ce[9][4] = 0.03
	ce[10][4] = 0.1
	ce[11][4] = 0.3
	ce[12][4] = 0.2
	# -------
	c1 = 1.4
	c2 = 0.4
	c3 = 0.1
	c4 = 1.0
	c5 = 1.4
	dnxm1 = 1.0 / (grid_points[0]-1)
	dnym1 = 1.0 / (grid_points[1]-1)
	dnzm1 = 1.0 / (grid_points[2]-1)
	c1c2 = c1 * c2
	c1c5 = c1 * c5
	c3c4 = c3 * c4
	c1345 = c1c5 * c3c4
	conz1 = (1.0-c1c5)
	tx1 = 1.0 / (dnxm1*dnxm1)
	tx2 = 1.0 / (2.0*dnxm1)
	tx3 = 1.0 / dnxm1
	ty1 = 1.0 / (dnym1*dnym1)
	ty2 = 1.0 / (2.0*dnym1)
	ty3 = 1.0 / dnym1
	tz1 = 1.0 / (dnzm1*dnzm1)
	tz2 = 1.0 / (2.0*dnzm1)
	tz3 = 1.0 / dnzm1
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
	xxcon3 = c3c4tx3 * conz1 * tx3
	xxcon4 = c3c4tx3 * con16 * tx3
	xxcon5 = c3c4tx3 * c1c5 * tx3
	yycon1 = c3c4ty3 * con43 * ty3
	yycon2 = c3c4ty3 * ty3
	yycon3 = c3c4ty3 * conz1 * ty3
	yycon4 = c3c4ty3 * con16 * ty3
	yycon5 = c3c4ty3 * c1c5 * ty3
	zzcon1 = c3c4tz3 * con43 * tz3
	zzcon2 = c3c4tz3 * tz3
	zzcon3 = c3c4tz3 * conz1 * tz3
	zzcon4 = c3c4tz3 * con16 * tz3
	zzcon5 = c3c4tz3 * c1c5 * tz3
#END set_constants()


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
		zeta = k * dnzm1
		for j in range(grid_points[1]):
			eta = j * dnym1
			for i in range(grid_points[0]):
				xi = i * dnxm1
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
	global rho_i, us, vs, ws, square, qs, rhs, u
	
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
	compute_rhs(rho_i, us, vs, ws, square, qs, rhs, u)
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
	print(" Verification being performed for class_npb %c" % (npbparams.CLASS))
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


#void binvrhs(double lhs[5][5], double r[5]){
@njit
def binvrhs(lhs, r):
	pivot = 1.00 / lhs[0][0]
	lhs[1][0] = lhs[1][0]*pivot
	lhs[2][0] = lhs[2][0]*pivot
	lhs[3][0] = lhs[3][0]*pivot
	lhs[4][0] = lhs[4][0]*pivot
	r[0] = r[0]*pivot
	# -------
	coeff = lhs[0][1]
	lhs[1][1] = lhs[1][1]-coeff*lhs[1][0]
	lhs[2][1] = lhs[2][1]-coeff*lhs[2][0]
	lhs[3][1] = lhs[3][1]-coeff*lhs[3][0]
	lhs[4][1] = lhs[4][1]-coeff*lhs[4][0]
	r[1] = r[1]-coeff*r[0]
	# -------
	coeff = lhs[0][2]
	lhs[1][2] = lhs[1][2]-coeff*lhs[1][0]
	lhs[2][2] = lhs[2][2]-coeff*lhs[2][0]
	lhs[3][2] = lhs[3][2]-coeff*lhs[3][0]
	lhs[4][2] = lhs[4][2]-coeff*lhs[4][0]
	r[2] = r[2]-coeff*r[0]
	# -------
	coeff = lhs[0][3]
	lhs[1][3] = lhs[1][3]-coeff*lhs[1][0]
	lhs[2][3] = lhs[2][3]-coeff*lhs[2][0]
	lhs[3][3] = lhs[3][3]-coeff*lhs[3][0]
	lhs[4][3] = lhs[4][3]-coeff*lhs[4][0]
	r[3] = r[3]-coeff*r[0]
	# -------
	coeff = lhs[0][4]
	lhs[1][4] = lhs[1][4]-coeff*lhs[1][0]
	lhs[2][4] = lhs[2][4]-coeff*lhs[2][0]
	lhs[3][4] = lhs[3][4]-coeff*lhs[3][0]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][0]
	r[4] = r[4]-coeff*r[0]
	# -------
	pivot = 1.00 / lhs[1][1]
	lhs[2][1] = lhs[2][1]*pivot
	lhs[3][1] = lhs[3][1]*pivot
	lhs[4][1] = lhs[4][1]*pivot
	r[1] = r[1]*pivot
	# -------
	coeff = lhs[1][0]
	lhs[2][0] = lhs[2][0]-coeff*lhs[2][1]
	lhs[3][0] = lhs[3][0]-coeff*lhs[3][1]
	lhs[4][0] = lhs[4][0]-coeff*lhs[4][1]
	r[0] = r[0]-coeff*r[1]
	# -------
	coeff = lhs[1][2]
	lhs[2][2] = lhs[2][2]-coeff*lhs[2][1]
	lhs[3][2] = lhs[3][2]-coeff*lhs[3][1]
	lhs[4][2] = lhs[4][2]-coeff*lhs[4][1]
	r[2] = r[2]-coeff*r[1]
	# -------
	coeff = lhs[1][3]
	lhs[2][3] = lhs[2][3]-coeff*lhs[2][1]
	lhs[3][3] = lhs[3][3]-coeff*lhs[3][1]
	lhs[4][3] = lhs[4][3]-coeff*lhs[4][1]
	r[3] = r[3]-coeff*r[1]
	# -------
	coeff = lhs[1][4]
	lhs[2][4] = lhs[2][4]-coeff*lhs[2][1]
	lhs[3][4] = lhs[3][4]-coeff*lhs[3][1]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][1]
	r[4] = r[4]-coeff*r[1]
	# -------
	pivot = 1.00 / lhs[2][2]
	lhs[3][2] = lhs[3][2]*pivot
	lhs[4][2] = lhs[4][2]*pivot
	r[2] = r[2]*pivot
	# -------
	coeff = lhs[2][0]
	lhs[3][0] = lhs[3][0]-coeff*lhs[3][2]
	lhs[4][0] = lhs[4][0]-coeff*lhs[4][2]
	r[0] = r[0]-coeff*r[2]
	# -------
	coeff = lhs[2][1]
	lhs[3][1] = lhs[3][1]-coeff*lhs[3][2]
	lhs[4][1] = lhs[4][1]-coeff*lhs[4][2]
	r[1] = r[1]-coeff*r[2]
	# -------
	coeff = lhs[2][3]
	lhs[3][3] = lhs[3][3]-coeff*lhs[3][2]
	lhs[4][3] = lhs[4][3]-coeff*lhs[4][2]
	r[3] = r[3]-coeff*r[2]
	# -------
	coeff = lhs[2][4]
	lhs[3][4] = lhs[3][4]-coeff*lhs[3][2]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][2]
	r[4] = r[4]-coeff*r[2]
	# -------
	pivot = 1.00/lhs[3][3]
	lhs[4][3] = lhs[4][3]*pivot
	r[3] = r[3]*pivot
	# -------
	coeff = lhs[3][0]
	lhs[4][0] = lhs[4][0]-coeff*lhs[4][3]
	r[0] = r[0]-coeff*r[3]
	# -------
	coeff = lhs[3][1]
	lhs[4][1] = lhs[4][1]-coeff*lhs[4][3]
	r[1] = r[1]-coeff*r[3]
	# -------
	coeff = lhs[3][2]
	lhs[4][2] = lhs[4][2]-coeff*lhs[4][3]
	r[2] = r[2]-coeff*r[3]
	# -------
	coeff = lhs[3][4]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][3]
	r[4] = r[4]-coeff*r[3]
	# -------
	pivot = 1.00/lhs[4][4]
	r[4] = r[4]*pivot
	# -------
	coeff = lhs[4][0]
	r[0] = r[0]-coeff*r[4]
	# -------
	coeff = lhs[4][1]
	r[1] = r[1]-coeff*r[4]
	# -------
	coeff = lhs[4][2]
	r[2] = r[2]-coeff*r[4]
	# -------
	coeff = lhs[4][3]
	r[3] = r[3]-coeff*r[4]
#END binvrhs()


# ---------------------------------------------------------------------
# subtracts a(i,j,k) X b(i,j,k) from c(i,j,k)
# ---------------------------------------------------------------------
#void matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5])
@njit
def matmul_sub(ablock, bblock, cblock):
	cblock[0][0] = ( cblock[0][0]-ablock[0][0]*bblock[0][0]
		-ablock[1][0]*bblock[0][1]
		-ablock[2][0]*bblock[0][2]
		-ablock[3][0]*bblock[0][3]
		-ablock[4][0]*bblock[0][4] )
	cblock[0][1] = ( cblock[0][1]-ablock[0][1]*bblock[0][0]
		-ablock[1][1]*bblock[0][1]
		-ablock[2][1]*bblock[0][2]
		-ablock[3][1]*bblock[0][3]
		-ablock[4][1]*bblock[0][4] )
	cblock[0][2] = ( cblock[0][2]-ablock[0][2]*bblock[0][0]
		-ablock[1][2]*bblock[0][1]
		-ablock[2][2]*bblock[0][2]
		-ablock[3][2]*bblock[0][3]
		-ablock[4][2]*bblock[0][4] )
	cblock[0][3] = ( cblock[0][3]-ablock[0][3]*bblock[0][0]
		-ablock[1][3]*bblock[0][1]
		-ablock[2][3]*bblock[0][2]
		-ablock[3][3]*bblock[0][3]
		-ablock[4][3]*bblock[0][4] )
	cblock[0][4] = ( cblock[0][4]-ablock[0][4]*bblock[0][0]
		-ablock[1][4]*bblock[0][1]
		-ablock[2][4]*bblock[0][2]
		-ablock[3][4]*bblock[0][3]
		-ablock[4][4]*bblock[0][4] )
	cblock[1][0] = ( cblock[1][0]-ablock[0][0]*bblock[1][0]
		-ablock[1][0]*bblock[1][1]
		-ablock[2][0]*bblock[1][2]
		-ablock[3][0]*bblock[1][3]
		-ablock[4][0]*bblock[1][4] )
	cblock[1][1] = ( cblock[1][1]-ablock[0][1]*bblock[1][0]
		-ablock[1][1]*bblock[1][1]
		-ablock[2][1]*bblock[1][2]
		-ablock[3][1]*bblock[1][3]
		-ablock[4][1]*bblock[1][4] )
	cblock[1][2] = ( cblock[1][2]-ablock[0][2]*bblock[1][0]
		-ablock[1][2]*bblock[1][1]
		-ablock[2][2]*bblock[1][2]
		-ablock[3][2]*bblock[1][3]
		-ablock[4][2]*bblock[1][4] )
	cblock[1][3] = ( cblock[1][3]-ablock[0][3]*bblock[1][0]
		-ablock[1][3]*bblock[1][1]
		-ablock[2][3]*bblock[1][2]
		-ablock[3][3]*bblock[1][3]
		-ablock[4][3]*bblock[1][4] )
	cblock[1][4] = ( cblock[1][4]-ablock[0][4]*bblock[1][0]
		-ablock[1][4]*bblock[1][1]
		-ablock[2][4]*bblock[1][2]
		-ablock[3][4]*bblock[1][3]
		-ablock[4][4]*bblock[1][4] )
	cblock[2][0] = ( cblock[2][0]-ablock[0][0]*bblock[2][0]
		-ablock[1][0]*bblock[2][1]
		-ablock[2][0]*bblock[2][2]
		-ablock[3][0]*bblock[2][3]
		-ablock[4][0]*bblock[2][4] )
	cblock[2][1] = ( cblock[2][1]-ablock[0][1]*bblock[2][0]
		-ablock[1][1]*bblock[2][1]
		-ablock[2][1]*bblock[2][2]
		-ablock[3][1]*bblock[2][3]
		-ablock[4][1]*bblock[2][4] )
	cblock[2][2] = ( cblock[2][2]-ablock[0][2]*bblock[2][0]
		-ablock[1][2]*bblock[2][1]
		-ablock[2][2]*bblock[2][2]
		-ablock[3][2]*bblock[2][3]
		-ablock[4][2]*bblock[2][4] )
	cblock[2][3] = ( cblock[2][3]-ablock[0][3]*bblock[2][0]
		-ablock[1][3]*bblock[2][1]
		-ablock[2][3]*bblock[2][2]
		-ablock[3][3]*bblock[2][3]
		-ablock[4][3]*bblock[2][4] )
	cblock[2][4] = ( cblock[2][4]-ablock[0][4]*bblock[2][0]
		-ablock[1][4]*bblock[2][1]
		-ablock[2][4]*bblock[2][2]
		-ablock[3][4]*bblock[2][3]
		-ablock[4][4]*bblock[2][4] )
	cblock[3][0] = ( cblock[3][0]-ablock[0][0]*bblock[3][0]
		-ablock[1][0]*bblock[3][1]
		-ablock[2][0]*bblock[3][2]
		-ablock[3][0]*bblock[3][3]
		-ablock[4][0]*bblock[3][4] )
	cblock[3][1] = ( cblock[3][1]-ablock[0][1]*bblock[3][0]
		-ablock[1][1]*bblock[3][1]
		-ablock[2][1]*bblock[3][2]
		-ablock[3][1]*bblock[3][3]
		-ablock[4][1]*bblock[3][4] )
	cblock[3][2] = ( cblock[3][2]-ablock[0][2]*bblock[3][0]
		-ablock[1][2]*bblock[3][1]
		-ablock[2][2]*bblock[3][2]
		-ablock[3][2]*bblock[3][3]
		-ablock[4][2]*bblock[3][4] )
	cblock[3][3] = ( cblock[3][3]-ablock[0][3]*bblock[3][0]
		-ablock[1][3]*bblock[3][1]
		-ablock[2][3]*bblock[3][2]
		-ablock[3][3]*bblock[3][3]
		-ablock[4][3]*bblock[3][4] )
	cblock[3][4] = ( cblock[3][4]-ablock[0][4]*bblock[3][0]
		-ablock[1][4]*bblock[3][1]
		-ablock[2][4]*bblock[3][2]
		-ablock[3][4]*bblock[3][3]
		-ablock[4][4]*bblock[3][4] )
	cblock[4][0] = ( cblock[4][0]-ablock[0][0]*bblock[4][0]
		-ablock[1][0]*bblock[4][1]
		-ablock[2][0]*bblock[4][2]
		-ablock[3][0]*bblock[4][3]
		-ablock[4][0]*bblock[4][4] )
	cblock[4][1] = ( cblock[4][1]-ablock[0][1]*bblock[4][0]
		-ablock[1][1]*bblock[4][1]
		-ablock[2][1]*bblock[4][2]
		-ablock[3][1]*bblock[4][3]
		-ablock[4][1]*bblock[4][4] )
	cblock[4][2] = ( cblock[4][2]-ablock[0][2]*bblock[4][0]
		-ablock[1][2]*bblock[4][1]
		-ablock[2][2]*bblock[4][2]
		-ablock[3][2]*bblock[4][3]
		-ablock[4][2]*bblock[4][4] )
	cblock[4][3] = ( cblock[4][3]-ablock[0][3]*bblock[4][0]
		-ablock[1][3]*bblock[4][1]
		-ablock[2][3]*bblock[4][2]
		-ablock[3][3]*bblock[4][3]
		-ablock[4][3]*bblock[4][4] )
	cblock[4][4] = ( cblock[4][4]-ablock[0][4]*bblock[4][0]
		-ablock[1][4]*bblock[4][1]
		-ablock[2][4]*bblock[4][2]
		-ablock[3][4]*bblock[4][3]
		-ablock[4][4]*bblock[4][4] )
#END matmul_sub()


# ---------------------------------------------------------------------
# subtracts bvec=bvec - ablock*avec
# ---------------------------------------------------------------------
#void matvec_sub(double ablock[5][5], double avec[5], double bvec[5])
@njit
def matvec_sub(ablock, avec, bvec):
	# ---------------------------------------------------------------------
	# rhs[kc][jc][ic][i] = rhs[kc][jc][ic][i] - lhs[ia][ablock][0][i]*
	# ---------------------------------------------------------------------
	bvec[0] = ( bvec[0]-ablock[0][0]*avec[0]
		-ablock[1][0]*avec[1]
		-ablock[2][0]*avec[2]
		-ablock[3][0]*avec[3]
		-ablock[4][0]*avec[4] )
	bvec[1] = ( bvec[1]-ablock[0][1]*avec[0]
		-ablock[1][1]*avec[1]
		-ablock[2][1]*avec[2]
		-ablock[3][1]*avec[3]
		-ablock[4][1]*avec[4] )
	bvec[2] = ( bvec[2]-ablock[0][2]*avec[0]
		-ablock[1][2]*avec[1]
		-ablock[2][2]*avec[2]
		-ablock[3][2]*avec[3]
		-ablock[4][2]*avec[4] )
	bvec[3] = ( bvec[3]-ablock[0][3]*avec[0]
		-ablock[1][3]*avec[1]
		-ablock[2][3]*avec[2]
		-ablock[3][3]*avec[3]
		-ablock[4][3]*avec[4] )
	bvec[4] = ( bvec[4]-ablock[0][4]*avec[0]
		-ablock[1][4]*avec[1]
		-ablock[2][4]*avec[2]
		-ablock[3][4]*avec[3]
		-ablock[4][4]*avec[4] )
#END matvec_sub()


#void binvcrhs(double lhs[5][5], double c[5][5], double r[5]){
@njit
def binvcrhs(lhs, c, r):
	pivot = 1.00 / lhs[0][0]
	lhs[1][0] = lhs[1][0]*pivot
	lhs[2][0] = lhs[2][0]*pivot
	lhs[3][0] = lhs[3][0]*pivot
	lhs[4][0] = lhs[4][0]*pivot
	c[0][0] = c[0][0]*pivot
	c[1][0] = c[1][0]*pivot
	c[2][0] = c[2][0]*pivot
	c[3][0] = c[3][0]*pivot
	c[4][0] = c[4][0]*pivot
	r[0] = r[0]*pivot
	# -------
	coeff = lhs[0][1]
	lhs[1][1] = lhs[1][1]-coeff*lhs[1][0]
	lhs[2][1] = lhs[2][1]-coeff*lhs[2][0]
	lhs[3][1] = lhs[3][1]-coeff*lhs[3][0]
	lhs[4][1] = lhs[4][1]-coeff*lhs[4][0]
	c[0][1] = c[0][1]-coeff*c[0][0]
	c[1][1] = c[1][1]-coeff*c[1][0]
	c[2][1] = c[2][1]-coeff*c[2][0]
	c[3][1] = c[3][1]-coeff*c[3][0]
	c[4][1] = c[4][1]-coeff*c[4][0]
	r[1] = r[1]-coeff*r[0]
	# -------
	coeff = lhs[0][2]
	lhs[1][2] = lhs[1][2]-coeff*lhs[1][0]
	lhs[2][2] = lhs[2][2]-coeff*lhs[2][0]
	lhs[3][2] = lhs[3][2]-coeff*lhs[3][0]
	lhs[4][2] = lhs[4][2]-coeff*lhs[4][0]
	c[0][2] = c[0][2]-coeff*c[0][0]
	c[1][2] = c[1][2]-coeff*c[1][0]
	c[2][2] = c[2][2]-coeff*c[2][0]
	c[3][2] = c[3][2]-coeff*c[3][0]
	c[4][2] = c[4][2]-coeff*c[4][0]
	r[2] = r[2]-coeff*r[0]
	# -------
	coeff = lhs[0][3]
	lhs[1][3] = lhs[1][3]-coeff*lhs[1][0]
	lhs[2][3] = lhs[2][3]-coeff*lhs[2][0]
	lhs[3][3] = lhs[3][3]-coeff*lhs[3][0]
	lhs[4][3] = lhs[4][3]-coeff*lhs[4][0]
	c[0][3] = c[0][3]-coeff*c[0][0]
	c[1][3] = c[1][3]-coeff*c[1][0]
	c[2][3] = c[2][3]-coeff*c[2][0]
	c[3][3] = c[3][3]-coeff*c[3][0]
	c[4][3] = c[4][3]-coeff*c[4][0]
	r[3] = r[3]-coeff*r[0]
	# -------
	coeff = lhs[0][4]
	lhs[1][4] = lhs[1][4]-coeff*lhs[1][0]
	lhs[2][4] = lhs[2][4]-coeff*lhs[2][0]
	lhs[3][4] = lhs[3][4]-coeff*lhs[3][0]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][0]
	c[0][4] = c[0][4]-coeff*c[0][0]
	c[1][4] = c[1][4]-coeff*c[1][0]
	c[2][4] = c[2][4]-coeff*c[2][0]
	c[3][4] = c[3][4]-coeff*c[3][0]
	c[4][4] = c[4][4]-coeff*c[4][0]
	r[4] = r[4]-coeff*r[0]
	# -------
	pivot = 1.00 / lhs[1][1]
	lhs[2][1] = lhs[2][1]*pivot
	lhs[3][1] = lhs[3][1]*pivot
	lhs[4][1] = lhs[4][1]*pivot
	c[0][1] = c[0][1]*pivot
	c[1][1] = c[1][1]*pivot
	c[2][1] = c[2][1]*pivot
	c[3][1] = c[3][1]*pivot
	c[4][1] = c[4][1]*pivot
	r[1] = r[1]*pivot
	# -------
	coeff=lhs[1][0]
	lhs[2][0] = lhs[2][0]-coeff*lhs[2][1]
	lhs[3][0] = lhs[3][0]-coeff*lhs[3][1]
	lhs[4][0] = lhs[4][0]-coeff*lhs[4][1]
	c[0][0] = c[0][0]-coeff*c[0][1]
	c[1][0] = c[1][0]-coeff*c[1][1]
	c[2][0] = c[2][0]-coeff*c[2][1]
	c[3][0] = c[3][0]-coeff*c[3][1]
	c[4][0] = c[4][0]-coeff*c[4][1]
	r[0] = r[0]-coeff*r[1]
	# -------
	coeff = lhs[1][2]
	lhs[2][2] = lhs[2][2]-coeff*lhs[2][1]
	lhs[3][2] = lhs[3][2]-coeff*lhs[3][1]
	lhs[4][2] = lhs[4][2]-coeff*lhs[4][1]
	c[0][2] = c[0][2]-coeff*c[0][1]
	c[1][2] = c[1][2]-coeff*c[1][1]
	c[2][2] = c[2][2]-coeff*c[2][1]
	c[3][2] = c[3][2]-coeff*c[3][1]
	c[4][2] = c[4][2]-coeff*c[4][1]
	r[2] = r[2]-coeff*r[1]
	# -------
	coeff = lhs[1][3]
	lhs[2][3] = lhs[2][3]-coeff*lhs[2][1]
	lhs[3][3] = lhs[3][3]-coeff*lhs[3][1]
	lhs[4][3] = lhs[4][3]-coeff*lhs[4][1]
	c[0][3] = c[0][3]-coeff*c[0][1]
	c[1][3] = c[1][3]-coeff*c[1][1]
	c[2][3] = c[2][3]-coeff*c[2][1]
	c[3][3] = c[3][3]-coeff*c[3][1]
	c[4][3] = c[4][3]-coeff*c[4][1]
	r[3] = r[3]-coeff*r[1]
	# -------
	coeff = lhs[1][4]
	lhs[2][4] = lhs[2][4]-coeff*lhs[2][1]
	lhs[3][4] = lhs[3][4]-coeff*lhs[3][1]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][1]
	c[0][4] = c[0][4]-coeff*c[0][1]
	c[1][4] = c[1][4]-coeff*c[1][1]
	c[2][4] = c[2][4]-coeff*c[2][1]
	c[3][4] = c[3][4]-coeff*c[3][1]
	c[4][4] = c[4][4]-coeff*c[4][1]
	r[4] = r[4]-coeff*r[1]
	# -------
	pivot = 1.00 / lhs[2][2]
	lhs[3][2] = lhs[3][2]*pivot
	lhs[4][2] = lhs[4][2]*pivot
	c[0][2] = c[0][2]*pivot
	c[1][2] = c[1][2]*pivot
	c[2][2] = c[2][2]*pivot
	c[3][2] = c[3][2]*pivot
	c[4][2] = c[4][2]*pivot
	r[2] = r[2]*pivot
	# -------
	coeff = lhs[2][0]
	lhs[3][0] = lhs[3][0]-coeff*lhs[3][2]
	lhs[4][0] = lhs[4][0]-coeff*lhs[4][2]
	c[0][0] = c[0][0]-coeff*c[0][2]
	c[1][0] = c[1][0]-coeff*c[1][2]
	c[2][0] = c[2][0]-coeff*c[2][2]
	c[3][0] = c[3][0]-coeff*c[3][2]
	c[4][0] = c[4][0]-coeff*c[4][2]
	r[0] = r[0]-coeff*r[2]
	# -------
	coeff = lhs[2][1]
	lhs[3][1] = lhs[3][1]-coeff*lhs[3][2]
	lhs[4][1] = lhs[4][1]-coeff*lhs[4][2]
	c[0][1] = c[0][1]-coeff*c[0][2]
	c[1][1] = c[1][1]-coeff*c[1][2]
	c[2][1] = c[2][1]-coeff*c[2][2]
	c[3][1] = c[3][1]-coeff*c[3][2]
	c[4][1] = c[4][1]-coeff*c[4][2]
	r[1] = r[1]-coeff*r[2]
	# -------
	coeff = lhs[2][3]
	lhs[3][3] = lhs[3][3]-coeff*lhs[3][2]
	lhs[4][3] = lhs[4][3]-coeff*lhs[4][2]
	c[0][3] = c[0][3]-coeff*c[0][2]
	c[1][3] = c[1][3]-coeff*c[1][2]
	c[2][3] = c[2][3]-coeff*c[2][2]
	c[3][3] = c[3][3]-coeff*c[3][2]
	c[4][3] = c[4][3]-coeff*c[4][2]
	r[3] = r[3]-coeff*r[2]
	# -------
	coeff = lhs[2][4]
	lhs[3][4] = lhs[3][4]-coeff*lhs[3][2]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][2]
	c[0][4] = c[0][4]-coeff*c[0][2]
	c[1][4] = c[1][4]-coeff*c[1][2]
	c[2][4] = c[2][4]-coeff*c[2][2]
	c[3][4] = c[3][4]-coeff*c[3][2]
	c[4][4] = c[4][4]-coeff*c[4][2]
	r[4] = r[4]-coeff*r[2]
	# -------
	pivot = 1.00 / lhs[3][3]
	lhs[4][3] = lhs[4][3]*pivot
	c[0][3] = c[0][3]*pivot
	c[1][3] = c[1][3]*pivot
	c[2][3] = c[2][3]*pivot
	c[3][3] = c[3][3]*pivot
	c[4][3] = c[4][3]*pivot
	r[3] = r[3] *pivot
	# -------
	coeff = lhs[3][0]
	lhs[4][0] = lhs[4][0]-coeff*lhs[4][3]
	c[0][0] = c[0][0]-coeff*c[0][3]
	c[1][0] = c[1][0]-coeff*c[1][3]
	c[2][0] = c[2][0]-coeff*c[2][3]
	c[3][0] = c[3][0]-coeff*c[3][3]
	c[4][0] = c[4][0]-coeff*c[4][3]
	r[0] = r[0]-coeff*r[3]
	# -------
	coeff = lhs[3][1]
	lhs[4][1] = lhs[4][1]-coeff*lhs[4][3]
	c[0][1] = c[0][1]-coeff*c[0][3]
	c[1][1] = c[1][1]-coeff*c[1][3]
	c[2][1] = c[2][1]-coeff*c[2][3]
	c[3][1] = c[3][1]-coeff*c[3][3]
	c[4][1] = c[4][1]-coeff*c[4][3]
	r[1] = r[1]-coeff*r[3]
	# -------
	coeff = lhs[3][2]
	lhs[4][2] = lhs[4][2]-coeff*lhs[4][3]
	c[0][2] = c[0][2]-coeff*c[0][3]
	c[1][2] = c[1][2]-coeff*c[1][3]
	c[2][2] = c[2][2]-coeff*c[2][3]
	c[3][2] = c[3][2]-coeff*c[3][3]
	c[4][2] = c[4][2]-coeff*c[4][3]
	r[2] = r[2]-coeff*r[3]
	# -------
	coeff = lhs[3][4]
	lhs[4][4] = lhs[4][4]-coeff*lhs[4][3]
	c[0][4] = c[0][4]-coeff*c[0][3]
	c[1][4] = c[1][4]-coeff*c[1][3]
	c[2][4] = c[2][4]-coeff*c[2][3]
	c[3][4] = c[3][4]-coeff*c[3][3]
	c[4][4] = c[4][4]-coeff*c[4][3]
	r[4] = r[4]-coeff*r[3]
	# -------
	pivot = 1.00 / lhs[4][4]
	c[0][4] = c[0][4]*pivot
	c[1][4] = c[1][4]*pivot
	c[2][4] = c[2][4]*pivot
	c[3][4] = c[3][4]*pivot
	c[4][4] = c[4][4]*pivot
	r[4] = r[4]*pivot
	# -------
	coeff = lhs[4][0]
	c[0][0] = c[0][0]-coeff*c[0][4]
	c[1][0] = c[1][0]-coeff*c[1][4]
	c[2][0] = c[2][0]-coeff*c[2][4]
	c[3][0] = c[3][0]-coeff*c[3][4]
	c[4][0] = c[4][0]-coeff*c[4][4]
	r[0] = r[0]-coeff*r[4]
	# -------
	coeff = lhs[4][1]
	c[0][1] = c[0][1]-coeff*c[0][4]
	c[1][1] = c[1][1]-coeff*c[1][4]
	c[2][1] = c[2][1]-coeff*c[2][4]
	c[3][1] = c[3][1]-coeff*c[3][4]
	c[4][1] = c[4][1]-coeff*c[4][4]
	r[1] = r[1]-coeff*r[4]
	# -------
	coeff = lhs[4][2]
	c[0][2] = c[0][2]-coeff*c[0][4]
	c[1][2] = c[1][2]-coeff*c[1][4]
	c[2][2] = c[2][2]-coeff*c[2][4]
	c[3][2] = c[3][2]-coeff*c[3][4]
	c[4][2] = c[4][2]-coeff*c[4][4]
	r[2] = r[2]-coeff*r[4]
	# -------
	coeff = lhs[4][3]
	c[0][3] = c[0][3]-coeff*c[0][4]
	c[1][3] = c[1][3]-coeff*c[1][4]
	c[2][3] = c[2][3]-coeff*c[2][4]
	c[3][3] = c[3][3]-coeff*c[3][4]
	c[4][3] = c[4][3]-coeff*c[4][4]
	r[3] = r[3]-coeff*r[4]
#END binvcrhs()


#lhsinit(double lhs[][3][5][5], int size)
@njit
def lhsinit(lhs, size):
	i = size
	# ---------------------------------------------------------------------
	# zero the whole left hand side for starters
	# ---------------------------------------------------------------------
	for m in range(5):
		for n in range(5):
			lhs[0][0][n][m] = 0.0
			lhs[0][1][n][m] = 0.0
			lhs[0][2][n][m] = 0.0
			lhs[i][0][n][m] = 0.0
			lhs[i][1][n][m] = 0.0
			lhs[i][2][n][m] = 0.0
	# ---------------------------------------------------------------------
	# next, set all diagonal values to 1. This is overkill, but convenient
	# ---------------------------------------------------------------------
	for m in range(5):
		lhs[0][1][m][m] = 1.0
		lhs[i][1][m][m] = 1.0
#END lhsinit()


# ---------------------------------------------------------------------
# addition of update to the vector u
# ---------------------------------------------------------------------
@njit
def add(u, rhs):
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_ADD)
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				for m in range(5):
					u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m]
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_ADD)
#END add()


# ---------------------------------------------------------------------
# performs line solves in Z direction by first factoring
# the block-tridiagonal matrix into an upper triangular matrix, 
# and then performing back substitution to solve for the unknow
# vectors of each line.  
#  
# make sure we treat elements zero to cell_size in the direction
# of the sweep.
# ---------------------------------------------------------------------
@njit
def z_solve(fjac, njac, lhs, rhs, u, qs, square):
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_ZSOLVE)
	# ---------------------------------------------------------------------
	# this function computes the left hand side for the three z-factors   
	# ---------------------------------------------------------------------
	ksize = grid_points[2] - 1
	# ---------------------------------------------------------------------
	# compute the indices for storing the block-diagonal matrix;
	# determine c (labeled f) and s jacobians
	# ---------------------------------------------------------------------
	for j in range(1, grid_points[1]-1):
		for i in range(1, grid_points[0]-1):
			for k in range(ksize+1):
				tmp1 = 1.0 / u[k][j][i][0]
				tmp2 = tmp1 * tmp1
				tmp3 = tmp1 * tmp2
				fjac[k][0][0] = 0.0
				fjac[k][1][0] = 0.0
				fjac[k][2][0] = 0.0
				fjac[k][3][0] = 1.0
				fjac[k][4][0] = 0.0
				fjac[k][0][1] = -(u[k][j][i][1]*u[k][j][i][3])*tmp2
				fjac[k][1][1] = u[k][j][i][3]*tmp1
				fjac[k][2][1] = 0.0
				fjac[k][3][1] = u[k][j][i][1]*tmp1
				fjac[k][4][1] = 0.0
				fjac[k][0][2] = -(u[k][j][i][2]*u[k][j][i][3])*tmp2
				fjac[k][1][2] = 0.0
				fjac[k][2][2] = u[k][j][i][3]*tmp1
				fjac[k][3][2] = u[k][j][i][2]*tmp1
				fjac[k][4][2] = 0.0
				fjac[k][0][3] = -(u[k][j][i][3]*u[k][j][i][3]*tmp2)+c2*qs[k][j][i]
				fjac[k][1][3] = -c2*u[k][j][i][1]*tmp1
				fjac[k][2][3] = -c2*u[k][j][i][2]*tmp1
				fjac[k][3][3] = (2.0-c2)*u[k][j][i][3]*tmp1
				fjac[k][4][3] = c2
				fjac[k][0][4] = (c2*2.0*square[k][j][i]-c1*u[k][j][i][4])*u[k][j][i][3]*tmp2
				fjac[k][1][4] = -c2*(u[k][j][i][1]*u[k][j][i][3])*tmp2
				fjac[k][2][4] = -c2*(u[k][j][i][2]*u[k][j][i][3])*tmp2
				fjac[k][3][4] = c1*(u[k][j][i][4]*tmp1)-c2*(qs[k][j][i]+u[k][j][i][3]*u[k][j][i][3]*tmp2)
				fjac[k][4][4] = c1*u[k][j][i][3]*tmp1
				njac[k][0][0] = 0.0
				njac[k][1][0] = 0.0
				njac[k][2][0] = 0.0
				njac[k][3][0] = 0.0
				njac[k][4][0] = 0.0
				njac[k][0][1] = -c3c4*tmp2*u[k][j][i][1]
				njac[k][1][1] = c3c4*tmp1
				njac[k][2][1] = 0.0
				njac[k][3][1] = 0.0
				njac[k][4][1] = 0.0
				njac[k][0][2] = -c3c4*tmp2*u[k][j][i][2]
				njac[k][1][2] = 0.0
				njac[k][2][2] = c3c4*tmp1
				njac[k][3][2] = 0.0
				njac[k][4][2] = 0.0
				njac[k][0][3] = -con43*c3c4*tmp2*u[k][j][i][3]
				njac[k][1][3] = 0.0
				njac[k][2][3] = 0.0
				njac[k][3][3] = con43*c3*c4*tmp1
				njac[k][4][3] = 0.0
				njac[k][0][4] = ( -(c3c4-c1345)*tmp3*(u[k][j][i][1]*u[k][j][i][1])
					-(c3c4-c1345)*tmp3*(u[k][j][i][2]*u[k][j][i][2])
					-(con43*c3c4-c1345)*tmp3*(u[k][j][i][3]*u[k][j][i][3])
					-c1345*tmp2*u[k][j][i][4] )
				njac[k][1][4] = (c3c4-c1345)*tmp2*u[k][j][i][1]
				njac[k][2][4] = (c3c4-c1345)*tmp2*u[k][j][i][2]
				njac[k][3][4] = (con43*c3c4-c1345)*tmp2*u[k][j][i][3]
				njac[k][4][4] = (c1345)*tmp1
			#END for k in range(ksize+1):
			
			# ---------------------------------------------------------------------
			# now jacobians set, so form left hand side in z direction
			# ---------------------------------------------------------------------
			lhsinit(lhs, ksize)
			for k in range(1, ksize):
				tmp1 = dt * tz1
				tmp2 = dt * tz2
				lhs[k][AA][0][0] = ( -tmp2*fjac[k-1][0][0]
					-tmp1*njac[k-1][0][0]
					-tmp1*dz1 ) 
				lhs[k][AA][1][0] = ( -tmp2*fjac[k-1][1][0]
					-tmp1*njac[k-1][1][0] )
				lhs[k][AA][2][0] = ( -tmp2*fjac[k-1][2][0]
					-tmp1*njac[k-1][2][0] )
				lhs[k][AA][3][0] = ( -tmp2*fjac[k-1][3][0]
					-tmp1*njac[k-1][3][0] )
				lhs[k][AA][4][0] = ( -tmp2*fjac[k-1][4][0]
					-tmp1*njac[k-1][4][0] )
				lhs[k][AA][0][1] = ( -tmp2*fjac[k-1][0][1]
					-tmp1*njac[k-1][0][1] )
				lhs[k][AA][1][1] = ( -tmp2*fjac[k-1][1][1]
					-tmp1*njac[k-1][1][1]
					-tmp1*dz2 )
				lhs[k][AA][2][1] = ( -tmp2*fjac[k-1][2][1]
					-tmp1*njac[k-1][2][1] )
				lhs[k][AA][3][1] = ( -tmp2*fjac[k-1][3][1]
					-tmp1*njac[k-1][3][1] )
				lhs[k][AA][4][1] = ( -tmp2*fjac[k-1][4][1]
					-tmp1*njac[k-1][4][1] )
				lhs[k][AA][0][2] = ( -tmp2*fjac[k-1][0][2]
					-tmp1*njac[k-1][0][2] )
				lhs[k][AA][1][2] = ( -tmp2*fjac[k-1][1][2]
					-tmp1*njac[k-1][1][2] )
				lhs[k][AA][2][2] = ( -tmp2*fjac[k-1][2][2]
					-tmp1*njac[k-1][2][2]
					-tmp1*dz3 )
				lhs[k][AA][3][2] = ( -tmp2*fjac[k-1][3][2]
					-tmp1*njac[k-1][3][2] )
				lhs[k][AA][4][2] = ( -tmp2*fjac[k-1][4][2]
					-tmp1*njac[k-1][4][2] )
				lhs[k][AA][0][3] = ( -tmp2*fjac[k-1][0][3]
					-tmp1*njac[k-1][0][3] )
				lhs[k][AA][1][3] = ( -tmp2*fjac[k-1][1][3]
					-tmp1*njac[k-1][1][3] )
				lhs[k][AA][2][3] = ( -tmp2*fjac[k-1][2][3]
					-tmp1*njac[k-1][2][3] )
				lhs[k][AA][3][3] = ( -tmp2*fjac[k-1][3][3]
					-tmp1*njac[k-1][3][3]
					-tmp1*dz4 )
				lhs[k][AA][4][3] = ( -tmp2*fjac[k-1][4][3]
					-tmp1*njac[k-1][4][3] )
				lhs[k][AA][0][4] = ( -tmp2*fjac[k-1][0][4]
					-tmp1*njac[k-1][0][4] )
				lhs[k][AA][1][4] = ( -tmp2*fjac[k-1][1][4]
					-tmp1*njac[k-1][1][4] )
				lhs[k][AA][2][4] = ( -tmp2*fjac[k-1][2][4]
					-tmp1*njac[k-1][2][4] )
				lhs[k][AA][3][4] = ( -tmp2*fjac[k-1][3][4]
					-tmp1*njac[k-1][3][4] )
				lhs[k][AA][4][4] = ( -tmp2*fjac[k-1][4][4]
					-tmp1*njac[k-1][4][4]
					-tmp1*dz5 )
				lhs[k][BB][0][0] = ( 1.0
					+tmp1*2.0*njac[k][0][0]
					+tmp1*2.0*dz1 )
				lhs[k][BB][1][0] = tmp1*2.0*njac[k][1][0]
				lhs[k][BB][2][0] = tmp1*2.0*njac[k][2][0]
				lhs[k][BB][3][0] = tmp1*2.0*njac[k][3][0]
				lhs[k][BB][4][0] = tmp1*2.0*njac[k][4][0]
				lhs[k][BB][0][1] = tmp1*2.0*njac[k][0][1]
				lhs[k][BB][1][1] = ( 1.0
					+tmp1*2.0*njac[k][1][1]
					+tmp1*2.0*dz2 )
				lhs[k][BB][2][1] = tmp1*2.0*njac[k][2][1]
				lhs[k][BB][3][1] = tmp1*2.0*njac[k][3][1]
				lhs[k][BB][4][1] = tmp1*2.0*njac[k][4][1]
				lhs[k][BB][0][2] = tmp1*2.0*njac[k][0][2]
				lhs[k][BB][1][2] = tmp1*2.0*njac[k][1][2]
				lhs[k][BB][2][2] = ( 1.0
					+tmp1*2.0*njac[k][2][2]
					+tmp1*2.0*dz3 )
				lhs[k][BB][3][2] = tmp1*2.0*njac[k][3][2]
				lhs[k][BB][4][2] = tmp1*2.0*njac[k][4][2]
				lhs[k][BB][0][3] = tmp1*2.0*njac[k][0][3]
				lhs[k][BB][1][3] = tmp1*2.0*njac[k][1][3]
				lhs[k][BB][2][3] = tmp1*2.0*njac[k][2][3]
				lhs[k][BB][3][3] = ( 1.0
					+tmp1*2.0*njac[k][3][3]
					+tmp1*2.0*dz4 )
				lhs[k][BB][4][3] = tmp1*2.0*njac[k][4][3]
				lhs[k][BB][0][4] = tmp1*2.0*njac[k][0][4]
				lhs[k][BB][1][4] = tmp1*2.0*njac[k][1][4]
				lhs[k][BB][2][4] = tmp1*2.0*njac[k][2][4]
				lhs[k][BB][3][4] = tmp1*2.0*njac[k][3][4]
				lhs[k][BB][4][4] = ( 1.0
					+tmp1*2.0*njac[k][4][4] 
					+tmp1*2.0*dz5 )
				lhs[k][CC][0][0] = ( tmp2*fjac[k+1][0][0]
					-tmp1*njac[k+1][0][0]
					-tmp1*dz1 )
				lhs[k][CC][1][0] = ( tmp2*fjac[k+1][1][0]
					-tmp1*njac[k+1][1][0] )
				lhs[k][CC][2][0] = ( tmp2*fjac[k+1][2][0]
					-tmp1*njac[k+1][2][0] )
				lhs[k][CC][3][0] = ( tmp2*fjac[k+1][3][0]
					-tmp1*njac[k+1][3][0] )
				lhs[k][CC][4][0] = ( tmp2*fjac[k+1][4][0]
					-tmp1*njac[k+1][4][0] )
				lhs[k][CC][0][1] = ( tmp2*fjac[k+1][0][1]
					-tmp1*njac[k+1][0][1] )
				lhs[k][CC][1][1] = ( tmp2*fjac[k+1][1][1]
					-tmp1*njac[k+1][1][1]
					-tmp1*dz2 )
				lhs[k][CC][2][1] = ( tmp2*fjac[k+1][2][1]
					-tmp1*njac[k+1][2][1] )
				lhs[k][CC][3][1] = ( tmp2*fjac[k+1][3][1]
					-tmp1*njac[k+1][3][1] )
				lhs[k][CC][4][1] = ( tmp2*fjac[k+1][4][1]
					-tmp1*njac[k+1][4][1] )
				lhs[k][CC][0][2] = ( tmp2*fjac[k+1][0][2]
					-tmp1*njac[k+1][0][2] )
				lhs[k][CC][1][2] = (  tmp2*fjac[k+1][1][2]
					-tmp1*njac[k+1][1][2] )
				lhs[k][CC][2][2] = ( tmp2*fjac[k+1][2][2]
					-tmp1*njac[k+1][2][2]
					-tmp1*dz3 )
				lhs[k][CC][3][2] = ( tmp2*fjac[k+1][3][2]
					-tmp1*njac[k+1][3][2] )
				lhs[k][CC][4][2] = ( tmp2*fjac[k+1][4][2]
					-tmp1*njac[k+1][4][2] )
				lhs[k][CC][0][3] = ( tmp2*fjac[k+1][0][3]
					-tmp1*njac[k+1][0][3] )
				lhs[k][CC][1][3] = ( tmp2*fjac[k+1][1][3]
					-tmp1*njac[k+1][1][3] )
				lhs[k][CC][2][3] = ( tmp2*fjac[k+1][2][3]
					-tmp1*njac[k+1][2][3] )
				lhs[k][CC][3][3] = ( tmp2*fjac[k+1][3][3]
					-tmp1*njac[k+1][3][3]
					-tmp1*dz4 )
				lhs[k][CC][4][3] = ( tmp2*fjac[k+1][4][3]
					-tmp1*njac[k+1][4][3] )
				lhs[k][CC][0][4] = ( tmp2*fjac[k+1][0][4]
					-tmp1*njac[k+1][0][4] )
				lhs[k][CC][1][4] = ( tmp2*fjac[k+1][1][4]
					-tmp1*njac[k+1][1][4] )
				lhs[k][CC][2][4] = ( tmp2*fjac[k+1][2][4]
					-tmp1*njac[k+1][2][4] )
				lhs[k][CC][3][4] = ( tmp2*fjac[k+1][3][4]
					-tmp1*njac[k+1][3][4] )
				lhs[k][CC][4][4] = ( tmp2*fjac[k+1][4][4]
					-tmp1*njac[k+1][4][4]
					-tmp1*dz5 )
			#END for k in range(1, ksize):
			
			# ---------------------------------------------------------------------
			# performs guaussian elimination on this cell.
			#  
			# assumes that unpacking routines for non-first cells 
			# preload c' and rhs' from previous cell.
			#  
			# assumed send happens outside this routine, but that
			# c'(KMAX) and rhs'(KMAX) will be sent to next cell.
			# ---------------------------------------------------------------------
			# outer most do loops - sweeping in i direction
			# ---------------------------------------------------------------------
			# multiply c(i,j,0) by b_inverse and copy back to c
			# multiply rhs(0) by b_inverse(0) and copy to rhs
			# ---------------------------------------------------------------------
			binvcrhs(lhs[0][BB], lhs[0][CC], rhs[0][j][i])
			# ---------------------------------------------------------------------
			# begin inner most do loop
			# do all the elements of the cell unless last 
			# ---------------------------------------------------------------------
			for k in range(1, ksize):
				# -------------------------------------------------------------------
				# subtract A*lhs_vector(k-1) from lhs_vector(k)
				#  
				# rhs(k) = rhs(k) - A*rhs(k-1)
				# -------------------------------------------------------------------
				matvec_sub(lhs[k][AA], rhs[k-1][j][i], rhs[k][j][i])
				# -------------------------------------------------------------------
				# B(k) = B(k) - C(k-1)*A(k)
				# matmul_sub(aa,i,j,k,c,cc,i,j,k-1,c,bb,i,j,k)
				# --------------------------------------------------------------------
				matmul_sub(lhs[k][AA], lhs[k-1][CC], lhs[k][BB])
				# -------------------------------------------------------------------
				# multiply c(i,j,k) by b_inverse and copy back to c
				# multiply rhs(i,j,1) by b_inverse(i,j,1) and copy to rhs
				# -------------------------------------------------------------------
				binvcrhs(lhs[k][BB], lhs[k][CC], rhs[k][j][i])

			# ---------------------------------------------------------------------
			# now finish up special cases for last cell
			# ---------------------------------------------------------------------
			# rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
			# ---------------------------------------------------------------------
			matvec_sub(lhs[ksize][AA], rhs[ksize-1][j][i], rhs[ksize][j][i])
			# ---------------------------------------------------------------------
			# B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
			# matmul_sub(aa,i,j,ksize,c,
			# $ cc,i,j,ksize-1,c,bb,i,j,ksize)
			# ---------------------------------------------------------------------
			matmul_sub(lhs[ksize][AA], lhs[ksize-1][CC], lhs[ksize][BB])
			# ---------------------------------------------------------------------
			# multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
			# ---------------------------------------------------------------------
			binvrhs(lhs[ksize][BB], rhs[ksize][j][i])
			# ---------------------------------------------------------------------
			# back solve: if last cell, then generate U(ksize)=rhs(ksize)
			# else assume U(ksize) is loaded in un pack backsub_info
			# so just use it
			# after u(kstart) will be sent to next cell
			# ---------------------------------------------------------------------
			for k in range(ksize-1, 0-1, -1):
				for m in range(BLOCK_SIZE):
					for n in range(BLOCK_SIZE):
						rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[k][CC][n][m] * rhs[k+1][j][i][n]
		#END for i in range(1, grid_points[0]-1):
	#END for j in range(1, grid_points[1]-1):
	
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_ZSOLVE)
#END z_solve()


# ---------------------------------------------------------------------
# performs line solves in y direction by first factoring
# the block-tridiagonal matrix into an upper triangular matrix, 
# and then performing back substitution to solve for the unknow
# vectors of each line.  
#  
# make sure we treat elements zero to cell_size in the direction
# of the sweep.
# ---------------------------------------------------------------------
@njit
def y_solve(fjac, njac, lhs, rhs, u, qs, square, rho_i):
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_YSOLVE)
	# ---------------------------------------------------------------------
	# this function computes the left hand side for the three y-factors   
	# ---------------------------------------------------------------------
	jsize = grid_points[1] - 1
	# ---------------------------------------------------------------------
	# compute the indices for storing the tri-diagonal matrix;
	# determine a (labeled f) and n jacobians for cell c
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		for i in range(1, grid_points[0]-1):
			for j in range(jsize+1):
				tmp1 = rho_i[k][j][i]
				tmp2 = tmp1 * tmp1
				tmp3 = tmp1 * tmp2
				fjac[j][0][0] = 0.0
				fjac[j][1][0] = 0.0
				fjac[j][2][0] = 1.0
				fjac[j][3][0] = 0.0
				fjac[j][4][0] = 0.0
				fjac[j][0][1] = -(u[k][j][i][1]*u[k][j][i][2])*tmp2
				fjac[j][1][1] = u[k][j][i][2]*tmp1
				fjac[j][2][1] = u[k][j][i][1]*tmp1
				fjac[j][3][1] = 0.0
				fjac[j][4][1] = 0.0
				fjac[j][0][2] = -(u[k][j][i][2]*u[k][j][i][2]*tmp2)+c2*qs[k][j][i]
				fjac[j][1][2] = -c2*u[k][j][i][1]*tmp1
				fjac[j][2][2] = (2.0-c2)*u[k][j][i][2]*tmp1
				fjac[j][3][2] = -c2*u[k][j][i][3]*tmp1
				fjac[j][4][2] = c2
				fjac[j][0][3] = -(u[k][j][i][2]*u[k][j][i][3])*tmp2
				fjac[j][1][3] = 0.0
				fjac[j][2][3] = u[k][j][i][3]*tmp1
				fjac[j][3][3] = u[k][j][i][2]*tmp1
				fjac[j][4][3] = 0.0
				fjac[j][0][4] = (c2*2.0*square[k][j][i]-c1*u[k][j][i][4])*u[k][j][i][2]*tmp2
				fjac[j][1][4] = -c2*u[k][j][i][1]*u[k][j][i][2]*tmp2
				fjac[j][2][4] = c1*u[k][j][i][4]*tmp1-c2*(qs[k][j][i]+u[k][j][i][2]*u[k][j][i][2]*tmp2)
				fjac[j][3][4] = -c2*(u[k][j][i][2]*u[k][j][i][3])*tmp2
				fjac[j][4][4] = c1*u[k][j][i][2]*tmp1
				njac[j][0][0] = 0.0
				njac[j][1][0] = 0.0
				njac[j][2][0] = 0.0
				njac[j][3][0] = 0.0
				njac[j][4][0] = 0.0
				njac[j][0][1] = -c3c4*tmp2*u[k][j][i][1]
				njac[j][1][1] = c3c4*tmp1
				njac[j][2][1] = 0.0
				njac[j][3][1] = 0.0
				njac[j][4][1] = 0.0
				njac[j][0][2] = -con43*c3c4*tmp2*u[k][j][i][2]
				njac[j][1][2] = 0.0
				njac[j][2][2] = con43*c3c4*tmp1
				njac[j][3][2] = 0.0
				njac[j][4][2] = 0.0
				njac[j][0][3] = -c3c4*tmp2*u[k][j][i][3]
				njac[j][1][3] = 0.0
				njac[j][2][3] = 0.0
				njac[j][3][3] = c3c4*tmp1
				njac[j][4][3] = 0.0
				njac[j][0][4] = ( -(c3c4-c1345)*tmp3*(u[k][j][i][1]*u[k][j][i][1])
					-(con43*c3c4-c1345)*tmp3*(u[k][j][i][2]*u[k][j][i][2])
					-(c3c4-c1345)*tmp3*(u[k][j][i][3]*u[k][j][i][3])
					-c1345*tmp2*u[k][j][i][4] )
				njac[j][1][4] = (c3c4-c1345)*tmp2*u[k][j][i][1]
				njac[j][2][4] = (con43*c3c4-c1345)*tmp2*u[k][j][i][2]
				njac[j][3][4] = (c3c4-c1345)*tmp2*u[k][j][i][3]
				njac[j][4][4] = (c1345)*tmp1
			#END for j in range(jsize+1):
			# ---------------------------------------------------------------------
			# now joacobians set, so form left hand side in y direction
			# ---------------------------------------------------------------------
			lhsinit(lhs, jsize)
			for j in range(1, jsize):
				tmp1 = dt * ty1
				tmp2 = dt * ty2
				lhs[j][AA][0][0] = ( -tmp2*fjac[j-1][0][0]
					-tmp1*njac[j-1][0][0]
					-tmp1*dy1 ) 
				lhs[j][AA][1][0] = ( -tmp2*fjac[j-1][1][0]
					-tmp1*njac[j-1][1][0] )
				lhs[j][AA][2][0] = ( -tmp2*fjac[j-1][2][0]
					-tmp1*njac[j-1][2][0] )
				lhs[j][AA][3][0] = ( -tmp2*fjac[j-1][3][0]
					-tmp1*njac[j-1][3][0] )
				lhs[j][AA][4][0] = ( -tmp2*fjac[j-1][4][0]
					-tmp1*njac[j-1][4][0] )
				lhs[j][AA][0][1] = ( -tmp2*fjac[j-1][0][1]
					-tmp1*njac[j-1][0][1] )
				lhs[j][AA][1][1] = ( -tmp2*fjac[j-1][1][1]
					-tmp1*njac[j-1][1][1]
					-tmp1*dy2 )
				lhs[j][AA][2][1] = ( -tmp2*fjac[j-1][2][1]
					-tmp1*njac[j-1][2][1] )
				lhs[j][AA][3][1] = ( -tmp2*fjac[j-1][3][1]
					-tmp1*njac[j-1][3][1] )
				lhs[j][AA][4][1] = ( -tmp2*fjac[j-1][4][1]
					-tmp1*njac[j-1][4][1] )
				lhs[j][AA][0][2] = ( -tmp2*fjac[j-1][0][2]
					-tmp1*njac[j-1][0][2] )
				lhs[j][AA][1][2] = ( -tmp2*fjac[j-1][1][2]
					-tmp1*njac[j-1][1][2] )
				lhs[j][AA][2][2] = ( -tmp2*fjac[j-1][2][2]
					-tmp1*njac[j-1][2][2]
					-tmp1*dy3 )
				lhs[j][AA][3][2] = ( -tmp2*fjac[j-1][3][2]
					-tmp1*njac[j-1][3][2] )
				lhs[j][AA][4][2] = ( -tmp2*fjac[j-1][4][2]
					-tmp1*njac[j-1][4][2] )
				lhs[j][AA][0][3] = ( -tmp2*fjac[j-1][0][3]
					-tmp1*njac[j-1][0][3] )
				lhs[j][AA][1][3] = ( -tmp2*fjac[j-1][1][3]
					-tmp1*njac[j-1][1][3] )
				lhs[j][AA][2][3] = ( -tmp2*fjac[j-1][2][3]
					-tmp1*njac[j-1][2][3] )
				lhs[j][AA][3][3] = ( -tmp2*fjac[j-1][3][3]
					-tmp1*njac[j-1][3][3]
					-tmp1*dy4 )
				lhs[j][AA][4][3] = ( -tmp2*fjac[j-1][4][3]
					-tmp1*njac[j-1][4][3] )
				lhs[j][AA][0][4] = ( -tmp2*fjac[j-1][0][4]
					-tmp1*njac[j-1][0][4] )
				lhs[j][AA][1][4] = ( -tmp2*fjac[j-1][1][4]
					-tmp1*njac[j-1][1][4] )
				lhs[j][AA][2][4] = ( -tmp2*fjac[j-1][2][4]
					-tmp1*njac[j-1][2][4] )
				lhs[j][AA][3][4] = ( -tmp2*fjac[j-1][3][4]
					-tmp1*njac[j-1][3][4] )
				lhs[j][AA][4][4] = ( -tmp2*fjac[j-1][4][4]
					-tmp1*njac[j-1][4][4]
					-tmp1*dy5 )
				lhs[j][BB][0][0] = ( 1.0
					+tmp1*2.0*njac[j][0][0]
					+tmp1*2.0*dy1 )
				lhs[j][BB][1][0] = tmp1*2.0*njac[j][1][0]
				lhs[j][BB][2][0] = tmp1*2.0*njac[j][2][0]
				lhs[j][BB][3][0] = tmp1*2.0*njac[j][3][0]
				lhs[j][BB][4][0] = tmp1*2.0*njac[j][4][0]
				lhs[j][BB][0][1] = tmp1*2.0*njac[j][0][1]
				lhs[j][BB][1][1] = ( 1.0
					+tmp1*2.0*njac[j][1][1]
					+tmp1*2.0*dy2 )
				lhs[j][BB][2][1] = tmp1*2.0*njac[j][2][1]
				lhs[j][BB][3][1] = tmp1*2.0*njac[j][3][1]
				lhs[j][BB][4][1] = tmp1*2.0*njac[j][4][1]
				lhs[j][BB][0][2] = tmp1*2.0*njac[j][0][2]
				lhs[j][BB][1][2] = tmp1*2.0*njac[j][1][2]
				lhs[j][BB][2][2] = ( 1.0
					+tmp1*2.0*njac[j][2][2]
					+tmp1*2.0*dy3 )
				lhs[j][BB][3][2] = tmp1*2.0*njac[j][3][2]
				lhs[j][BB][4][2] = tmp1*2.0*njac[j][4][2]
				lhs[j][BB][0][3] = tmp1*2.0*njac[j][0][3]
				lhs[j][BB][1][3] = tmp1*2.0*njac[j][1][3]
				lhs[j][BB][2][3] = tmp1*2.0*njac[j][2][3]
				lhs[j][BB][3][3] = ( 1.0
					+tmp1*2.0*njac[j][3][3]
					+tmp1*2.0*dy4 )
				lhs[j][BB][4][3] = tmp1*2.0*njac[j][4][3]
				lhs[j][BB][0][4] = tmp1*2.0*njac[j][0][4]
				lhs[j][BB][1][4] = tmp1*2.0*njac[j][1][4]
				lhs[j][BB][2][4] = tmp1*2.0*njac[j][2][4]
				lhs[j][BB][3][4] = tmp1*2.0*njac[j][3][4]
				lhs[j][BB][4][4] = ( 1.0
					+tmp1*2.0*njac[j][4][4] 
					+tmp1*2.0*dy5 )
				lhs[j][CC][0][0] = ( tmp2*fjac[j+1][0][0]
					-tmp1*njac[j+1][0][0]
					-tmp1*dy1 )
				lhs[j][CC][1][0] = ( tmp2*fjac[j+1][1][0]
					-tmp1*njac[j+1][1][0] )
				lhs[j][CC][2][0] = ( tmp2*fjac[j+1][2][0]
					-tmp1*njac[j+1][2][0] )
				lhs[j][CC][3][0] = ( tmp2*fjac[j+1][3][0]
					-tmp1*njac[j+1][3][0] )
				lhs[j][CC][4][0] = ( tmp2*fjac[j+1][4][0]
					-tmp1*njac[j+1][4][0] )
				lhs[j][CC][0][1] = ( tmp2*fjac[j+1][0][1]
					-tmp1*njac[j+1][0][1] )
				lhs[j][CC][1][1] = ( tmp2*fjac[j+1][1][1]
					-tmp1*njac[j+1][1][1]
					-tmp1*dy2 )
				lhs[j][CC][2][1] = ( tmp2*fjac[j+1][2][1]
					-tmp1*njac[j+1][2][1] )
				lhs[j][CC][3][1] = ( tmp2*fjac[j+1][3][1]
					-tmp1*njac[j+1][3][1] )
				lhs[j][CC][4][1] = ( tmp2*fjac[j+1][4][1]
					-tmp1*njac[j+1][4][1] )
				lhs[j][CC][0][2] = ( tmp2*fjac[j+1][0][2]
					-tmp1*njac[j+1][0][2] )
				lhs[j][CC][1][2] = ( tmp2*fjac[j+1][1][2]
					-tmp1*njac[j+1][1][2] )
				lhs[j][CC][2][2] = ( tmp2*fjac[j+1][2][2]
					-tmp1*njac[j+1][2][2]
					-tmp1*dy3 )
				lhs[j][CC][3][2] = ( tmp2*fjac[j+1][3][2]
					-tmp1*njac[j+1][3][2] )
				lhs[j][CC][4][2] = ( tmp2*fjac[j+1][4][2]
					-tmp1*njac[j+1][4][2] )
				lhs[j][CC][0][3] = ( tmp2*fjac[j+1][0][3]
					-tmp1*njac[j+1][0][3] )
				lhs[j][CC][1][3] = ( tmp2*fjac[j+1][1][3]
					-tmp1*njac[j+1][1][3] )
				lhs[j][CC][2][3] = ( tmp2*fjac[j+1][2][3]
					-tmp1*njac[j+1][2][3] )
				lhs[j][CC][3][3] = ( tmp2*fjac[j+1][3][3]
					-tmp1*njac[j+1][3][3]
					-tmp1*dy4 )
				lhs[j][CC][4][3] = ( tmp2*fjac[j+1][4][3]
					-tmp1*njac[j+1][4][3] )
				lhs[j][CC][0][4] = ( tmp2*fjac[j+1][0][4]
					-tmp1*njac[j+1][0][4] )
				lhs[j][CC][1][4] = ( tmp2*fjac[j+1][1][4]
					-tmp1*njac[j+1][1][4] )
				lhs[j][CC][2][4] = ( tmp2*fjac[j+1][2][4]
					-tmp1*njac[j+1][2][4] )
				lhs[j][CC][3][4] = ( tmp2*fjac[j+1][3][4]
					-tmp1*njac[j+1][3][4] )
				lhs[j][CC][4][4] = ( tmp2*fjac[j+1][4][4]
					-tmp1*njac[j+1][4][4]
					-tmp1*dy5 )
			#END for j in range(1, jsize):
			
			# ---------------------------------------------------------------------
			# performs guaussian elimination on this cell.
			#
			# assumes that unpacking routines for non-first cells 
			# preload c' and rhs' from previous cell.
			# 
			# assumed send happens outside this routine, but that
			# c'(JMAX) and rhs'(JMAX) will be sent to next cell
			# ---------------------------------------------------------------------
			# multiply c(i,0,k) by b_inverse and copy back to c
			# multiply rhs(0) by b_inverse(0) and copy to rhs
			# ---------------------------------------------------------------------
			binvcrhs(lhs[0][BB], lhs[0][CC], rhs[k][0][i])
			# ---------------------------------------------------------------------
			# begin inner most do loop
			# do all the elements of the cell unless last 
			# ---------------------------------------------------------------------
			for j in range(1, jsize):
				# -------------------------------------------------------------------
				# subtract A*lhs_vector(j-1) from lhs_vector(j)
				#  
				# rhs(j) = rhs(j) - A*rhs(j-1)
				# -------------------------------------------------------------------
				matvec_sub(lhs[j][AA], rhs[k][j-1][i], rhs[k][j][i])
				# -------------------------------------------------------------------
				# B(j) = B(j) - C(j-1)*A(j)
				# -------------------------------------------------------------------
				matmul_sub(lhs[j][AA], lhs[j-1][CC], lhs[j][BB])
				# -------------------------------------------------------------------
				# multiply c(i,j,k) by b_inverse and copy back to c
				# multiply rhs(i,1,k) by b_inverse(i,1,k) and copy to rhs
				# -------------------------------------------------------------------
				binvcrhs(lhs[j][BB], lhs[j][CC], rhs[k][j][i])
			
			# ---------------------------------------------------------------------
			# rhs(jsize) = rhs(jsize) - A*rhs(jsize-1)
			# ---------------------------------------------------------------------
			matvec_sub(lhs[jsize][AA], rhs[k][jsize-1][i], rhs[k][jsize][i])
			# ---------------------------------------------------------------------
			# B(jsize) = B(jsize) - C(jsize-1)*A(jsize)
			# matmul_sub(aa,i,jsize,k,c,
			# $ cc,i,jsize-1,k,c,bb,i,jsize,k)
			# ---------------------------------------------------------------------
			matmul_sub(lhs[jsize][AA], lhs[jsize-1][CC], lhs[jsize][BB])
			# ---------------------------------------------------------------------
			# multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
			# ---------------------------------------------------------------------
			binvrhs(lhs[jsize][BB], rhs[k][jsize][i])
			# ---------------------------------------------------------------------
			# back solve: if last cell, then generate U(jsize)=rhs(jsize)
			# else assume U(jsize) is loaded in un pack backsub_info
			# so just use it
			# after u(jstart) will be sent to next cell
			# ---------------------------------------------------------------------
			for j in range(jsize-1, 0-1, -1):
				for m in range(BLOCK_SIZE):
					for n in range(BLOCK_SIZE):
						rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][CC][n][m] * rhs[k][j+1][i][n]
		#END for i in range(1, grid_points[0]-1):
	#END for k in range(1, grid_points[2]-1):
	
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_YSOLVE)
#END y_solve()


# ---------------------------------------------------------------------
# performs line solves in X direction by first factoring
# the block-tridiagonal matrix into an upper triangular matrix, 
# and then performing back substitution to solve for the unknow
# vectors of each line.  
# 
# make sure we treat elements zero to cell_size in the direction
# of the sweep. 
# ---------------------------------------------------------------------
@njit
def x_solve(fjac, njac, lhs, rhs, u, qs, square, rho_i):
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_XSOLVE)
	# ---------------------------------------------------------------------
	# this function computes the left hand side in the xi-direction
	# ---------------------------------------------------------------------
	isize = grid_points[0]-1
	# ---------------------------------------------------------------------
	# determine a (labeled f) and n jacobians
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(isize+1):
				tmp1 = rho_i[k][j][i]
				tmp2 = tmp1 * tmp1
				tmp3 = tmp1 * tmp2
				fjac[i][0][0] = 0.0
				fjac[i][1][0] = 1.0
				fjac[i][2][0] = 0.0
				fjac[i][3][0] = 0.0
				fjac[i][4][0] = 0.0
				fjac[i][0][1] = -(u[k][j][i][1]*tmp2*u[k][j][i][1])+c2*qs[k][j][i]
				fjac[i][1][1] = (2.0-c2)*(u[k][j][i][1]/u[k][j][i][0])
				fjac[i][2][1] = -c2*(u[k][j][i][2]*tmp1)
				fjac[i][3][1] = -c2*(u[k][j][i][3]*tmp1)
				fjac[i][4][1] = c2
				fjac[i][0][2] = -(u[k][j][i][1]*u[k][j][i][2])*tmp2
				fjac[i][1][2] = u[k][j][i][2]*tmp1
				fjac[i][2][2] = u[k][j][i][1]*tmp1
				fjac[i][3][2] = 0.0
				fjac[i][4][2] = 0.0
				fjac[i][0][3] = -(u[k][j][i][1]*u[k][j][i][3])*tmp2
				fjac[i][1][3] = u[k][j][i][3]*tmp1
				fjac[i][2][3] = 0.0
				fjac[i][3][3] = u[k][j][i][1]*tmp1
				fjac[i][4][3] = 0.0
				fjac[i][0][4] = (c2*2.0*square[k][j][i]-c1*u[k][j][i][4])*(u[k][j][i][1]*tmp2)
				fjac[i][1][4] = c1*u[k][j][i][4]*tmp1-c2*(u[k][j][i][1]*u[k][j][i][1]*tmp2+qs[k][j][i])
				fjac[i][2][4] = -c2*(u[k][j][i][2]*u[k][j][i][1])*tmp2
				fjac[i][3][4] = -c2*(u[k][j][i][3]*u[k][j][i][1])*tmp2
				fjac[i][4][4] = c1*(u[k][j][i][1]*tmp1)
				njac[i][0][0] = 0.0
				njac[i][1][0] = 0.0
				njac[i][2][0] = 0.0
				njac[i][3][0] = 0.0
				njac[i][4][0] = 0.0
				njac[i][0][1] = -con43*c3c4*tmp2*u[k][j][i][1]
				njac[i][1][1] = con43*c3c4*tmp1
				njac[i][2][1] = 0.0
				njac[i][3][1] = 0.0
				njac[i][4][1] = 0.0
				njac[i][0][2] = -c3c4*tmp2*u[k][j][i][2]
				njac[i][1][2] = 0.0
				njac[i][2][2] = c3c4*tmp1
				njac[i][3][2] = 0.0
				njac[i][4][2] = 0.0
				njac[i][0][3] = -c3c4*tmp2*u[k][j][i][3]
				njac[i][1][3] = 0.0
				njac[i][2][3] = 0.0
				njac[i][3][3] = c3c4*tmp1
				njac[i][4][3] = 0.0
				njac[i][0][4] = ( -(con43*c3c4-c1345)*tmp3*(u[k][j][i][1]*u[k][j][i][1])
					-(c3c4-c1345)*tmp3*(u[k][j][i][2]*u[k][j][i][2])
					-(c3c4-c1345)*tmp3*(u[k][j][i][3]*u[k][j][i][3])
					-c1345*tmp2*u[k][j][i][4] )
				njac[i][1][4] = (con43*c3c4-c1345)*tmp2*u[k][j][i][1]
				njac[i][2][4] = (c3c4-c1345)*tmp2*u[k][j][i][2]
				njac[i][3][4] = (c3c4-c1345)*tmp2*u[k][j][i][3]
				njac[i][4][4] = (c1345)*tmp1
			#END for i in range(isize+1):
			
			# ---------------------------------------------------------------------
			# now jacobians set, so form left hand side in x direction
			# ---------------------------------------------------------------------
			lhsinit(lhs, isize)
			for i in range(1, isize):
				tmp1 = dt * tx1
				tmp2 = dt * tx2
				lhs[i][AA][0][0] = ( -tmp2*fjac[i-1][0][0]
					-tmp1*njac[i-1][0][0]
					-tmp1*dx1 ) 
				lhs[i][AA][1][0] = ( -tmp2*fjac[i-1][1][0]
					-tmp1*njac[i-1][1][0] )
				lhs[i][AA][2][0] = ( -tmp2*fjac[i-1][2][0]
					-tmp1*njac[i-1][2][0] )
				lhs[i][AA][3][0] = ( -tmp2*fjac[i-1][3][0]
					-tmp1*njac[i-1][3][0] )
				lhs[i][AA][4][0] = ( -tmp2*fjac[i-1][4][0]
					-tmp1*njac[i-1][4][0] )
				lhs[i][AA][0][1] = ( -tmp2*fjac[i-1][0][1]
					-tmp1*njac[i-1][0][1] )
				lhs[i][AA][1][1] = ( -tmp2*fjac[i-1][1][1]
					-tmp1*njac[i-1][1][1]
					-tmp1*dx2 )
				lhs[i][AA][2][1] = ( -tmp2*fjac[i-1][2][1]
					-tmp1*njac[i-1][2][1] )
				lhs[i][AA][3][1] = ( -tmp2*fjac[i-1][3][1]
					-tmp1*njac[i-1][3][1] )
				lhs[i][AA][4][1] = ( -tmp2*fjac[i-1][4][1]
					-tmp1*njac[i-1][4][1] )
				lhs[i][AA][0][2] = ( -tmp2*fjac[i-1][0][2]
					-tmp1*njac[i-1][0][2] )
				lhs[i][AA][1][2] = ( -tmp2*fjac[i-1][1][2]
					-tmp1*njac[i-1][1][2] )
				lhs[i][AA][2][2] = ( -tmp2*fjac[i-1][2][2]
					-tmp1*njac[i-1][2][2]
					-tmp1*dx3 )
				lhs[i][AA][3][2] = ( -tmp2*fjac[i-1][3][2]
					-tmp1*njac[i-1][3][2] )
				lhs[i][AA][4][2] = ( -tmp2*fjac[i-1][4][2]
					-tmp1*njac[i-1][4][2] )
				lhs[i][AA][0][3] = ( -tmp2*fjac[i-1][0][3]
					-tmp1*njac[i-1][0][3] )
				lhs[i][AA][1][3] = ( -tmp2*fjac[i-1][1][3]
					-tmp1*njac[i-1][1][3] )
				lhs[i][AA][2][3] = ( -tmp2*fjac[i-1][2][3]
					-tmp1*njac[i-1][2][3] )
				lhs[i][AA][3][3] = ( -tmp2*fjac[i-1][3][3]
					-tmp1*njac[i-1][3][3]
					-tmp1*dx4 )
				lhs[i][AA][4][3] = ( -tmp2*fjac[i-1][4][3]
					-tmp1*njac[i-1][4][3] )
				lhs[i][AA][0][4] = ( -tmp2*fjac[i-1][0][4]
					-tmp1*njac[i-1][0][4] )
				lhs[i][AA][1][4] = ( -tmp2*fjac[i-1][1][4]
					-tmp1*njac[i-1][1][4] )
				lhs[i][AA][2][4] = ( -tmp2*fjac[i-1][2][4]
					-tmp1*njac[i-1][2][4] )
				lhs[i][AA][3][4] = ( -tmp2*fjac[i-1][3][4]
					-tmp1*njac[i-1][3][4] )
				lhs[i][AA][4][4] = ( -tmp2*fjac[i-1][4][4]
					-tmp1*njac[i-1][4][4]
					-tmp1*dx5 )
				lhs[i][BB][0][0] = ( 1.0
					+tmp1*2.0*njac[i][0][0]
					+tmp1*2.0*dx1 )
				lhs[i][BB][1][0] = tmp1*2.0*njac[i][1][0]
				lhs[i][BB][2][0] = tmp1*2.0*njac[i][2][0]
				lhs[i][BB][3][0] = tmp1*2.0*njac[i][3][0]
				lhs[i][BB][4][0] = tmp1*2.0*njac[i][4][0]
				lhs[i][BB][0][1] = tmp1*2.0*njac[i][0][1]
				lhs[i][BB][1][1] = ( 1.0
					+tmp1*2.0*njac[i][1][1]
					+tmp1*2.0*dx2 )
				lhs[i][BB][2][1] = tmp1*2.0*njac[i][2][1]
				lhs[i][BB][3][1] = tmp1*2.0*njac[i][3][1]
				lhs[i][BB][4][1] = tmp1*2.0*njac[i][4][1]
				lhs[i][BB][0][2] = tmp1*2.0*njac[i][0][2]
				lhs[i][BB][1][2] = tmp1*2.0*njac[i][1][2]
				lhs[i][BB][2][2] = ( 1.0
					+tmp1*2.0*njac[i][2][2]
					+tmp1*2.0*dx3 )
				lhs[i][BB][3][2] = tmp1*2.0*njac[i][3][2]
				lhs[i][BB][4][2] = tmp1*2.0*njac[i][4][2]
				lhs[i][BB][0][3] = tmp1*2.0*njac[i][0][3]
				lhs[i][BB][1][3] = tmp1*2.0*njac[i][1][3]
				lhs[i][BB][2][3] = tmp1*2.0*njac[i][2][3]
				lhs[i][BB][3][3] = ( 1.0
					+tmp1*2.0*njac[i][3][3]
					+tmp1*2.0*dx4 )
				lhs[i][BB][4][3] = tmp1*2.0*njac[i][4][3]
				lhs[i][BB][0][4] = tmp1*2.0*njac[i][0][4]
				lhs[i][BB][1][4] = tmp1*2.0*njac[i][1][4]
				lhs[i][BB][2][4] = tmp1*2.0*njac[i][2][4]
				lhs[i][BB][3][4] = tmp1*2.0*njac[i][3][4]
				lhs[i][BB][4][4] = ( 1.0
					+tmp1*2.0*njac[i][4][4]
					+tmp1*2.0*dx5 )
				lhs[i][CC][0][0] = ( tmp2*fjac[i+1][0][0]
					-tmp1*njac[i+1][0][0]
					-tmp1*dx1 )
				lhs[i][CC][1][0] = ( tmp2*fjac[i+1][1][0]
					-tmp1*njac[i+1][1][0] )
				lhs[i][CC][2][0] = ( tmp2*fjac[i+1][2][0]
					-tmp1*njac[i+1][2][0] )
				lhs[i][CC][3][0] = ( tmp2*fjac[i+1][3][0]
					-tmp1*njac[i+1][3][0] )
				lhs[i][CC][4][0] = ( tmp2*fjac[i+1][4][0]
					-tmp1*njac[i+1][4][0] )
				lhs[i][CC][0][1] = ( tmp2*fjac[i+1][0][1]
					-tmp1*njac[i+1][0][1] )
				lhs[i][CC][1][1] = ( tmp2*fjac[i+1][1][1]
					-tmp1*njac[i+1][1][1]
					-tmp1*dx2 )
				lhs[i][CC][2][1] = ( tmp2*fjac[i+1][2][1]
					-tmp1*njac[i+1][2][1] )
				lhs[i][CC][3][1] = ( tmp2*fjac[i+1][3][1]
					-tmp1*njac[i+1][3][1] )
				lhs[i][CC][4][1] = ( tmp2*fjac[i+1][4][1]
					-tmp1*njac[i+1][4][1] )
				lhs[i][CC][0][2] = ( tmp2*fjac[i+1][0][2]
					-tmp1*njac[i+1][0][2] )
				lhs[i][CC][1][2] = ( tmp2*fjac[i+1][1][2]
					-tmp1*njac[i+1][1][2] )
				lhs[i][CC][2][2] = ( tmp2*fjac[i+1][2][2]
					-tmp1*njac[i+1][2][2]
					-tmp1*dx3 )
				lhs[i][CC][3][2] = ( tmp2*fjac[i+1][3][2]
					-tmp1*njac[i+1][3][2] )
				lhs[i][CC][4][2] = ( tmp2*fjac[i+1][4][2]
					-tmp1*njac[i+1][4][2] )
				lhs[i][CC][0][3] = ( tmp2*fjac[i+1][0][3]
					-tmp1*njac[i+1][0][3] )
				lhs[i][CC][1][3] = ( tmp2*fjac[i+1][1][3]
					-tmp1*njac[i+1][1][3] )
				lhs[i][CC][2][3] = ( tmp2*fjac[i+1][2][3]
					-tmp1*njac[i+1][2][3] )
				lhs[i][CC][3][3] = ( tmp2*fjac[i+1][3][3]
					-tmp1*njac[i+1][3][3]
					-tmp1*dx4 )
				lhs[i][CC][4][3] = ( tmp2*fjac[i+1][4][3]
					-tmp1*njac[i+1][4][3] )
				lhs[i][CC][0][4] = ( tmp2*fjac[i+1][0][4]
					-tmp1*njac[i+1][0][4] )
				lhs[i][CC][1][4] = ( tmp2*fjac[i+1][1][4]
					-tmp1*njac[i+1][1][4] )
				lhs[i][CC][2][4] = ( tmp2*fjac[i+1][2][4]
					-tmp1*njac[i+1][2][4] )
				lhs[i][CC][3][4] = ( tmp2 * fjac[i+1][3][4]
					-tmp1*njac[i+1][3][4] )
				lhs[i][CC][4][4] = ( tmp2*fjac[i+1][4][4]
					-tmp1*njac[i+1][4][4]
					-tmp1*dx5 )
			#END for i in (1, isize):
			
			# ---------------------------------------------------------------------
			# performs guaussian elimination on this cell.
			# 
			# assumes that unpacking routines for non-first cells 
			# preload C' and rhs' from previous cell.
			# 
			# assumed send happens outside this routine, but that
			# c'(IMAX) and rhs'(IMAX) will be sent to next cell
			# ---------------------------------------------------------------------
			# outer most do loops - sweeping in i direction
			# ---------------------------------------------------------------------
			# multiply c(0,j,k) by b_inverse and copy back to c
			# multiply rhs(0) by b_inverse(0) and copy to rhs
			# ---------------------------------------------------------------------
			binvcrhs(lhs[0][BB], lhs[0][CC], rhs[k][j][0])
			# ---------------------------------------------------------------------
			# begin inner most do loop
			# do all the elements of the cell unless last 
			# ---------------------------------------------------------------------
			for i in range(1, isize):
				# -------------------------------------------------------------------
				# rhs(i) = rhs(i) - A*rhs(i-1)
				# -------------------------------------------------------------------
				matvec_sub(lhs[i][AA], rhs[k][j][i-1], rhs[k][j][i])
				# -------------------------------------------------------------------
				# B(i) = B(i) - C(i-1)*A(i)
				# -------------------------------------------------------------------
				matmul_sub(lhs[i][AA], lhs[i-1][CC], lhs[i][BB])
				# -------------------------------------------------------------------
				# multiply c(i,j,k) by b_inverse and copy back to c
				# multiply rhs(1,j,k) by b_inverse(1,j,k) and copy to rhs
				# -------------------------------------------------------------------
				binvcrhs(lhs[i][BB], lhs[i][CC], rhs[k][j][i])
			# ---------------------------------------------------------------------
			# rhs(isize) = rhs(isize) - A*rhs(isize-1)
			# ---------------------------------------------------------------------
			matvec_sub(lhs[isize][AA], rhs[k][j][isize-1], rhs[k][j][isize])
			# ---------------------------------------------------------------------
			# B(isize) = B(isize) - C(isize-1)*A(isize)
			# ---------------------------------------------------------------------
			matmul_sub(lhs[isize][AA], lhs[isize-1][CC], lhs[isize][BB])
			# ---------------------------------------------------------------------
			# multiply rhs() by b_inverse() and copy to rhs
			# ---------------------------------------------------------------------
			binvrhs(lhs[isize][BB], rhs[k][j][isize])
			# ---------------------------------------------------------------------
			# back solve: if last cell, then generate U(isize)=rhs(isize)
			# else assume U(isize) is loaded in un pack backsub_info
			# so just use it
			# after u(istart) will be sent to next cell
			# ---------------------------------------------------------------------
			for i in range(isize-1, 0-1, -1):
				for m in range(BLOCK_SIZE):
					for n in range(BLOCK_SIZE):
						rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[i][CC][n][m] * rhs[k][j][i+1][n]
		#END for j in range(1, grid_points[1]-1):
	#END for k in range(1, grid_points[2]-1):
	
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_XSOLVE)
#END x_solve()


@njit
def compute_rhs(rho_i, us, vs, ws, square, qs, rhs, u): #Receive arrays as parameters
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHS)
	# ---------------------------------------------------------------------
	# compute the reciprocal of density, and the kinetic energy, 
	# and the speed of sound.
	# ---------------------------------------------------------------------
	for k in range(grid_points[2]):
		for j in range(grid_points[1]):
			for i in range(grid_points[0]):
				rho_inv = 1.0 / u[k][j][i][0]
				rho_i[k][j][i] = rho_inv
				us[k][j][i] = u[k][j][i][1] * rho_inv
				vs[k][j][i] = u[k][j][i][2] * rho_inv
				ws[k][j][i] = u[k][j][i][3] * rho_inv
				square[k][j][i] = ( 0.5*(
						u[k][j][i][1]*u[k][j][i][1]+ 
						u[k][j][i][2]*u[k][j][i][2]+
						u[k][j][i][3]*u[k][j][i][3])*rho_inv )
				qs[k][j][i] = square[k][j][i] * rho_inv
	#END for k in range(grid_points[2]):
	
	# ---------------------------------------------------------------------
	# copy the exact forcing term to the right hand side; because 
	# this forcing term is known, we can store it on the whole grid
	# including the boundary                   
	# ---------------------------------------------------------------------
	for k in range(grid_points[2]):
		for j in range(grid_points[1]):
			for i in range(grid_points[0]):
				for m in range(5):
					rhs[k][j][i][m] = forcing[k][j][i][m]
					
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHSX)
	# ---------------------------------------------------------------------
	# compute xi-direction fluxes 
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				uijk = us[k][j][i]
				up1 = us[k][j][i+1]
				um1 = us[k][j][i-1]
				rhs[k][j][i][0] = ( rhs[k][j][i][0]+dx1tx1* 
					(u[k][j][i+1][0]-2.0*u[k][j][i][0]+ 
					 u[k][j][i-1][0])-
					tx2*(u[k][j][i+1][1]-u[k][j][i-1][1]) )
				rhs[k][j][i][1] = ( rhs[k][j][i][1]+dx2tx1* 
					(u[k][j][i+1][1]-2.0*u[k][j][i][1]+ 
					 u[k][j][i-1][1])+
					xxcon2*con43*(up1-2.0*uijk+um1)-
					tx2*(u[k][j][i+1][1]*up1- 
							u[k][j][i-1][1]*um1+
							(u[k][j][i+1][4]- square[k][j][i+1]-
							 u[k][j][i-1][4]+ square[k][j][i-1])*
							c2) )
				rhs[k][j][i][2] = ( rhs[k][j][i][2]+dx3tx1* 
					(u[k][j][i+1][2]-2.0*u[k][j][i][2]+
					 u[k][j][i-1][2])+
					xxcon2*(vs[k][j][i+1]-2.0*vs[k][j][i]+
							vs[k][j][i-1])-
					tx2*(u[k][j][i+1][2]*up1- 
							u[k][j][i-1][2]*um1) )
				rhs[k][j][i][3] = ( rhs[k][j][i][3]+dx4tx1* 
					(u[k][j][i+1][3]-2.0*u[k][j][i][3]+
					 u[k][j][i-1][3])+
					xxcon2*(ws[k][j][i+1]-2.0*ws[k][j][i]+
							ws[k][j][i-1])-
					tx2*(u[k][j][i+1][3]*up1- 
							u[k][j][i-1][3]*um1) )
				rhs[k][j][i][4] = ( rhs[k][j][i][4]+dx5tx1* 
					(u[k][j][i+1][4]-2.0*u[k][j][i][4]+
					 u[k][j][i-1][4])+
					xxcon3*(qs[k][j][i+1]-2.0*qs[k][j][i]+
							qs[k][j][i-1])+
					xxcon4*(up1*up1-2.0*uijk*uijk+ 
							um1*um1)+
					xxcon5*(u[k][j][i+1][4]*rho_i[k][j][i+1]- 
							2.0*u[k][j][i][4]*rho_i[k][j][i]+
							u[k][j][i-1][4]*rho_i[k][j][i-1])-
					tx2*((c1*u[k][j][i+1][4]- 
								c2*square[k][j][i+1])*up1-
							(c1*u[k][j][i-1][4]- 
							 c2*square[k][j][i-1])*um1) )
			#END for i in range(1, grid_points[0]-1):
		#END for j in range(1, grid_points[1]-1):
		
		# ---------------------------------------------------------------------
		# add fourth order xi-direction dissipation               
		# ---------------------------------------------------------------------
		for j in range(1, grid_points[1]-1):
			i = 1
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
					(5.0*u[k][j][i][m]-4.0*u[k][j][i+1][m]+
					 u[k][j][i+2][m]) )

			i = 2
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
					(-4.0*u[k][j][i-1][m]+6.0*u[k][j][i][m]-
					 4.0*u[k][j][i+1][m]+u[k][j][i+2][m]) )

		for j in range(1, grid_points[1]-1):
			for i in range(3, grid_points[0]-3):
				for m in range(5):
					rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp * 
						(u[k][j][i-2][m]-4.0*u[k][j][i-1][m]+ 
						 6.0*u[k][j][i][m]-4.0*u[k][j][i+1][m]+ 
						 u[k][j][i+2][m]) )

		for j in range(1, grid_points[1]-1):
			i = grid_points[0]-3
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp*
					(u[k][j][i-2][m]-4.0*u[k][j][i-1][m]+ 
					 6.0*u[k][j][i][m]-4.0*u[k][j][i+1][m]) )

			i = grid_points[0]-2
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp*
					(u[k][j][i-2][m]-4.*u[k][j][i-1][m]+
					 5.*u[k][j][i][m]) )
	#END for k in range(1, grid_points[2]-1):
	
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHSX)
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHSY)
	
	# ---------------------------------------------------------------------
	# compute eta-direction fluxes 
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				vijk = vs[k][j][i]
				vp1 = vs[k][j+1][i]
				vm1 = vs[k][j-1][i]
				rhs[k][j][i][0] = ( rhs[k][j][i][0]+dy1ty1* 
					(u[k][j+1][i][0]-2.0*u[k][j][i][0]+ 
					 u[k][j-1][i][0])-
					ty2*(u[k][j+1][i][2]-u[k][j-1][i][2]) )
				rhs[k][j][i][1] = ( rhs[k][j][i][1]+dy2ty1* 
					(u[k][j+1][i][1]-2.0*u[k][j][i][1]+ 
					 u[k][j-1][i][1])+
					yycon2*(us[k][j+1][i]-2.0*us[k][j][i]+ 
							us[k][j-1][i])-
					ty2*(u[k][j+1][i][1]*vp1- 
							u[k][j-1][i][1]*vm1) )
				rhs[k][j][i][2] = ( rhs[k][j][i][2]+dy3ty1* 
					(u[k][j+1][i][2]-2.0*u[k][j][i][2]+ 
					 u[k][j-1][i][2])+
					yycon2*con43*(vp1-2.0*vijk+vm1)-
					ty2*(u[k][j+1][i][2]*vp1- 
							u[k][j-1][i][2]*vm1+
							(u[k][j+1][i][4]-square[k][j+1][i]- 
							 u[k][j-1][i][4]+square[k][j-1][i])
							*c2) )
				rhs[k][j][i][3] = ( rhs[k][j][i][3]+dy4ty1* 
					(u[k][j+1][i][3]-2.0*u[k][j][i][3]+ 
					 u[k][j-1][i][3])+
					yycon2*(ws[k][j+1][i]-2.0*ws[k][j][i]+ 
							ws[k][j-1][i])-
					ty2*(u[k][j+1][i][3]*vp1- 
							u[k][j-1][i][3]*vm1) )
				rhs[k][j][i][4] = ( rhs[k][j][i][4]+dy5ty1* 
					(u[k][j+1][i][4]-2.0*u[k][j][i][4]+ 
					 u[k][j-1][i][4])+
					yycon3*(qs[k][j+1][i]-2.0*qs[k][j][i]+ 
							qs[k][j-1][i])+
					yycon4*(vp1*vp1-2.0*vijk*vijk+ 
							vm1*vm1)+
					yycon5*(u[k][j+1][i][4]*rho_i[k][j+1][i]- 
							2.0*u[k][j][i][4]*rho_i[k][j][i]+
							u[k][j-1][i][4]*rho_i[k][j-1][i])-
					ty2*((c1*u[k][j+1][i][4]- 
								c2*square[k][j+1][i])*vp1-
							(c1*u[k][j-1][i][4]- 
							 c2*square[k][j-1][i])*vm1) )
			#END for i in range(1, grid_points[0]-1):
		#END for j in range(1, grid_points[1]-1):
		
		# ---------------------------------------------------------------------
		# add fourth order eta-direction dissipation         
		# ---------------------------------------------------------------------
		j = 1
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
					(5.0*u[k][j][i][m]-4.0*u[k][j+1][i][m]+
					 u[k][j+2][i][m]) )

		j = 2
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
					(-4.0*u[k][j-1][i][m]+6.0*u[k][j][i][m]-
					 4.0*u[k][j+1][i][m]+u[k][j+2][i][m]) )

		for j in range(3, grid_points[1]-3):
			for i in range(1, grid_points[0]-1):
				for m in range(5):
					rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
						(u[k][j-2][i][m]-4.0*u[k][j-1][i][m]+ 
						 6.0*u[k][j][i][m]-4.0*u[k][j+1][i][m]+ 
						 u[k][j+2][i][m]) )

		j = grid_points[1]-3
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp*
					(u[k][j-2][i][m]-4.0*u[k][j-1][i][m]+ 
					 6.0*u[k][j][i][m]-4.0*u[k][j+1][i][m]) )

		j = grid_points[1]-2
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp*
					(u[k][j-2][i][m]-4.*u[k][j-1][i][m]+
					 5.*u[k][j][i][m]) )
	#END for k in range(1, grid_points[2]-1):
	
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHSY)
	#if timeron: Not supported by @njit
	#	c_timers.timer_start(T_RHSZ)
	
	# ---------------------------------------------------------------------
	# compute zeta-direction fluxes 
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				wijk = ws[k][j][i]
				wp1 = ws[k+1][j][i]
				wm1 = ws[k-1][j][i]
				rhs[k][j][i][0] = ( rhs[k][j][i][0]+dz1tz1* 
					(u[k+1][j][i][0]-2.0*u[k][j][i][0]+ 
					 u[k-1][j][i][0])-
					tz2*(u[k+1][j][i][3]-u[k-1][j][i][3]) )
				rhs[k][j][i][1] = ( rhs[k][j][i][1]+dz2tz1* 
					(u[k+1][j][i][1]-2.0*u[k][j][i][1]+ 
					 u[k-1][j][i][1])+
					zzcon2*(us[k+1][j][i]-2.0*us[k][j][i]+ 
							us[k-1][j][i])-
					tz2*(u[k+1][j][i][1]*wp1- 
							u[k-1][j][i][1]*wm1) )
				rhs[k][j][i][2] = ( rhs[k][j][i][2]+dz3tz1* 
					(u[k+1][j][i][2]-2.0*u[k][j][i][2]+ 
					 u[k-1][j][i][2])+
					zzcon2*(vs[k+1][j][i]-2.0*vs[k][j][i]+ 
							vs[k-1][j][i])-
					tz2*(u[k+1][j][i][2]*wp1- 
							u[k-1][j][i][2]*wm1) )
				rhs[k][j][i][3] = ( rhs[k][j][i][3]+dz4tz1* 
					(u[k+1][j][i][3]-2.0*u[k][j][i][3]+ 
					 u[k-1][j][i][3])+
					zzcon2*con43*(wp1-2.0*wijk+wm1)-
					tz2*(u[k+1][j][i][3]*wp1- 
							u[k-1][j][i][3]*wm1+
							(u[k+1][j][i][4]-square[k+1][j][i]- 
							 u[k-1][j][i][4]+square[k-1][j][i])
							*c2) )
				rhs[k][j][i][4] = ( rhs[k][j][i][4]+dz5tz1* 
					(u[k+1][j][i][4]-2.0*u[k][j][i][4]+ 
					 u[k-1][j][i][4])+
					zzcon3*(qs[k+1][j][i]-2.0*qs[k][j][i]+ 
							qs[k-1][j][i])+
					zzcon4*(wp1*wp1-2.0*wijk*wijk+ 
							wm1*wm1)+
					zzcon5*(u[k+1][j][i][4]*rho_i[k+1][j][i]- 
							2.0*u[k][j][i][4]*rho_i[k][j][i]+
							u[k-1][j][i][4]*rho_i[k-1][j][i])-
					tz2*((c1*u[k+1][j][i][4]- 
								c2*square[k+1][j][i])*wp1-
							(c1*u[k-1][j][i][4]- 
							 c2*square[k-1][j][i])*wm1) )
			#END for i in range(1, grid_points[0]-1):
		#END for j in range(1, grid_points[1]-1):
	#END for k in range(1, grid_points[2]-1):
	
	# ---------------------------------------------------------------------
	# add fourth order zeta-direction dissipation                
	# ---------------------------------------------------------------------
	k = 1
	for j in range(1, grid_points[1]-1):
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
					(5.0*u[k][j][i][m]-4.0*u[k+1][j][i][m]+
					 u[k+2][j][i][m]) )
					
	k = 2
	for j in range(1, grid_points[1]-1):
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
					(-4.0*u[k-1][j][i][m]+6.0*u[k][j][i][m]-
					 4.0*u[k+1][j][i][m]+u[k+2][j][i][m]) )

	for k in range(3, grid_points[2]-3): 
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				for m in range(5):
					rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp* 
						(u[k-2][j][i][m]-4.0*u[k-1][j][i][m]+ 
						 6.0*u[k][j][i][m]-4.0*u[k+1][j][i][m]+ 
						 u[k+2][j][i][m]) )
	
	k = grid_points[2]-3
	for j in range(1, grid_points[1]-1):
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp*
					(u[k-2][j][i][m]-4.0*u[k-1][j][i][m]+ 
					 6.0*u[k][j][i][m]-4.0*u[k+1][j][i][m]) )

	k = grid_points[2]-2 
	for j in range(1, grid_points[1]-1):
		for i in range(1, grid_points[0]-1):
			for m in range(5):
				rhs[k][j][i][m] = ( rhs[k][j][i][m]-dssp*
					(u[k-2][j][i][m]-4.*u[k-1][j][i][m]+
					 5.*u[k][j][i][m]) )
					
	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHSZ)
	
	for k in range(1, grid_points[2]-1):
		for j in range(1, grid_points[1]-1):
			for i in range(1, grid_points[0]-1):
				for m in range(5):
					rhs[k][j][i][m] = rhs[k][j][i][m]*dt

	#if timeron: Not supported by @njit
	#	c_timers.timer_stop(T_RHS)
#END compute_rhs()


def adi(rho_i, us, vs, ws, square, qs, rhs, fjac, njac, lhs, u):
	if timeron:
		c_timers.timer_start(T_RHS)
	compute_rhs(rho_i, us, vs, ws, square, qs, rhs, u)
	if timeron:
		c_timers.timer_stop(T_RHS)
	
	if timeron:
		c_timers.timer_start(T_XSOLVE)
	x_solve(fjac, njac, lhs, rhs, u, qs, square, rho_i)
	if timeron:
		c_timers.timer_stop(T_XSOLVE)
		
	if timeron:
		c_timers.timer_start(T_YSOLVE)
	y_solve(fjac, njac, lhs, rhs, u, qs, square, rho_i)
	if timeron:
		c_timers.timer_stop(T_YSOLVE)
		
	if timeron:
		c_timers.timer_start(T_ZSOLVE)
	z_solve(fjac, njac, lhs, rhs, u, qs, square)
	if timeron:
		c_timers.timer_stop(T_ZSOLVE)
	
	if timeron:
		c_timers.timer_start(T_ADD)
	add(u, rhs)
	if timeron:
		c_timers.timer_stop(T_ADD)
#END adi()


# ---------------------------------------------------------------------
# this function returns the exact solution at point xi, eta, zeta  
# ---------------------------------------------------------------------
@njit
def exact_solution(xi, eta, zeta, dtemp): #double dtemp[5]
	for m in range(5):
		dtemp[m] = ( ce[0][m]+
			xi*(ce[1][m]+
					xi*(ce[4][m]+
						xi*(ce[7][m]+
							xi*ce[10][m])))+
			eta*(ce[2][m]+
					eta*(ce[5][m]+
						eta*(ce[8][m]+
							eta*ce[11][m])))+
			zeta*(ce[3][m]+
					zeta*(ce[6][m]+
						zeta*(ce[9][m]+ 
							zeta*ce[12][m]))) )
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
		zeta = k * dnzm1
		for j in range(1, grid_points[1]-1):
			eta = j * dnym1
			for i in range(grid_points[0]):
				xi = i * dnxm1
				exact_solution(xi, eta, zeta, dtemp)
				for m in range(5):
					ue[m][i] = dtemp[m]

				dtpp = 1.0 / dtemp[0]
				for m in range(1, 5):
					buf[m][i] = dtpp * dtemp[m]

				cuf[i] = buf[1][i] * buf[1][i]
				buf[0][i] = cuf[i]+buf[2][i]*buf[2][i]+buf[3][i]*buf[3][i]
				q[i]= (0.5*(buf[1][i]*ue[1][i]+buf[2][i]*ue[2][i]+
						buf[3][i]*ue[3][i]) )

			for i in range(1, grid_points[0]-1):
				im1 = i-1
				ip1 = i+1
				forcing[k][j][i][0] = ( forcing[k][j][i][0]-
					tx2*(ue[1][ip1]-ue[1][im1])+
					dx1tx1*(ue[0][ip1]-2.0*ue[0][i]+ue[0][im1]))
				forcing[k][j][i][1] = ( forcing[k][j][i][1]-tx2*(
						(ue[1][ip1]*buf[1][ip1]+c2*(ue[4][ip1]-q[ip1]))-
						(ue[1][im1]*buf[1][im1]+c2*(ue[4][im1]-q[im1])))+
					xxcon1*(buf[1][ip1]-2.0*buf[1][i]+buf[1][im1])+
					dx2tx1*(ue[1][ip1]-2.0*ue[1][i]+ue[1][im1]) )
				forcing[k][j][i][2] = ( forcing[k][j][i][2]-tx2*(
						ue[2][ip1]*buf[1][ip1]-ue[2][im1]*buf[1][im1])+
					xxcon2*(buf[2][ip1]-2.0*buf[2][i]+buf[2][im1])+
					dx3tx1*(ue[2][ip1]-2.0*ue[2][i] +ue[2][im1]) )
				forcing[k][j][i][3] = ( forcing[k][j][i][3]-tx2*(
						ue[3][ip1]*buf[1][ip1]-ue[3][im1]*buf[1][im1])+
					xxcon2*(buf[3][ip1]-2.0*buf[3][i]+buf[3][im1])+
					dx4tx1*(ue[3][ip1]-2.0*ue[3][i]+ue[3][im1]) )
				forcing[k][j][i][4] = ( forcing[k][j][i][4]-tx2*(
						buf[1][ip1]*(c1*ue[4][ip1]-c2*q[ip1])-
						buf[1][im1]*(c1*ue[4][im1]-c2*q[im1]))+
					0.5*xxcon3*(buf[0][ip1]-2.0*buf[0][i]+
							buf[0][im1])+
					xxcon4*(cuf[ip1]-2.0*cuf[i]+cuf[im1])+
					xxcon5*(buf[4][ip1]-2.0*buf[4][i]+buf[4][im1])+
					dx5tx1*(ue[4][ip1]-2.0*ue[4][i]+ue[4][im1]) )
			#END for i in range(1, grid_points[0]-1):
			
			# ---------------------------------------------------------------------
			# fourth-order dissipation                         
			# ---------------------------------------------------------------------
			for m in range(5):
				i = 1
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(5.0*ue[m][i]-4.0*ue[m][i+1]+ue[m][i+2]))
				i = 2
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(-4.0*ue[m][i-1]+6.0*ue[m][i]-
					 4.0*ue[m][i+1]+ue[m][i+2]) )

			for i in range(3, grid_points[0]-3):
				for m in range(5):
					forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
						(ue[m][i-2]-4.0*ue[m][i-1]+
						 6.0*ue[m][i]-4.0*ue[m][i+1]+ue[m][i+2]) )

			for m in range(5):
				i = grid_points[0]-3
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(ue[m][i-2]-4.0*ue[m][i-1]+
					 6.0*ue[m][i]-4.0*ue[m][i+1]) )
				i = grid_points[0]-2
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(ue[m][i-2]-4.0*ue[m][i-1]+5.0*ue[m][i]) )
		#END for j in range(1, grid_points[1]-1):
	#END for k in range(1, grid_points[2]-1):
	
	# ---------------------------------------------------------------------
	# eta-direction flux differences             
	# ---------------------------------------------------------------------
	for k in range(1, grid_points[2]-1):
		zeta = k * dnzm1
		for i in range(1, grid_points[0]-1):
			xi = i * dnxm1
			for j in range(grid_points[1]):
				eta = j * dnym1
				exact_solution(xi, eta, zeta, dtemp)
				for m in range(5):
					ue[m][j] = dtemp[m]

				dtpp = 1.0 / dtemp[0]
				for m in range(1, 5):
					buf[m][j] = dtpp * dtemp[m]

				cuf[j] = buf[2][j] * buf[2][j]
				buf[0][j] = cuf[j]+buf[1][j]*buf[1][j]+buf[3][j]*buf[3][j]
				q[j] = ( 0.5*(buf[1][j]*ue[1][j]+buf[2][j]*ue[2][j]+
						buf[3][j]*ue[3][j]) )

			for j in range(1, grid_points[1]-1):
				jm1 = j-1
				jp1 = j+1
				forcing[k][j][i][0] = ( forcing[k][j][i][0]-
					ty2*(ue[2][jp1]-ue[2][jm1])+
					dy1ty1*(ue[0][jp1]-2.0*ue[0][j]+ue[0][jm1]) )
				forcing[k][j][i][1] = ( forcing[k][j][i][1]-ty2*(
						ue[1][jp1]*buf[2][jp1]-ue[1][jm1]*buf[2][jm1])+
					yycon2*(buf[1][jp1]-2.0*buf[1][j]+buf[1][jm1])+
					dy2ty1*(ue[1][jp1]-2.0*ue[1][j]+ue[1][jm1]) )
				forcing[k][j][i][2] = ( forcing[k][j][i][2]-ty2*(
						(ue[2][jp1]*buf[2][jp1]+c2*(ue[4][jp1]-q[jp1]))-
						(ue[2][jm1]*buf[2][jm1]+c2*(ue[4][jm1]-q[jm1])))+
					yycon1*(buf[2][jp1]-2.0*buf[2][j]+buf[2][jm1])+
					dy3ty1*(ue[2][jp1]-2.0*ue[2][j]+ue[2][jm1]) )
				forcing[k][j][i][3] = ( forcing[k][j][i][3]-ty2*(
						ue[3][jp1]*buf[2][jp1]-ue[3][jm1]*buf[2][jm1])+
					yycon2*(buf[3][jp1]-2.0*buf[3][j]+buf[3][jm1])+
					dy4ty1*(ue[3][jp1]-2.0*ue[3][j]+ue[3][jm1]) )
				forcing[k][j][i][4] = ( forcing[k][j][i][4]-ty2*(
						buf[2][jp1]*(c1*ue[4][jp1]-c2*q[jp1])-
						buf[2][jm1]*(c1*ue[4][jm1]-c2*q[jm1]))+
					0.5*yycon3*(buf[0][jp1]-2.0*buf[0][j]+
							buf[0][jm1])+
					yycon4*(cuf[jp1]-2.0*cuf[j]+cuf[jm1])+
					yycon5*(buf[4][jp1]-2.0*buf[4][j]+buf[4][jm1])+
					dy5ty1*(ue[4][jp1]-2.0*ue[4][j]+ue[4][jm1]) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation                      
			# ---------------------------------------------------------------------
			for m in range(5):
				j = 1
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(5.0*ue[m][j]-4.0*ue[m][j+1] +ue[m][j+2]) )
				j = 2
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(-4.0*ue[m][j-1]+6.0*ue[m][j]-
					 4.0*ue[m][j+1]+ue[m][j+2]) )

			for j in range(3, grid_points[1]-3):
				for m in range(5):
					forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
						(ue[m][j-2]-4.0*ue[m][j-1]+
						 6.0*ue[m][j]-4.0*ue[m][j+1]+ue[m][j+2]) )

			for m in range(5):
				j = grid_points[1]-3
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(ue[m][j-2]-4.0*ue[m][j-1]+
					 6.0*ue[m][j]-4.0*ue[m][j+1]) )
				j = grid_points[1]-2
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(ue[m][j-2]-4.0*ue[m][j-1]+5.0*ue[m][j]) )
		#END for i in range(1, grid_points[0]-1):
	#END for k in range(1, grid_points[2]-1):
	
	# ---------------------------------------------------------------------
	# zeta-direction flux differences                      
	# ---------------------------------------------------------------------
	for j in range(1, grid_points[1]-1): 
		eta = j * dnym1
		for i in range(1, grid_points[0]-1):
			xi = i * dnxm1
			for k in range(grid_points[2]):
				zeta = k * dnzm1
				exact_solution(xi, eta, zeta, dtemp)
				for m in range(5):
					ue[m][k] = dtemp[m]

				dtpp = 1.0 / dtemp[0]
				for m in range(1, 5):
					buf[m][k] = dtpp * dtemp[m]

				cuf[k] = buf[3][k] * buf[3][k]
				buf[0][k] = cuf[k]+buf[1][k]*buf[1][k]+buf[2][k]*buf[2][k]
				q[k] = ( 0.5*(buf[1][k]*ue[1][k]+buf[2][k]*ue[2][k]+
						buf[3][k]*ue[3][k]) )

			for k in range(1, grid_points[2]-1):
				km1 = k-1
				kp1 = k+1
				forcing[k][j][i][0] = ( forcing[k][j][i][0]-
					tz2*(ue[3][kp1]-ue[3][km1])+
					dz1tz1*(ue[0][kp1]-2.0*ue[0][k]+ue[0][km1]) )
				forcing[k][j][i][1] = ( forcing[k][j][i][1]-tz2*(
						ue[1][kp1]*buf[3][kp1]-ue[1][km1]*buf[3][km1])+
					zzcon2*(buf[1][kp1]-2.0*buf[1][k]+buf[1][km1])+
					dz2tz1*(ue[1][kp1]-2.0*ue[1][k]+ue[1][km1]) )
				forcing[k][j][i][2] = ( forcing[k][j][i][2]-tz2*(
						ue[2][kp1]*buf[3][kp1]-ue[2][km1]*buf[3][km1])+
					zzcon2*(buf[2][kp1]-2.0*buf[2][k]+buf[2][km1])+
					dz3tz1*(ue[2][kp1]-2.0*ue[2][k]+ue[2][km1]) )
				forcing[k][j][i][3] = ( forcing[k][j][i][3]-tz2*(
						(ue[3][kp1]*buf[3][kp1]+c2*(ue[4][kp1]-q[kp1]))-
						(ue[3][km1]*buf[3][km1]+c2*(ue[4][km1]-q[km1])))+
					zzcon1*(buf[3][kp1]-2.0*buf[3][k]+buf[3][km1])+
					dz4tz1*(ue[3][kp1]-2.0*ue[3][k]+ue[3][km1]) )
				forcing[k][j][i][4] = ( forcing[k][j][i][4]-tz2*(
						buf[3][kp1]*(c1*ue[4][kp1]-c2*q[kp1])-
						buf[3][km1]*(c1*ue[4][km1]-c2*q[km1]))+
					0.5*zzcon3*(buf[0][kp1]-2.0*buf[0][k]
							+buf[0][km1])+
					zzcon4*(cuf[kp1]-2.0*cuf[k]+cuf[km1])+
					zzcon5*(buf[4][kp1]-2.0*buf[4][k]+buf[4][km1])+
					dz5tz1*(ue[4][kp1]-2.0*ue[4][k]+ue[4][km1]) )

			# ---------------------------------------------------------------------
			# fourth-order dissipation                        
			# ---------------------------------------------------------------------
			for m in range(5):
				k = 1
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(5.0*ue[m][k]-4.0*ue[m][k+1]+ue[m][k+2]) )
				k = 2
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(-4.0*ue[m][k-1]+6.0*ue[m][k]-
					 4.0*ue[m][k+1]+ue[m][k+2]) )

			for k in range(3, grid_points[2]-3):
				for m in range(5):
					forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
						(ue[m][k-2]-4.0*ue[m][k-1]+
						 6.0*ue[m][k]-4.0*ue[m][k+1]+ue[m][k+2]) )

			for m in range(5):
				k = grid_points[2]-3
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(ue[m][k-2]-4.0*ue[m][k-1]+
					 6.0*ue[m][k]-4.0*ue[m][k+1]) )
				k = grid_points[2]-2
				forcing[k][j][i][m] = ( forcing[k][j][i][m]-dssp*
					(ue[m][k-2]-4.0*ue[m][k-1]+5.0*ue[m][k]) )
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
		zeta = k * dnzm1
		for j in range(grid_points[1]):
			eta = j * dnym1 
			for i in range(grid_points[0]):
				xi = i * dnxm1
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
		zeta = k * dnzm1
		for j in range(grid_points[1]):
			eta = j * dnym1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# east face                                                      
	# ---------------------------------------------------------------------
	i = grid_points[0]-1
	xi = 1.0
	for k in range(grid_points[2]):
		zeta = k * dnzm1
		for j in range(grid_points[1]):
			eta = j * dnym1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# south face                                                 
	# ---------------------------------------------------------------------
	j = 0
	eta = 0.0
	for k in range(grid_points[2]):
		zeta = k * dnzm1
		for i in range(grid_points[0]):
			xi = i * dnxm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# north face                                    
	# ---------------------------------------------------------------------
	j = grid_points[1]-1
	eta = 1.0
	for k in range(grid_points[2]):
		zeta = k * dnzm1
		for i in range(grid_points[0]):
			xi = i * dnxm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# bottom face                                       
	# ---------------------------------------------------------------------
	k = 0
	zeta = 0.0
	for j in range(grid_points[1]):
		eta = j * dnym1
		for i in range(grid_points[0]):
			xi = i * dnxm1
			exact_solution(xi, eta, zeta, temp)
			for m in range(5):
				u[k][j][i][m] = temp[m]

	# ---------------------------------------------------------------------
	# top face     
	# ---------------------------------------------------------------------
	k = grid_points[2]-1
	zeta = 1.0
	for j in range(grid_points[1]):
		eta = j * dnym1
		for i in range(grid_points[0]):
			xi = i * dnxm1
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
	
	t_names = numpy.empty(T_LAST+1, dtype=object)
	
	timeron = os.path.isfile("timer.flag")
	if timeron:
		t_names[T_TOTAL] = "total"
		t_names[T_RHSX] = "rhsx*"
		t_names[T_RHSY] = "rhsy*"
		t_names[T_RHSZ] = "rhsz*"
		t_names[T_RHS] = "rhs"
		t_names[T_XSOLVE] = "xsolve"
		t_names[T_YSOLVE] = "ysolve"
		t_names[T_ZSOLVE] = "zsolve"
		t_names[T_RDIS1] = "redist1"
		t_names[T_RDIS2] = "redist2"
		t_names[T_ADD] = "add"
		
	print("\n\n NAS Parallel Benchmarks 4.1 Serial Python version - BT Benchmark\n")
	print(" Size: %4dx%4dx%4d" % (grid_points[0], grid_points[1], grid_points[2]))
	print(" Iterations: %4d    dt: %10.6f" % (niter, dt))
	if (grid_points[0] > IMAX) or (grid_points[1] > JMAX) or (grid_points[2] > KMAX):
		print(" %d, %d, %d" % (grid_points[0], grid_points[1], grid_points[2]))
		print(" Problem size too big for compiled array sizes")
		sys.exit()

	set_constants()
	for i in range(1, T_LAST+1):
		c_timers.timer_clear(i)
	
	initialize(u)
	exact_rhs(forcing, ue, buf, cuf, q)
	
	# ---------------------------------------------------------------------
	# do one time step to touch all code, and reinitialize
	# ---------------------------------------------------------------------
	adi(rho_i, us, vs, ws, square, qs, rhs, fjac, njac, lhs, u)
	initialize(u)
	
	for i in range(1, T_LAST+1):
		c_timers.timer_clear(i)
	c_timers.timer_start(1)
	
	for step in range(1, niter+1):
		if (step % 20) == 0 or step == 1:
			print(" Time step %4d" % (step))
		adi(rho_i, us, vs, ws, square, qs, rhs, fjac, njac, lhs, u)
	
	c_timers.timer_stop(1)
	tmax = c_timers.timer_read(1)
	
	verified = verify()
	n3 = 1.0 * grid_points[0] * grid_points[1] * grid_points[2]
	navg = (grid_points[0] + grid_points[1] + grid_points[2]) / 3.0
	mflops = 0.0
	if tmax != 0.0:
		mflops = ( 1.0e-6 * niter *
			(3478.8*n3 - 17655.7*(navg*navg) + 28023.7*navg)
			/ tmax )
	
	c_print_results.c_print_results("BT",
			npbparams.CLASS,
			grid_points[0], 
			grid_points[1],
			grid_points[2],
			niter,
			tmax,
			mflops,
			"          floating point",
			verified)

	if timeron:
		trecs = numpy.empty(T_LAST+1, dtype=numpy.float64)
		for i in range(1, T_LAST+1):
			trecs[i] = c_timers.timer_read(i)
		if tmax == 0.0:
			tmax = 1.0
		print("  SECTION   Time (secs)")
		for i in range(1, T_LAST+1):
			print("  %-8s:%9.3f  (%6.2f%%)" % (t_names[i], trecs[i], trecs[i]*100/tmax))
			if i == T_RHS:
				t = trecs[T_RHSX] + trecs[T_RHSY] + trecs[T_RHSZ]
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("sub-rhs*", t, t*100/tmax))
				t = trecs[T_RHS] - t
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("rest-rhs", t, t*100/tmax))
			elif i == T_ZSOLVE:
				t = trecs[T_ZSOLVE] - trecs[T_RDIS1] - trecs[T_RDIS2]
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("sub-zsol", t, t*100/tmax))
			elif i == T_RDIS2:
				t = trecs[T_RDIS1] + trecs[T_RDIS2]
				print("    --> %8s:%9.3f  (%6.2f%%)" % ("redist", t, t*100/tmax))
		print("  (* Time hasn't gauged: operation is not supported by @njit)")
#END main()


#Starting of execution
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='NPB-PYTHON-SER BT')
	parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
	args = parser.parse_args()
	
	npbparams.set_bt_info(args.CLASS)
	set_global_variables()
	
	main()
