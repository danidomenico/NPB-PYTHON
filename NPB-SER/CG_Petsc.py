# ------------------------------------------------------------------------------
# NPB 4.1 Serial Python version - CG Benchmark (petsc4py adaptation)
# ------------------------------------------------------------------------------

import argparse
import sys
import os
import numpy
import math
from numba import njit

# Initialize petsc4py before importing PETSc
import petsc4py
petsc4py.init()
from petsc4py import PETSc

# Local imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common'))
import npbparams
from c_randdp import randlc
import c_timers
import c_print_results


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

# PETSc objects
A_petsc = None
x_petsc = None
z_petsc = None
p_petsc = None
q_petsc = None
r_petsc = None

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
    
    NZ = npbparams.NA * (npbparams.NONZER + 1) * (npbparams.NONZER + 1)
    NAZ = npbparams.NA * (npbparams.NONZER + 1)
    
    colidx = numpy.repeat(0, NZ)
    rowstr = numpy.repeat(0, npbparams.NA + 1)
    iv = numpy.repeat(0, npbparams.NA)
    arow = numpy.repeat(0, npbparams.NA)
    acol = numpy.repeat(0, NAZ)
    aelt = numpy.repeat(0.0, NAZ)
    a = numpy.repeat(0.0, NZ)
    x = numpy.repeat(1.0, npbparams.NA + 2)
    z = numpy.repeat(0.0, npbparams.NA + 2)
    p = numpy.repeat(0.0, npbparams.NA + 2)
    q = numpy.repeat(0.0, npbparams.NA + 2)
    r = numpy.repeat(0.0, npbparams.NA + 2)


def build_petsc_structures():
    """
    Converts the NPB CSR arrays (rowstr, colidx, a) into a PETSc AIJ (CSR) Matrix
    and allocates the corresponding PETSc state vectors.
    """
    global A_petsc, x_petsc, z_petsc, p_petsc, q_petsc, r_petsc
    global rowstr, colidx, a, naa

    nrows = lastrow - firstrow + 1
    nnz = rowstr[nrows]

    # Ensure array types strictly match compiled PETSc IntType (usually int32)
    # and ScalarType (usually float64), slicing to the exact compressed NNZ count.
    i_csr = numpy.asarray(rowstr[:nrows + 1], dtype=PETSc.IntType)
    j_csr = numpy.asarray(colidx[:nnz], dtype=PETSc.IntType)
    v_csr = numpy.asarray(a[:nnz], dtype=PETSc.ScalarType)

    # Create Sequential AIJ (CSR) Matrix directly backed by the CSR arrays
    A_petsc = PETSc.Mat().createAIJWithArrays(
        size=(nrows, nrows),
        csr=(i_csr, j_csr, v_csr),
        comm=PETSc.COMM_SELF
    )
    A_petsc.assemble()

    # Allocate sequential PETSc vectors of size `nrows`
    x_petsc = A_petsc.createVecRight()
    z_petsc = A_petsc.createVecRight()
    p_petsc = A_petsc.createVecRight()
    q_petsc = A_petsc.createVecRight()
    r_petsc = A_petsc.createVecRight()

    # Initialize starting vector x to 1.0
    x_petsc.set(1.0)


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


def conj_grad(A_mat, x_vec, z_vec, p_vec, q_vec, r_vec):
    """
    PETSc implementation of the NPB CG routine.
    Executes 25 fixed iterations using PETSc Mat/Vec kernels.
    """
    cgitmax = 25

    # Initialize the CG algorithm:
    # q = 0.0, z = 0.0, r = x, p = r
    q_vec.set(0.0)
    z_vec.set(0.0)
    x_vec.copy(result=r_vec)
    r_vec.copy(result=p_vec)

    # rho = r . r
    rho = r_vec.dot(r_vec)

    # The conjugate gradient iteration loop
    for cgit in range(1, cgitmax + 1):
        # q = A . p (Sparse Matrix-Vector Multiply)
        A_mat.mult(p_vec, q_vec)

        # d = p . q
        d = p_vec.dot(q_vec)

        # alpha = rho / (p . q)
        alpha = rho / d

        # Save a temporary of rho
        rho0 = rho

        # z = z + alpha * p
        # r = r - alpha * q
        z_vec.axpy(alpha, p_vec)
        r_vec.axpy(-alpha, q_vec)

        # rho = r . r
        rho = r_vec.dot(r_vec)

        # beta = rho / rho0
        beta = rho / rho0

        # p = r + beta * p
        p_vec.aypx(beta, r_vec)

    # Compute residual norm explicitly: ||r|| = ||x - A.z||
    # First, form r = A . z
    A_mat.mult(z_vec, r_vec)

    # r = x - r  (using aypx: r = -1.0 * r + x)
    r_vec.aypx(-1.0, x_vec)

    # rnorm = ||r||_2
    rnorm = r_vec.norm(PETSc.NormType.NORM_2)
    return rnorm


@njit
def sparse(a,
        colidx,
        rowstr,
        n,
        nz,
        nozer,
        arow,
        acol,
        aelt,
        firstrow,
        lastrow,
        nzloc,
        rcond,
        shift):
    
    NONZER_aux = npbparams.NONZER + 1
    nrows = lastrow - firstrow + 1

    for j in range(nrows + 1):
        rowstr[j] = 0
    
    for i in range(n):
        for nza in range(arow[i]):
            j = acol[i * NONZER_aux + nza] + 1
            rowstr[j] = rowstr[j] + arow[i]
    
    rowstr[0] = 0
    for j in range(1, nrows + 1):
        rowstr[j] = rowstr[j] + rowstr[j - 1]
    nza = rowstr[nrows] - 1

    if nza > nz:
        print("Space for matrix elements exceeded in sparse")
        print("nza, nzmax = ", nza, ", ", nz)

    for j in range(nrows):
        for k in range(rowstr[j], rowstr[j + 1]):
            a[k] = 0.0
            colidx[k] = -1
        nzloc[j] = 0

    size = 1.0
    ratio = pow(rcond, (1.0 / n))
    for i in range(n):
        for nza in range(arow[i]):
            j = acol[i * NONZER_aux + nza]

            scale = size * aelt[i * NONZER_aux + nza]
            for nzrow in range(arow[i]):
                jcol = acol[i * NONZER_aux + nzrow]
                va = aelt[i * NONZER_aux + nzrow] * scale

                if jcol == j and j == i:
                    va = va + rcond - shift

                goto_40 = False
                for k in range(rowstr[j], rowstr[j + 1]):
                    if colidx[k] > jcol:
                        start = rowstr[j + 1] - 2
                        for kk in range(start, k - 1, -1):
                            if colidx[kk] > -1:
                                a[kk + 1] = a[kk]
                                colidx[kk + 1] = colidx[kk]

                        colidx[k] = jcol
                        a[k] = 0.0
                        goto_40 = True
                        break
                    elif colidx[k] == -1:
                        colidx[k] = jcol
                        goto_40 = True
                        break
                    elif colidx[k] == jcol:
                        nzloc[j] = nzloc[j] + 1
                        goto_40 = True
                        break
                if not goto_40:
                    print("internal error in sparse: i=", i)
                a[k] = a[k] + va
        size = size * ratio

    for j in range(1, nrows):
        nzloc[j] = nzloc[j] + nzloc[j - 1]

    for j in range(nrows):
        if j > 0:
            j1 = rowstr[j] - nzloc[j - 1]
        else:
            j1 = 0
        j2 = rowstr[j + 1] - nzloc[j]
        nza = rowstr[j]
        for k in range(j1, j2):
            a[k] = a[nza]
            colidx[k] = colidx[nza]
            nza = nza + 1

    for j in range(1, nrows + 1):
        rowstr[j] = rowstr[j] - nzloc[j - 1]

    nza = rowstr[nrows] - 1


@njit
def icnvrt(x, ipwr2):
    return int(ipwr2 * x)


@njit
def vecset(n, v, iv, nzv, i, val):
    sett = False
    for k in range(nzv):
        if iv[k] == i:
            v[k] = val
            sett = True

    if not sett:
        v[nzv] = val
        iv[nzv] = i
        nzv = nzv + 1
    
    return nzv


@njit
def sprnvc(n, nz, nn1, v, iv, tran_aux):
    nzv = 0
    vecelt = 0.0
    vecloc = 0.0

    while nzv < nz:
        vecelt, tran_aux = randlc(tran_aux, amult)
        vecloc, tran_aux = randlc(tran_aux, amult)
        i = icnvrt(vecloc, nn1) + 1
        if i > n:
            continue

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
    return tran_aux


@njit
def makea(n,
        nz,
        a,
        colidx,
        rowstr,
        firstrow,
        lastrow,
        arow,
        acol,
        aelt,
        iv,
        tran_aux):
    
    NONZER_aux = npbparams.NONZER + 1
    ivc = numpy.empty(NONZER_aux, dtype=numpy.int32)
    vc = numpy.empty(NONZER_aux, dtype=numpy.float64)
    
    nn1 = 1
    while True:
        nn1 = 2 * nn1
        if nn1 >= n:
            break
    
    for iouter in range(n):
        nzv = npbparams.NONZER
        tran_aux = sprnvc(n, nzv, nn1, vc, ivc, tran_aux)
        nzv = vecset(n, vc, ivc, nzv, iouter + 1, 0.5)
        arow[iouter] = nzv
        for ivelt in range(nzv):
            acol[iouter * NONZER_aux + ivelt] = ivc[ivelt] - 1
            aelt[iouter * NONZER_aux + ivelt] = vc[ivelt]
            
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


def main():
    global naa, nzz, firstrow, lastrow, firstcol, lastcol
    global tran, amult
    global colidx, rowstr, iv, arow, acol, aelt
    global a, x, z, p, q, r
    global A_petsc, x_petsc, z_petsc, p_petsc, q_petsc, r_petsc
    
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
    lastrow  = npbparams.NA - 1
    firstcol = 0
    lastcol  = npbparams.NA - 1
    
    zeta_verify_value = create_zeta_verify_value()
        
    print("\n\n NAS Parallel Benchmarks 4.1 Serial Python (petsc4py) - CG Benchmark\n")
    print(" Size: %11d" % (npbparams.NA))
    print(" Iterations: %5d" % (npbparams.NITER))
    
    naa = npbparams.NA
    nzz = NZ
    
    # Initialize random number generator
    tran  = 314159265.0
    amult = 1220703125.0
    zeta, tran = randlc(tran, amult)
    
    tran = makea(naa, 
                nzz, 
                a, 
                colidx, 
                rowstr, 
                firstrow, 
                lastrow,
                arow, 
                acol,
                aelt,
                iv,
                tran)
    
    # Shift colidx from actual (firstcol --> lastcol) to local (0 --> lastcol-firstcol)
    end = lastrow - firstrow + 1 
    for j in range(end):
        for k in range(rowstr[j], rowstr[j + 1]):
            colidx[k] = colidx[k] - firstcol

    # Convert CSR arrays into PETSc AIJ Matrix and allocate PETSc Vectors
    build_petsc_structures()

    zeta = 0.0
    
    # -------------------------------------------------------------------
    # Do one iteration untimed to init all code and data page tables
    # -------------------------------------------------------------------
    for it in range(1, 1 + 1):
        rnorm = conj_grad(A_petsc, x_petsc, z_petsc, p_petsc, q_petsc, r_petsc)

        norm_temp1 = x_petsc.dot(z_petsc)
        norm_temp2 = z_petsc.dot(z_petsc)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)

        # Normalize z to obtain x: x = norm_temp2 * z
        z_petsc.copy(result=x_petsc)
        x_petsc.scale(norm_temp2)

    # Reset starting vector x to (1, 1, .... 1)
    x_petsc.set(1.0)
    zeta = 0.0

    c_timers.timer_stop(T_INIT)

    print(" Initialization time = %15.3f seconds" % (c_timers.timer_read(T_INIT)))

    c_timers.timer_start(T_BENCH)
    
    # --------------------------------------------------------------------
    # Main iteration for inverse power method
    # --------------------------------------------------------------------
    for it in range(1, npbparams.NITER + 1):
        if timeron:
            c_timers.timer_start(T_CONJ_GRAD)
        rnorm = conj_grad(A_petsc, x_petsc, z_petsc, p_petsc, q_petsc, r_petsc)
        if timeron:
            c_timers.timer_stop(T_CONJ_GRAD)

        # zeta = shift + 1/(x.z)
        norm_temp1 = x_petsc.dot(z_petsc)
        norm_temp2 = z_petsc.dot(z_petsc)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)
        zeta = npbparams.SHIFT + 1.0 / norm_temp1
        if it == 1:
            print("\n   iteration           ||r||                 zeta")
        print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

        # Normalize z to obtain x: x = norm_temp2 * z
        z_petsc.copy(result=x_petsc)
        x_petsc.scale(norm_temp2)

    c_timers.timer_stop(T_BENCH)

    # --------------------------------------------------------------------
    # End of timed section
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
                * ( 3.0 + (npbparams.NONZER * (npbparams.NONZER + 1))
                  + 25.0 * (5.0 + (npbparams.NONZER * (npbparams.NONZER + 1))) + 3.0 )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPB-PYTHON-SER CG (petsc4py)')
    parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
    args, unknown = parser.parse_known_args()
    
    npbparams.set_cg_info(args.CLASS)
    set_global_variables()
    
    main()