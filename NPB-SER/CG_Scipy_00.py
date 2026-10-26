# ------------------------------------------------------------------------------
# 
# VERSÃO MODIFICADA PARA SCIPY / NUMPY
# O algoritmo de Gradiente Conjugado manual foi substituído pelo solver 
# scipy.sparse.linalg.cg e as reduções de vetores foram vetorizadas com NumPy.
#
# ------------------------------------------------------------------------------

import argparse
import sys
import os
import numpy
import scipy
import math
import scipy.sparse as sp
from scipy.sparse.linalg import cg as scipy_cg
from numba import njit
  
# Local imports
sys.path.append(sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common')))
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
# Os vetores p e q não são mais necessários, pois o SciPy gerencia os vetores internos do CG
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
    global a, x, z, r
    
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
# Nova versão do conj_grad baseada inteiramente no SciPy
# ---------------------------------------------------------------------
def conj_grad_scipy(A_sp, x, z, r, naa): 
    # O NPB exige exatamente 25 iterações (cgitmax = 25).
    # Usamos tol=1e-15 para garantir que o solver não pare antes de 25 iterações.
    # O initial guess (x0) no NPB é o vetor zerado.
    
    z_res, info = scipy_cg(A_sp, x[:naa], x0=numpy.zeros(naa), rtol=1e-15, maxiter=25)
    
    # Copia o resultado de volta para o vetor global
    z[:naa] = z_res

    # Calcula a norma do resíduo: ||r|| = ||x - A.z|| usando NumPy e dot product da matriz
    r[:naa] = x[:naa] - A_sp.dot(z[:naa])
    rnorm = numpy.linalg.norm(r[:naa])
    
    return rnorm
#END conj_grad_scipy()

# ---------------------------------------------------------------------
# Funções de Geração de Matriz Nativas (Numba njit mantido para inicialização rápida)
# ---------------------------------------------------------------------
@njit
def sparse(a, colidx, rowstr, n, nz, nozer, arow, acol, aelt, firstrow, lastrow, nzloc, rcond, shift):
    NONZER_aux = npbparams.NONZER + 1
    nrows = lastrow - firstrow + 1
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
    if nza > nz:
        print("Space for matrix elements exceeded in sparse")
    for j in range(nrows):
        for k in range(rowstr[j], rowstr[j+1]):
            a[k] = 0.0
            colidx[k] = -1
        nzloc[j] = 0
    size = 1.0
    ratio = pow(rcond, (1.0 / n))
    for i in range(n):
        for nza in range(arow[i]):
            j = acol[i*NONZER_aux+nza]
            scale = size * aelt[i*NONZER_aux+nza]
            for nzrow in range(arow[i]):
                jcol = acol[i*NONZER_aux+nzrow]
                va = aelt[i*NONZER_aux+nzrow] * scale
                if jcol == j and j == i:
                    va = va + rcond - shift
                goto_40 = False
                for k in range(rowstr[j], rowstr[j+1]):
                    if colidx[k] > jcol:
                        start = rowstr[j+1]-2
                        for kk in range(start, k-1, -1): 
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
                        nzloc[j] = nzloc[j] + 1
                        goto_40 = True
                        break
                if (not goto_40):
                    print("internal error in sparse: i=", i)
                a[k] = a[k] + va
        size = size * ratio
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

@njit
def icnvrt(x, ipwr2):
    return int(ipwr2 * x)

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

@njit
def sprnvc(n, nz, nn1, v, iv, tran_aux):
    nzv = 0
    while nzv < nz:
        vecelt, tran_aux = randlc(tran_aux, amult)
        vecloc, tran_aux = randlc(tran_aux, amult)
        i = icnvrt(vecloc, nn1) + 1
        if i>n: 
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
def makea(n, nz, a, colidx, rowstr, firstrow, lastrow, arow, acol, aelt, iv, tran_aux):
    NONZER_aux = npbparams.NONZER+1
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
        nzv = vecset(n, vc, ivc, nzv, iouter+1, 0.5)
        arow[iouter] = nzv
        for ivelt in range(nzv):
            acol[iouter*NONZER_aux+ivelt] = ivc[ivelt] - 1
            aelt[iouter*NONZER_aux+ivelt] = vc[ivelt] 
            
    sparse(a, colidx, rowstr, n, nz, npbparams.NONZER, arow, acol, aelt, firstrow, lastrow, iv, npbparams.RCOND, npbparams.SHIFT)
    return tran_aux

def main():
    global naa, nzz, firstrow, lastrow, firstcol, lastcol
    global tran, amult
    global colidx, rowstr, iv, arow, acol, aelt
    global a, x, z, r
    
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
        
    print("\n\n NAS Parallel Benchmarks 4.1 Serial Python/SciPy version - CG Benchmark\n")
    print(" Size: %11d" % (npbparams.NA))
    print(" Iterations: %5d" % (npbparams.NITER))
    
    naa = npbparams.NA
    nzz = NZ
    
    tran  = 314159265.0
    amult = 1220703125.0
    zeta, tran = randlc(tran, amult)
    
    # Gera a matriz bruta no formato interno
    tran = makea(naa, nzz, a, colidx, rowstr, firstrow, lastrow, arow, acol, aelt, iv, tran)
    
    # Corrige os índices (mantido do original)
    end = lastrow - firstrow + 1 
    for j in range(end):
        for k in range(rowstr[j], rowstr[j+1]):
            colidx[k] = colidx[k] - firstcol

    # -------------------------------------------------------------------
    # CONVERSÃO SCIPY: Encapsulamos os arrays nativos no formato csr_matrix
    # -------------------------------------------------------------------
    nnz_total = rowstr[naa]
    A_sp = sp.csr_matrix((a[:nnz_total], colidx[:nnz_total], rowstr[:naa+1]), shape=(naa, naa))

    zeta = 0.0
    
    # Iteração única não temporizada para instanciar as funções (warmup)
    for it in range(1, 1+1):
        rnorm = conj_grad_scipy(A_sp, x, z, r, naa)

        # Vetorização NumPy: Substitui o laço lento de produto escalar
        norm_temp1 = numpy.dot(x[:naa], z[:naa])
        norm_temp2 = numpy.dot(z[:naa], z[:naa])
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)

        # Vetorização NumPy: Atualiza o vetor x diretamente
        x[:naa] = norm_temp2 * z[:naa]

    # Reseta o vetor para o teste cronometrado
    for i in range(npbparams.NA+1):
        x[i] = 1.0
    zeta = 0.0

    c_timers.timer_stop(T_INIT)
    print(" Initialization time = %15.3f seconds" % (c_timers.timer_read(T_INIT)))

    c_timers.timer_start(T_BENCH)
    
    # --------------------------------------------------------------------
    # Laço Principal Cronometrado
    # --------------------------------------------------------------------
    for it in range(1, npbparams.NITER+1):
        if timeron:
            c_timers.timer_start(T_CONJ_GRAD)
            
        # Chamada ao método do SciPy
        rnorm = conj_grad_scipy(A_sp, x, z, r, naa)
        
        if timeron:
            c_timers.timer_stop(T_CONJ_GRAD)

        # Vetorização NumPy: Cálculo de zet, norm_temp1 e norm_temp2
        norm_temp1 = numpy.dot(x[:naa], z[:naa])
        norm_temp2 = numpy.dot(z[:naa], z[:naa])
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)
        
        zeta = npbparams.SHIFT + 1.0 / norm_temp1
        
        if it == 1:
            print("\n   iteration           ||r||                 zeta")
        print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

        # Vetorização NumPy: Normalize z para obter x
        x[:naa] = norm_temp2 * z[:naa]

    c_timers.timer_stop(T_BENCH)

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

#Starting of execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NPB-PYTHON-SER CG (SciPy Version)')
    parser.add_argument("-c", "--CLASS", required=True, help="WORKLOADs CLASSes")
    args = parser.parse_args()
    
    npbparams.set_cg_info(args.CLASS)
    set_global_variables()
    
    main()