import math

import numpy
import petsc4py

petsc4py.init()
from petsc4py import PETSc


def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
    nrows = naa
    nnz = rowstr[nrows]

    i_csr = numpy.asarray(rowstr[:nrows + 1], dtype=PETSc.IntType)
    j_csr = numpy.asarray(colidx[:nnz], dtype=PETSc.IntType)
    v_csr = numpy.asarray(a[:nnz], dtype=PETSc.ScalarType)

    a_petsc = PETSc.Mat().createAIJWithArrays(
        size=(nrows, nrows),
        csr=(i_csr, j_csr, v_csr),
        comm=PETSc.COMM_SELF
    )
    a_petsc.assemble()

    x_petsc = a_petsc.createVecRight()
    z_petsc = a_petsc.createVecRight()
    p_petsc = a_petsc.createVecRight()
    q_petsc = a_petsc.createVecRight()
    r_petsc = a_petsc.createVecRight()
    
    x_petsc.set(1.0)

    return a_petsc, x_petsc, z_petsc, p_petsc, q_petsc, r_petsc


def cg_warmup(a, x, z, p , q, r):
    for it in range(1, 1 + 1):
        rnorm = conj_grad(a, x, z, p, q, r)

        norm_temp1 = x.dot(z)                # vecvecmul(x, z)
        norm_temp2 = z.dot(z)                # vecvecmul(z, z)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)

        # Atualização in-place no PETSc: z é copiado para x, e x é escalado.
        z.copy(result=x)
        x.scale(norm_temp2)


def cg(a, x, z, p , q, r, npbparams):
    x.set(1.0) # init_x()

    zeta = 0.0
    for it in range(1, npbparams.NITER + 1):
        rnorm = conj_grad(a, x, z, p, q, r)

        norm_temp1 = x.dot(z)                # vecvecmul(x, z)
        norm_temp2 = z.dot(z)                # vecvecmul(z, z)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)
        
        zeta = npbparams.SHIFT + 1.0 / norm_temp1

        if it == 1:
            print("\n   iteration           ||r||                 zeta")
        print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

        z.copy(result=x)
        x.scale(norm_temp2)
    
    return zeta


def conj_grad(a, x, z, p, q, r): 
    cgitmax = 25
    rho = 0.0

    # init
    q.set(0.0)
    z.set(0.0)
    x.copy(result=r)
    r.copy(result=p)

    rho = r.dot(r)                       # vecvecmul(r, r)

    for _ in range(1, cgitmax + 1):
        a.mult(p, q)                     # matvecmul(a, p)
        d = p.dot(q)
        alpha = rho / d
        rho0 = rho
        
        z.axpy(alpha, p)                 # scalarvecmul2( alpha, p, z)
        r.axpy(-alpha, q)                # scalarvecmul2(-alpha, q, r)
        rho = r.dot(r)                   # vecvecmul(r, r)
        beta = rho / rho0
        p.aypx(beta, r)                  # scalarvecmul1(beta, r, p)
        
    a.mult(z, r)                         # matvecmul(a, z)
    
    # r = x - r (utilizando aypx equivalente in-place)
    r.aypx(-1.0, x)                      
    rnorm = r.norm(PETSc.NormType.NORM_2)# norm(x, r)
    
    return rnorm