import math

import numpy
import scipy.sparse as sp


def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
    nnz = rowstr[naa]
    
    a_csr = sp.csr_matrix(
        (a[:nnz], colidx[:nnz], rowstr[:naa+1]), 
        shape=(naa, naa)
    )
    
    x = x[:naa]
    z = z[:naa]
    p = p[:naa]
    q = q[:naa]
    r = r[:naa]
    
    return a_csr, x, z, p, q, r


def cg_warmup(a, x, z, p , q, r):
    for it in range(1, 1 + 1):
        rnorm = conj_grad(a, x, z, p, q, r)

        norm_temp1 = numpy.dot(x, z)
        norm_temp2 = numpy.dot(z, z)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)

        numpy.multiply(z, norm_temp2, out=x)


def cg(a, x, z, p , q, r, npbparams):
    x.fill(1.0) # init_x()

    zeta = 0.0
    for it in range(1, npbparams.NITER + 1):
        rnorm = conj_grad(a, x, z, p, q, r)

        norm_temp1 = numpy.dot(x, z)      # vecvecmul(r, r)
        norm_temp2 = numpy.dot(z, z)      # vecvecmul(r, r)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)
        
        zeta = npbparams.SHIFT + 1.0 / norm_temp1

        if it == 1:
            print("\n   iteration           ||r||                 zeta")
        print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

        numpy.multiply(z, norm_temp2, out=x)
    
    return zeta


def conj_grad(a_csr, x, z, p, q, r): 
    cgitmax = 25

    # init
    q.fill(0.0)
    z.fill(0.0)
    r[:] = x
    p[:] = r

    rho = numpy.dot(r, r)         # vecvecmul(r, r)

    for _ in range(1, cgitmax+1):
        q[:] = a_csr.dot(p)    	  # matvecmul(a, p)
        d = numpy.dot(p, q)		  # vecvecmul(p, q)
        alpha = rho / d
        rho0 = rho
        z += alpha * p       	  # scalarvecmul2( alpha, p, z)
        r -= alpha * q      	  # scalarvecmul2(-alpha, q, r)
        
        rho = numpy.dot(r, r) 	  # vecvecmul(r, r)
        beta = rho / rho0
        
        p *= beta 				  # scalarvecmul1(beta, r, p)
        p += r
        
    r[:] = a_csr.dot(z)

    # norm(x, r)
    diff = x - r
    summ = numpy.dot(diff, diff)
    rnorm = math.sqrt(summ)
    # end
    
    return rnorm