import math

import numpy
from numba import njit


def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
    A_matrix = (a, colidx, rowstr, naa)
    return A_matrix, x, z, p, q, r


def cg_warmup(a_matrix, x, z, p, q, r):
    a, colidx, rowstr, naa = a_matrix

    for it in range(1, 1 + 1):
        rnorm = conj_grad(a_matrix, x, z, p, q, r)

        norm_temp1 = 0.0
        norm_temp2 = 0.0
        end = naa
        
        for j in range(0, end):
            norm_temp1 = norm_temp1 + x[j] * z[j]
            norm_temp2 = norm_temp2 + z[j] * z[j]
            
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)

        for j in range(0, end):
            x[j] = norm_temp2 * z[j]


def cg(a_matrix, x, z, p, q, r, npbparams):
    for i in range(npbparams.NA): # init_x()
        x[i] = 1.0

	
    zeta = 0.0
    for it in range(1, npbparams.NITER + 1):
        rnorm = conj_grad(a_matrix, x, z, p, q, r)

        norm_temp1 = 0.0
        norm_temp2 = 0.0
        end = npbparams.NA
        
        for j in range(0, end):
            norm_temp1 = norm_temp1 + x[j] * z[j]   # vecvecmul(x, z)
            norm_temp2 = norm_temp2 + z[j] * z[j]   # vecvecmul(z, z)
            
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)
        zeta = npbparams.SHIFT + 1.0 / norm_temp1

        if it == 1:
            print("\n   iteration           ||r||                 zeta")
        print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

        # normalize z to obtain x
        for j in range(0, end):
            x[j] = norm_temp2 * z[j]
            
    return zeta


@njit
def conj_grad(a_matrix, x, z, p, q, r): 
    a, colidx, rowstr, naa = a_matrix
    
    firstrow = 0
    lastrow = naa - 1
    firstcol = 0
    lastcol = naa - 1
    
    cgitmax = 25
    rho = 0.0

    #init
    for j in range(naa):
        q[j] = 0.0
        z[j] = 0.0
        r[j] = x[j]
        p[j] = r[j]

    end = lastcol - firstcol + 1
    for j in range(end):
        rho = rho + r[j]*r[j]         # vecvecmul(r, r)

    for _ in range(1, cgitmax+1):
		# matvecmul(a, p)
        end = lastrow - firstrow + 1
        for j in range(end):
            summ = 0.0
            for k in range (rowstr[j], rowstr[j+1]):
                summ = summ + a[k]*p[colidx[k]]
            q[j] = summ
        # end

        # vecvecmul(p, q)
        d = 0.0
        end = lastcol - firstcol + 1
        for j in range(end):
            d = d + p[j]*q[j]
        # end

        alpha = rho / d
        rho0 = rho

		# scalarvecmul2 ( alpha, p, z) and (-alpha, q, r)
        rho = 0.0
        end = lastcol - firstcol + 1
        for j in range(end):
            z[j] = z[j] + alpha*p[j] # scalarvecmul2( alpha, p, z)
            r[j] = r[j] - alpha*q[j] # scalarvecmul2(-alpha, q, r)
        # end

        # vecvecmul(r, r)
        end = lastcol - firstcol + 1 
        for j in range(end):
            rho = rho + r[j]*r[j]
        # end

        beta = rho / rho0

        # scalarvecmul1(beta, r, p)
        end = lastcol - firstcol + 1 
        for j in range(end):
            p[j] = r[j] + beta*p[j]
        # end
            
    # norm(x, r)
    summ = 0.0
    end = lastrow - firstrow + 1
    for j in range(end):
        d = 0.0
        for k in range(rowstr[j], rowstr[j+1]):
            d = d + a[k]*z[colidx[k]]
        r[j] = d

    end = lastcol - firstcol + 1
    for j in range(end):
        d = x[j] - r[j]
        summ = summ + d*d

    rnorm = math.sqrt(summ)
    # end
    return rnorm
