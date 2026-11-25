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


def init_cg(x, z, p, q, r):
    q.fill(0.0)
    z.fill(0.0)
    r[:] = x
    p[:] = x


def matvecmul(A_matrix, p, q):
    q[:] = A_matrix.dot(p)


def vecvecmul(v1, v2):
    return numpy.dot(v1, v2)


def scalarvecmul1(alpha, x, y):
    y *= alpha
    y += x


def scalarvecmul2(alpha, x, y):
    y += alpha * x


def norm(x, y):
    diff = x - y
    summ = numpy.dot(diff, diff)
    return math.sqrt(summ)


def update_x(z, norm_temp2, x):
    numpy.multiply(z, norm_temp2, out=x)
