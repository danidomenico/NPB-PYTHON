import math
import numpy
from numba import njit


def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
    A_matrix = (a, colidx, rowstr)
    return A_matrix, x, z, p, q, r

@njit
def init_cg(x, z, p, q, r):
    for j in range(len(q)):
        q[j] = 0.0
        z[j] = 0.0
        r[j] = x[j]
        p[j] = r[j]

@njit
def update_x(z, norm_temp2, x):
    for j in range(len(x)):
        x[j] = norm_temp2 * z[j]

@njit
def matvecmul(A_matrix, p, q):
    a, colidx, rowstr = A_matrix
    nrows = len(q)
    for i in range(nrows):
        summ = 0.0
        for j in range(rowstr[i], rowstr[i+1]):
            summ += a[j] * p[colidx[j]]
        q[i] = summ

@njit
def vecvecmul(a, b):
    return numpy.dot(a, b)

@njit
def scalarvecmul1(alpha, x, y):
    for i in range(len(y)):
        y[i] = x[i] + alpha * y[i]

@njit
def scalarvecmul2(alpha, x, y):
    for i in range(len(y)):
        y[i] = y[i] + alpha * x[i]

@njit
def norm(x, y):
    summ = 0.0
    for i in range(len(x)):
        d = x[i] - y[i]
        summ += d * d
    return math.sqrt(summ)