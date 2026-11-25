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


def init_cg(x, z, p, q, r):
    q.set(0.0)
    z.set(0.0)
    x.copy(result=r)
    x.copy(result=p)


def matvecmul(a_matrix, p, q):
    a_matrix.mult(p, q)


def vecvecmul(v1, v2):
    return v1.dot(v2)


def scalarvecmul1(alpha, x, y):
    y.aypx(alpha, x)


def scalarvecmul2(alpha, x, y):
    y.axpy(alpha, x)


def norm(x, y):
    y.aypx(-1.0, x)                      
    return y.norm(PETSc.NormType.NORM_2)


def update_x(z, norm_temp2, x):
    z.copy(result=x)
    x.scale(norm_temp2)
