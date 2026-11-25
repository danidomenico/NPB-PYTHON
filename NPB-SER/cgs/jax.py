import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse


def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
    nrows = naa
    ncols = naa
    nnz = int(rowstr[nrows])

    colidx_jnp = jnp.array(colidx[:nnz], dtype=jnp.int32)
    rowstr_jnp = jnp.array(rowstr[:nrows+1], dtype=jnp.int32)
    a_jnp = jnp.array(a[:nnz], dtype=jnp.float64)

    mat = jax_sparse.CSR((a_jnp, colidx_jnp, rowstr_jnp), shape=(nrows, ncols))

    x = jnp.array(x[:naa], dtype=jnp.float64)
    z = jnp.array(z[:naa], dtype=jnp.float64)
    p = jnp.array(p[:naa], dtype=jnp.float64)
    q = jnp.array(q[:naa], dtype=jnp.float64)
    r = jnp.array(r[:naa], dtype=jnp.float64)
    
    return mat, x, z, p, q, r


def cg_warmup(mat, x, z, p, q, r):
    for it in range(1, 1 + 1):
        rnorm_jax, x, z, p, q, r = conj_grad(mat, x, z, p, q, r)

        norm_temp1 = jnp.dot(x, z)                  # vecvecmul(x, z)
        norm_temp2 = jnp.dot(z, z)                  # vecvecmul(z, z)
        norm_temp2 = 1.0 / jnp.sqrt(norm_temp2)     # 1.0 / sqrt(norm_temp2)

        x = norm_temp2 * z
    
    return x, z, p, q, r


def cg(mat, x, z, p, q, r, npbparams):
    x = jnp.ones_like(x) # init_x()

    zeta = 0.0
    for it in range(1, npbparams.NITER + 1):
        rnorm_jax, x, z, p, q, r = conj_grad(mat, x, z, p, q, r)
        rnorm = float(rnorm_jax)

        norm_temp1 = float(jnp.dot(x, z))           # vecvecmul(x, z)
        norm_temp2 = float(jnp.dot(z, z))           # vecvecmul(z, z)
        norm_temp2 = 1.0 / math.sqrt(norm_temp2)    # 1.0 / sqrt(norm_temp2)
        
        zeta = npbparams.SHIFT + 1.0 / norm_temp1

        if it == 1:
            print("\n   iteration           ||r||                 zeta")
        print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

        x = norm_temp2 * z
    
    return zeta


@jax.jit
def conj_grad(mat, x, z, p, q, r): 
    cgitmax = 25

    # init
    q = jnp.zeros_like(x)
    z = jnp.zeros_like(x)
    r = x
    p = r

    rho = jnp.dot(r, r)                             # vecvecmul(r, r)

    def body_fun(i, val):
        p_val, q_val, r_val, z_val, rho_val = val
        
        q_val = jax_sparse.csr_matvec(mat, p_val)   # matvecmul(a, p)
        d = jnp.dot(p_val, q_val)
        alpha = rho_val / d
        rho0 = rho_val
        
        z_val = z_val + alpha * p_val               # scalarvecmul2( alpha, p, z)
        r_val = r_val - alpha * q_val               # scalarvecmul2(-alpha, q, r)
        
        rho_val = jnp.dot(r_val, r_val)             # vecvecmul(r, r)
        beta = rho_val / rho0
        
        p_val = r_val + beta * p_val                # scalarvecmul1(beta, r, p)
        
        return p_val, q_val, r_val, z_val, rho_val

    init_val = (p, q, r, z, rho)
    p, q, r, z, rho = jax.lax.fori_loop(1, cgitmax + 1, body_fun, init_val)

    r = jax_sparse.csr_matvec(mat, z)               # matvecmul(a, z)

    diff = x - r
    summ = jnp.dot(diff, diff)
    rnorm = jnp.sqrt(summ)                          # norm(x, r)
    
    return rnorm, x, z, p, q, r
