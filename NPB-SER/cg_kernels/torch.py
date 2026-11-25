import torch
import math
import os

def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
    colidx_t = torch.from_numpy(colidx)
    rowstr_t = torch.from_numpy(rowstr)
    a_t = torch.from_numpy(a)
    
    nnz = int(rowstr_t[naa].item())
    
    A_matrix = torch.sparse_csr_tensor(
        rowstr_t[:naa+1],
        colidx_t[:nnz],
        a_t[:nnz],
        size=(naa, naa),
        dtype=torch.float64
    )
    
    x_t = torch.from_numpy(x).to(torch.float64)
    z_t = torch.from_numpy(z).to(torch.float64)
    p_t = torch.from_numpy(p).to(torch.float64)
    q_t = torch.from_numpy(q).to(torch.float64)
    r_t = torch.from_numpy(r).to(torch.float64)
    
    return A_matrix, x_t, z_t, p_t, q_t, r_t


def init_cg(x, z, p, q, r):
    q.zero_()
    z.zero_()
    r.copy_(x)
    p.copy_(x)


def matvecmul(A_matrix, p, q):
    torch.mv(A_matrix, p, out=q)


def vecvecmul(v1, v2):
    return torch.dot(v1, v2)


def scalarvecmul1(alpha, x, y):
    torch.add(x, y, alpha=alpha, out=y)


def scalarvecmul2(alpha, x, y):
    y.add_(x, alpha=alpha)


def norm(x, y):
    return torch.dist(x, y, p=2)


def update_x(z, norm_temp2, x):
    return torch.mul(z, norm_temp2, out=x)
