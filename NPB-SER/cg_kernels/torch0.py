import torch
import math
import os

# DICA DE OURO PARA BENCHMARK SERIAL:
# O PyTorch tenta usar todos os núcleos do processador por padrão.
# Para vetores relativamente pequenos, o tempo que a CPU gasta sincronizando 
# as threads (Overhead de Dispatch) é maior do que o tempo da conta em si.
# Como o NPB-SER é um benchmark estritamente serial, forçar 1 thread 
# acelera operações pequenas e reduz o thrashing de cache.
torch.set_num_threads(1)

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


def matvecmul(A_matrix, p, q):
    # Passamos direto para a engine C++ (ATen).
    # Assumimos que o tamanho de 'p' e 'q' já vêm fatiados do chamador.
    torch.mv(A_matrix, p, out=q)


def vecvecmul(v1, v2):
    return torch.dot(v1, v2).item()


def scalarvecmul1(alpha, x, y):
    # Original: y[i] = x[i] + alpha * y[i]
    # Executa a operação fundida (Fused Multiply-Add) em uma única chamada ao C++
    torch.add(x, y, alpha=alpha, out=y)


def scalarvecmul2(alpha, x, y):
    # Original: y[i] = y[i] + alpha * x[i]
    # Este método in-place já executa de forma otimizada em uma chamada
    y.add_(x, alpha=alpha)


def norm(x, y):
    # Calcula a Norma L2 de (x - y) nativamente sem alocar um tensor temporário
    return torch.dist(x, y, p=2).item()