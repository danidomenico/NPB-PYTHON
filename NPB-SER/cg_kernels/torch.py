import torch


def prepare_data(a, colidx, rowstr, x, z, p, q, r, naa):
	colidx = torch.from_numpy(colidx)
	rowstr = torch.from_numpy(rowstr)
	a = torch.from_numpy(a)
    
	nnz = int(rowstr[naa].item())
    
	a = torch.sparse_csr_tensor(
        rowstr[:naa+1],
        colidx[:nnz],
        a[:nnz],
        size=(naa, naa),
        dtype=torch.float64
    )
    
	x = torch.from_numpy(x).to(torch.float64)
	z = torch.from_numpy(z).to(torch.float64)
	p = torch.from_numpy(p).to(torch.float64)
	q = torch.from_numpy(q).to(torch.float64)
	r = torch.from_numpy(r).to(torch.float64)
    
	return a, x, z, p, q, r

def cg_warmup(a, x, z, p , q, r):
	for it in range(1, 1 + 1):
		rnorm = conj_grad(a, x, z, p, q, r)

		norm_temp1 = torch.dot(x, z)                # vecvecmul(x, z)
		norm_temp2 = torch.dot(z, z)                # vecvecmul(z, z)
		norm_temp2 = torch.rsqrt(norm_temp2).item() # 1.0 / sqrt(norm_temp2)

		torch.mul(z, norm_temp2, out=x)


def cg(a, x, z, p , q, r, npbparams):
	x.fill_(1.0) # init_x()

	zeta = 0.0
	for it in range(1, npbparams.NITER + 1):
		rnorm = conj_grad(a, x, z, p, q, r)


		norm_temp1 = torch.dot(x, z)                # vecvecmul(x, z)
		norm_temp2 = torch.dot(z, z)                # vecvecmul(z, z)
		norm_temp2 = torch.rsqrt(norm_temp2).item() # 1.0 / sqrt(norm_temp2)
		zeta = npbparams.SHIFT + 1.0 / norm_temp1

		if it == 1:
			print("\n   iteration           ||r||                 zeta")
		print("    %5d       %20.14e%20.13e" % (it, rnorm, zeta))

		x = norm_temp2 * z
	
	return zeta


def conj_grad(a, x, z, p, q, r): 
	cgitmax = 25
	rho = 0.0

	# init
	q.zero_()
	z.zero_()
	r.copy_(x)
	p.copy_(r)

	rho = torch.dot(r, r)       	 # vecvecmul(r, r)

	for _ in range(1, cgitmax+1):
		q = torch.mv(a, p)    		 # matvecmul(a, p)
		d = torch.dot(p, q).item()	 # vecvecmul(p, q)
		alpha = rho / d
		rho0 = rho
		rho = 0.0
		z.add_(p, alpha=alpha)       # scalarvecmul2( alpha, p, z)
		r.add_(q, alpha=-alpha)      # scalarvecmul2(-alpha, q, r)
		rho = torch.dot(r, r).item() # vecvecmul(r, r)
		beta = rho / rho0
		p.mul_(beta).add_(r)		 # scalarvecmul1(beta, r, p)
	r = torch.mv(a, z)

	rnorm = torch.dist(x, r, p=2).item() # norm(x, r)
	return rnorm
