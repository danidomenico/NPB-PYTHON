# NAS Parallel Benchmarks with Python - Serial and GPU versions

The NPB programs ([version 3.4.1](https://www.nas.nasa.gov/publications/npb.html)) were ported to Python based on [C++](https://github.com/GMAP/NPB-CPP) and [CUDA](https://github.com/GMAP/NPB-GPU) implementations.
We employed [Numba](https://numba.pydata.org/numba-doc/latest/cuda/index.html) environment to design the NPB CUDA programs in Python.

## How to cite this work
Daniel Di Domenico, Gerson G. H. Cavalheiro, João V. F. Lima, **NAS Parallel Benchmark Kernels with Python: A performance and programming effort analysis focusing on GPUs**, *2022 30th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*, 2022, pp. 26-33, doi: [10.1109/PDP55904.2022.00013](https://doi.org/10.1109/PDP55904.2022.00013)

Daniel Di Domenico, João V. F. Lima, & Gerson G. H. Cavalheiro. **NAS Parallel Benchmarks with Python: a performance and programming effort analysis focusing on GPUs**, *The Journal of Supercomputing (2022)*, doi: [10.1007/s11227-022-04932-3](https://doi.org/10.1007/s11227-022-04932-3)

### Folders inside the  project:

**NPB-SER** - This directory contains the sequential version implemented with Python.

**NPB-CUDA** - This directory contains the GPU version using Numba to enable CUDA support in Python.

# The NPB programs

Each file contains the implemented version of the kernels/applications:

## Kernels

	EP - Embarrassingly Parallel, floating-point operation capacity
	MG - Multi-Grid, non-local memory accesses, short- and long-distance communication
	CG - Conjugate Gradient, irregular memory accesses and communication
	FT - discrete 3D fast Fourier Transform, intensive long-distance communication
	IS - Integer Sort, integer computation and communication
	
## Pseudo-applications

	BT - Block Tri-diagonal solver
	LU - Scalar Penta-diagonal solver
	SP - Lower-Upper Gauss-Seidel solver


# Software Requirements

*Warning: our experiments were executed using the following environment*

	CUDA 11.2
	Python 3.8.8
	Conda 4.11.0
	numpy 1.20.0
	numba 0.53.1
	cudatoolkit 10.2.89
	cpuinfo 8.0.0 

**More info about the environment can be found in the** [reproducibility](reproducibility) **folder.**


# How to Execute

Enter the directory from the version desired and execute:

`$ python _BENCHMARK.py -c _WORKLOAD_CLASS`

_BENCHMARKs are: 
		
	BT, CG, EP, FT, IS, LU, MG, SP
																										
_WORKLOAD_CLASS are: 
	
	Class S: small for quick test purposes
	Class W: workstation size (a 90's workstation; now likely too small)	
	Classes A, B, C: standard test problems; ~4X size increase going from one class to the next	
	Classes D, E, F: large test problems; ~16X size increase from each of the previous Classes

Command example:

`$ python EP.py -c A`


## Configurations for executions

NPB-SER offers timers that can be activated aiming to profile the application.
To enable these timers, create a dummy file 'timer.flag' in the main directory of the NPB-SER.

NPB-GPU allows configuring the number of threads per block of each GPU kernel in the benchmarks.
Also, it is possible to enable additional timers for profiling pursose.
The user can define such configurations using the file `config/gpu_config.py`.
