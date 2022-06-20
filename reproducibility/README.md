# NAS Parallel Benchmark Kernels with Python: A performance and programming effort analysis focusing on GPUs
- Paper published on 30th Euromicro International Conference on Parallel, Distributed and Network-based Processing ([PDP 2022](https://pdp2022.infor.uva.es/))

### Authors: 
- Daniel Di Domenico - Federal University of Pelotas (UFPEL)
- Gerson G. H. Cavalheiro - Federal University of Pelotas (UFPEL)
- João V. F. Lima - Federal University of Santa Maria (UFSM)

### What is here?
- This repository makes available the environment that can be use to reproduce the experimental results described in the paper.

### Folders:

#### docker
- Contains a Dockerfile with the environment required to execute the experiments. To be used, this Dockerfile requires the [Nvidia Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

#### profiling
- Constains proling files obtained from GPU executions with *nvprof* tool, as well as R files to process and generate charts from them.

#### results
- Constains data from the results of the executions on the platform and a R file to generate charts from it.


