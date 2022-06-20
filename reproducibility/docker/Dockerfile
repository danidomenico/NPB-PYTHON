FROM nvidia/cuda:11.2.0-devel

#################################################################
########################## REPOSITORY ###########################
#################################################################
RUN apt-get update

#Install HTOP
RUN apt-get -y install htop

#Install CURL
RUN apt-get -y install curl


#################################################################
######################## NVIDIA HPC SDK #########################
#################################################################
RUN mkdir -p /opt/nvidia

#Download NVIDIA HPC SDK (https://developer.nvidia.com/nvidia-hpc-sdk-212-downloads) for a local directory and copy to image
COPY <<local directory>>/nvhpc_2021_212_Linux_x86_64_cuda_11.2 /opt/nvidia/nvhpc_files

#NVHPC_SILENT (required) Set this variable to "true" to enable silent installation.
ENV NVHPC_SILENT="true"
#NVHPC_INSTALL_DIR (required) Set this variable to a string containing the desired installation location, e.g. /opt/nvidia/hpc_sdk.
#NVHPC_INSTALL_TYPE (required) Set this variable to select the type of install. The accepted values are "single" for a single system install or "network" for a network install.
ENV NVHPC_INSTALL_TYPE="single"
#NVHPC_INSTALL_LOCAL_DIR	(required for network install) Set this variable to a string containing the path to a local file system when choosing a network install.
#NVHPC_DEFAULT_CUDA	(optional) Set this variable to the desired CUDA version in the form of XX.Y, e.g. 10.1 or 11.0.
#NVHPC_STDPAR_CUDACC	(optional) Set this variable to force C++ stdpar GPU-compilation to target a specific compute capability by default, e.g. 60, 70, 75, etc.

#Install NVIDIA HPC SDK
RUN /opt/nvidia/nvhpc_files/install

ENV PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/bin:${PATH}"
ENV MANPATH="${MANPATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/compilers/man"

ENV PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/comm_libs/mpi/bin:${PATH}"
ENV MANPATH="${MANPATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/comm_libs/mpi/man"

#Remove installing files from image
RUN rm -rf /opt/nvidia/nvhpc_files


#################################################################
####################### PYTHON and NUMBA ########################
#################################################################
#Install ANACONDA to enable CUDA support to python3
RUN mkdir -p /opt/anaconda_tmp

RUN curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh --output /opt/anaconda_tmp/anaconda.sh

RUN bash /opt/anaconda_tmp/anaconda.sh -b -p /opt/anaconda

RUN /opt/anaconda/bin/conda install -y numba cudatoolkit py-cpuinfo python=3

ENV PATH="/opt/anaconda/bin:${PATH}"

RUN rm -rf /opt/anaconda_tmp


#################################################################
########################## ENTRYPOINT ###########################
#################################################################
#Keep the original entrypoint of the image
##


