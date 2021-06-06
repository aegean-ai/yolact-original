ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"
ENV WITH_CUDA="1"
ENV NVIDIA_VISIBLE_DEVICES="0,1"
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 git ninja-build libglib2.0-0  libxrender-dev libxext6  build-essential python3-dev  binutils libproj-dev gdal-bin libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note that yolactpp is using DCNv2 and this is a submodule - in the .gitmodules files we list the specific branch (pytorch version that we need)
RUN git clone --recurse-submodules https://github.com/upabove-app/yolact-original.git /yolactpp

# set pythonpath
ENV PYTHONPATH=/yolactpp:${PYTHONPATH}

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

## add conda to the path so we can execute it by name
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

# certain doc building tools such as sphinx and others are installed in local bin folder
ENV PATH=/opt/conda/envs/yolactpp-env/bin:${PATH}
ENV PATH="/home/root/.local/bin:${PATH}"


# The code to run when container is started:
ENV CONDA_PREFIX=/opt/conda

## Create /entry.sh which will be our new shell entry point. This performs actions to configure the environment
## before starting a new shell (which inherits the env).
## The exec is important! This allows signals to pass
COPY    conda_run.sh /conda_run.sh
COPY    conda_entry.sh /conda_entry.sh
RUN     chmod +x /conda_run.sh && chmod +x /conda_entry.sh

## Tell the docker build process to use this for RUN.
## The default shell on Linux is ["/bin/sh", "-c"], and on Windows is ["cmd", "/S", "/C"]
SHELL ["/conda_run.sh"]

## Now, every following invocation of RUN will start with the entry script
RUN     conda update -n base conda -y \
     &&  conda install -n base pip

# Create the conda environment. 
COPY ./environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml

# install GDAL - this is needed here as its currebtly unknown how to pass the options in the environment.yml file
RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"

## I added this variable such that I have the entry script activate a specific env
ENV CONDA_DEFAULT_ENV=yolactpp-env

## Configure .bashrc to drop into a conda env and immediately activate our TARGET env
RUN CONDA_DEFAULT_ENV=yolactpp-env conda init && echo 'conda activate "${CONDA_DEFAULT_ENV:-base}"' >>  ~/.bashrc

WORKDIR /yolactpp
ENTRYPOINT ["/conda_entry.sh"]


