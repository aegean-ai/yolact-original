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

# Create first the conda environment:

RUN conda env create -f /yolactpp/environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure pytorch is installed and CUDA is seen:"
RUN python -c "import torch" \ "torch.cuda.is_available(), flush=True"


# Delete the DCNv2 and replace it with the latest DCNv2 with the right branch
#RUN rm -rf /yolactpp/external/DCNv2
# RUN git clone  --branch pytorch_1.7 https://github.com/upabove-app/DCNv2.git /yolactpp/external/DCNv2

# install GDAL 
RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
#COPY ./requirements.txt /tmp/requirements.txt 
# RUN pip install -r /tmp/requirements.txt
# RUN rm -Rf /.cache/pip

# set pythonpath
ENV PYTHONPATH=/yolactpp:${PYTHONPATH}


# certain doc building tools such as sphinx and others are installed in local bin folder
ENV PATH="/home/root/.local/bin:${PATH}"


# The code to run when container is started:
COPY run.py entrypoint.sh ./
ENTRYPOINT ["./entrypoint.sh"]

# Install DCNv2
WORKDIR /yolactpp/external/DCNv2
RUN ./make.sh && \
    rm -Rf /root/.cache/pip

