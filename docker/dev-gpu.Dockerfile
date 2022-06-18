ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

ENV TORCH_CUDA_ARCH_LIST="6.1 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"
ENV WITH_CUDA="1"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV QT_X11_NO_MITSHM 1

# Set this until this is resolved https://github.com/pytorch/pytorch/issues/37377 or package versions are fixed
ENV 'MKL_THREADING_LAYER' = 'GNU'

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y wget ffmpeg libsm6 git ninja-build libglib2.0-0  libxrender-dev libxext6  build-essential python3-dev  binutils libproj-dev libcurl4 tzdata\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gdal-bin libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN gdal-config --version

# Install the Extensis MrSID DFK
RUN mkdir -p /workspaces/mrsid
WORKDIR /workspaces/mrsid
RUN wget https://bin.extensis.com/download/developer/MrSID_DSDK-9.5.4.4709-rhel6.x86-64.gcc531.tar.gz

RUN tar -xvf MrSID_DSDK-9.5.4.4709-rhel6.x86-64.gcc531.tar.gz

RUN mv MrSID_DSDK-9.5.4.4709-rhel6.x86-64.gcc531 sdk

ENV PATH="/workspaces/mrsid/sdk/Raster_DSDK/bin:${PATH}"
ENV PATH="/workspaces/mrsid/sdk/Lidar_DSDK/bin:${PATH}"

# Install dependencies of label-maker
RUN mkdir -p /workspaces/tippecanoe
RUN git clone https://github.com/mapbox/tippecanoe.git /workspaces/tippecanoe
WORKDIR /workspaces/tippecanoe
RUN make -j
RUN make install

WORKDIR /workspaces

# Note that yolactpp is using DCNv2_latest and this is a submodule 
# in the .gitmodules files we list the specific branch (pytorch version that we need)
RUN git clone --recurse-submodules -b model-performance-verification https://github.com/upabove-app/yolact-original.git /yolactpp

ENV CONDA_PREFIX=/opt/conda

## add conda to the path so we can execute it by name
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

RUN conda create --name sidewalk-env --clone base

COPY ./environment.yml /tmp/environment.yml
RUN conda env update  --file /tmp/environment.yml

RUN /opt/conda/bin/conda clean -ya

# the specific env
ENV CONDA_DEFAULT_ENV=sidewalk-env

# Configure .bashrc to drop into a conda env and immediately activate our TARGET env
RUN CONDA_DEFAULT_ENV=sidewalk-env conda init && echo 'conda activate "${CONDA_DEFAULT_ENV:-base}"' >>  ~/.bashrc

ENV LD_LIBRARY_PATH=/opt/conda/lib:/opt/conda/envs/sidewalk-env/lib:${LD_LIBRARY_PATH}

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx

RUN /opt/conda/bin/conda clean -ya

#  PATH into conda environment
ENV PATH=/opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# set pythonpath
ENV PYTHONPATH=/workspaces/sidewalk-detection:${PYTHONPATH}

# Compile 
WORKDIR  /yolactpp/external/DCNv2_latest
# git submodule update --remote 
# We check the specific commit per one of the opened issues in https://github.com/jinfagang/DCNv2_latest
RUN git checkout fa9b2fd740ced2a22e0e7e913c3bf3934fd08098
RUN  python setup.py build develop && \
    rm -Rf /root/.cache/pip


# specify vscode as the user name in the docker
# This user name should match that of the VS Code .devcontainer to allow seamless development inside the docker container via vscode 
ARG USERNAME=vscode
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # [Optional] Add sudo support for the non-root user - this is ok for development dockers only
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  # Cleanup
  && rm -rf /var/lib/apt/lists/* \
  # Set up git completion.
  && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc 


RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc"

# certain doc building tools such as sphinx and others are installed in local bin folder
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Specify matplotlib backend
WORKDIR /home/${USERNAME}/.config/matplotlib
RUN echo "backend : Agg" >> matplotlibrc

# RUN  ln -sf /usr/share/zoneinfo/America/New_York /etc/timezone && \
#     ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime && \

WORKDIR /workspaces/sidewalk-detection


