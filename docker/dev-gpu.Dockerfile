ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

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
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 git ninja-build libglib2.0-0  libxrender-dev libxext6  build-essential python3-dev  binutils libproj-dev gdal-bin libgdal-dev libcurl4\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN gdal-config --version

# Note that yolactpp is using DCNv2 and this is a submodule - in the .gitmodules files we list the specific branch (pytorch version that we need)
RUN git clone --recurse-submodules -b model-performance-verification https://github.com/upabove-app/yolact-original.git /yolactpp

ENV CONDA_PREFIX=/opt/conda

## add conda to the path so we can execute it by name
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

RUN conda create --name sidewalk-env --clone base

COPY ./environment.yml /tmp/environment.yml
RUN conda env update  --file /tmp/environment.yml

RUN /opt/conda/bin/conda clean -ya

# #This is now  replaced with conda gdal but kept here temporarily
# # install GDAL - this is needed here as its currebtly unknown how to pass the options in the environment.yml file
# RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"

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
RUN cd /workspaces/sidewalk-detection/segmentation/yolact-original/external/DCNv2_latest && \
    python setup.py build develop && \
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
#ENV DEBIAN_FRONTEND=

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc"

# certain doc building tools such as sphinx and others are installed in local bin folder
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Specify matplotlib backend
WORKDIR /${USERNAME}/.config/matplotlib
RUN echo "backend : Agg" >> matplotlibrc

WORKDIR /workspaces/sidewalk-detection

