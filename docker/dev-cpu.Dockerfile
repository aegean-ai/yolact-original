ARG PYTORCH="1.7.0"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV QT_X11_NO_MITSHM 1

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 git ninja-build libglib2.0-0  libxrender-dev libxext6  build-essential python3-dev  binutils libproj-dev gdal-bin libgdal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note that yolactpp is using DCNv2 and this is a submodule - in the .gitmodules files we list the specific branch (pytorch version that we need)
# RUN git clone --recurse-submodules https://github.com/upabove-app/yolact-original.git /yolactpp

# set pythonpath
ENV PYTHONPATH=/yolactpp:${PYTHONPATH}

#RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

## add conda to the path so we can execute it by name
ENV PATH=${CONDA_PREFIX}/bin:${PATH}

# The code to run when container is started:
ENV CONDA_PREFIX=/opt/conda

## Create /entry.sh which will be our new shell entry point. This performs actions to configure the environment
## before starting a new shell (which inherits the env).
## The exec is important! This allows signals to pass
#COPY    conda_run.sh /conda_run.sh
#COPY    conda_entry.sh /conda_entry.sh
#RUN     chmod +x /conda_run.sh && chmod +x /conda_entry.sh

## Tell the docker build process to use this for RUN.
## The default shell on Linux is ["/bin/sh", "-c"], and on Windows is ["cmd", "/S", "/C"]
# SHELL ["/conda_run.sh"]

## Now, every following invocation of RUN will start with the entry script
RUN     conda update -n base conda -y \
     &&  conda install -n base pip

# Create the conda environment. 
COPY ./environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml


RUN /opt/conda/bin/conda clean -ya

# install GDAL - this is needed here as its currebtly unknown how to pass the options in the environment.yml file
# replaced with conda gdal
#RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"

## I added this variable such that I have the entry script activate a specific env
ENV CONDA_DEFAULT_ENV=yolactpp-env

## Configure .bashrc to drop into a conda env and immediately activate our TARGET env
RUN CONDA_DEFAULT_ENV=yolactpp-env conda init && echo 'conda activate "${CONDA_DEFAULT_ENV:-base}"' >>  ~/.bashrc


ENV LD_LIBRARY_PATH /opt/conda/lib:/opt/conda/envs/cresi/lib:${LD_LIBRARY_PATH}

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx

RUN /opt/conda/bin/conda clean -ya

#  PATH into conda environment
ENV PATH /opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH


# specify vscode as the user name in the docker
# This user name should match that of the VS Code .devcontainer to allow seamless development inside the docker container via vscode 
ARG USERNAME=vscode
ARG USER_UID=1000
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
ENV DEBIAN_FRONTEND=

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc"

# certain doc building tools such as sphinx and others are installed in local bin folder
ENV PATH="/home/USERNAME/.local/bin:${PATH}"

# Expose ports needed for development
EXPOSE 8100

# Specify matplotlib backend
WORKDIR /${USERNAME}/.config/matplotlib
RUN echo "backend : Agg" >> matplotlibrc

WORKDIR /workspaces/sidewalk
#ENTRYPOINT ["/conda_entry.sh"]


