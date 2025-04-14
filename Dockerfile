FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set noninteractive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget curl git unzip libgl1-mesa-glx libosmesa6-dev patchelf \
    python3.9 python3.9-dev python3-pip python-is-python3 \
    libglew-dev libglfw3-dev libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Create conda env and install requirements
RUN conda create -n contrastive_rl python=3.9 -y && \
    /opt/conda/envs/contrastive_rl/bin/pip install -r /app/requirements.txt --no-deps

# Install extra pip packages in strict versions
RUN /opt/conda/envs/contrastive_rl/bin/pip install dm-acme[jax,tf] && \
    /opt/conda/envs/contrastive_rl/bin/pip install jax==0.4.10 jaxlib==0.4.10 && \
    /opt/conda/envs/contrastive_rl/bin/pip install ml_dtypes==0.2.0 && \
    /opt/conda/envs/contrastive_rl/bin/pip install dm-haiku==0.0.9 && \
    /opt/conda/envs/contrastive_rl/bin/pip install gymnasium-robotics && \
    /opt/conda/envs/contrastive_rl/bin/pip uninstall -y scipy && \
    /opt/conda/envs/contrastive_rl/bin/pip install scipy==1.12 && \
    /opt/conda/envs/contrastive_rl/bin/pip install torch==2.1.2 scikit-learn pandas && \
    /opt/conda/envs/contrastive_rl/bin/pip install 'cython<3'

# Set up default environment when shell starts
RUN echo "source activate contrastive_rl" > ~/.bashrc
ENV PATH="/opt/conda/envs/contrastive_rl/bin:$PATH"

# Copy repo files
WORKDIR /app
COPY . /app

# Set LD_LIBRARY_PATH for mujoco
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64

# Mujoco setup (you'll need to download mujoco manually or mount it in)
# If you have mujoco210, place it in ~/.mujoco inside container

CMD [ "/bin/bash" ]
