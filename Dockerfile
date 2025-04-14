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

# Create conda env and install dependencies
COPY requirements.txt /app/requirements.txt
RUN conda create -n contrastive_rl python=3.9 -y && \
    echo "conda activate contrastive_rl" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda activate contrastive_rl && pip install -r /app/requirements.txt --no-deps"

# Install extra pip packages in strict versions
RUN /bin/bash -c "source ~/.bashrc && conda activate contrastive_rl && \
    pip install dm-acme[jax,tf] \
    && pip install jax==0.4.10 jaxlib==0.4.10 \
    && pip install ml_dtypes==0.2.0 \
    && pip install dm-haiku==0.0.9 \
    && pip install gymnasium-robotics \
    && pip uninstall -y scipy && pip install scipy==1.12 \
    && pip install torch==2.1.2 scikit-learn pandas \
    && pip install 'cython<3'"

# Copy repo files
WORKDIR /app
COPY . /app

# Set LD_LIBRARY_PATH for mujoco
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64

# Mujoco setup (you'll need to download mujoco manually or mount it in)
# If you have mujoco210, place it in ~/.mujoco inside container

CMD [ "/bin/bash" ]
