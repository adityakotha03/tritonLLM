FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

WORKDIR /

# Install Python 3.12 and build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and create venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

COPY requirements.txt /

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /
COPY eval/eval.py /eval/eval.py
COPY eval/eval_helpers.py /eval/eval_helpers.py

CMD ["python3", "-u", "rp_handler.py"]