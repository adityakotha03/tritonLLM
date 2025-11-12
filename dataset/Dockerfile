FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KERNEL_TRACE_CONFIG=/workspace/dataset/config.json

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    python3 -m pip install \
      vllm \
      datasets \
      pandas \
      transformers

WORKDIR /workspace

COPY . /workspace

CMD ["python3", "-m", "dataset.trace_generator"]

