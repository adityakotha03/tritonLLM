FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

WORKDIR /

# Install Python 3.12 and venv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /
COPY eval/eval.py /eval/eval.py
COPY eval/eval_helpers.py /eval/eval_helpers.py

CMD ["python3", "-u", "rp_handler.py"]