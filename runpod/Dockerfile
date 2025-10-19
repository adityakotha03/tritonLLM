FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

WORKDIR /

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /
COPY eval/eval.py /eval/eval.py
COPY eval/eval_helpers.py /eval/eval_helpers.py

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

CMD ["python3", "-u", "rp_handler.py"]