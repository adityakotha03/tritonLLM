FROM python:3.12-slim

WORKDIR /

COPY requirements.txt /

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /
COPY eval/eval.py /eval/eval.py
COPY eval/eval_helpers.py /eval/eval_helpers.py

CMD ["python3", "-u", "rp_handler.py"]