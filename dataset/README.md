# Kernel Reasoning Trace Pipeline

Generates markdown reasoning traces for KernelBook pairs (PyTorch + Triton), selecting vLLM on CUDA GPUs or compiled Hugging Face models on Apple Silicon. All runtime options live in `config.json`.

## Requirements
- CUDA-enabled GPU with enough VRAM for the chosen model
- Python 3.10+
- `pip install -r requirements.txt` (see Dockerfile for the minimal list)
- Hugging Face credentials configured if the model/dataset is gated

## Config (`config.json`)
- `model`: vLLM model ID, tensor parallel settings, dtype, max length
- `dataset`: HF dataset name/config/split, optional limit and shuffling
- `generation`: decoding params (`max_new_tokens`, `temperature`, `stop_sequences`, …)
- `runtime`: batch size, worker threads, prompt template, output CSV path, overwrite flag

## Local Run
```
python -m dataset.trace_generator
```
Set `KERNEL_TRACE_CONFIG=/path/to/config.json` to use an alternate config. The script streams KernelBook rows, formats prompts, detects CUDA vs MPS (selecting the matching backend), generates reasoning traces, and writes `dataset/kernel_traces.csv`.

## Expected Output
- CSV columns: `sample_index`, `python_code`, `triton_code`, `reasoning_trace`, optional `task_id` or `source_id`
- Each reasoning trace is markdown-formatted and references the Triton code causally.

## Docker
Build and run on a cloud GPU:
```
docker build -t kernel-trace -f dataset/Dockerfile .
docker run --gpus all --rm -v $(pwd):/workspace kernel-trace
```
Override config by mounting a custom file and setting `KERNEL_TRACE_CONFIG` as shown above. On Apple Silicon, prefer running locally—the Docker image targets CUDA GPUs.

