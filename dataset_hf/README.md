# Hugging Face Reasoning Trace Generator

Generates reasoning traces for KernelBook dataset using Hugging Face transformers.

## Usage

```bash
python -m dataset_hf.trace_generator
```

Set `KERNEL_TRACE_CONFIG=/path/to/config.json` to use a custom config.

## Config

- `model.model_id`: Hugging Face model ID
- `model.dtype`: Model dtype (bfloat16/float16)
- `model.max_model_len`: Maximum sequence length
- `dataset.name`: Dataset name from Hugging Face
- `dataset.config`: Optional dataset config
- `dataset.split`: Dataset split
- `generation`: Generation parameters (max_new_tokens, temperature, top_p, stop_sequences)
- `runtime`: Runtime settings (num_threads, prompt_template, output_csv, overwrite)

## Output

CSV file with columns: `sample_index`, `python_code`, `triton_code`, `reasoning_trace`, and optional `task_id`/`source_id`.

