# Gemini Reasoning Trace Generator

Generates reasoning traces for KernelBook dataset using Google's Gemini API with multi-threaded processing.

## Setup

1. Install dependencies:
```bash
pip install google-generativeai pandas datasets tqdm
```

2. Set up API key:
   - Add your `GEMINI_API_KEY` to `config.json` under `"api_key"`, or
   - Set environment variable: `export GEMINI_API_KEY=your_key_here`

## Usage

```bash
python dataset_gemini/trace_generator.py
```

Or with custom config:
```bash
GEMINI_TRACE_CONFIG=/path/to/config.json python dataset_gemini/trace_generator.py
```

## Config

- `api_key`: Gemini API key (or set `GEMINI_API_KEY` env var)
- `model_name`: Gemini model to use (default: `gemini-2.0-flash-exp`)
- `dataset`: Dataset configuration
- `generation`: Generation parameters (temperature, top_p, max_new_tokens, stop_sequences)
- `runtime`: Runtime settings (num_threads, prompt_template, output_csv, save_interval, overwrite)

## Features

- Multi-threaded processing for parallel API calls
- Automatic checkpointing at configurable intervals
- Resume support from existing CSV files
- Error handling for individual sample failures

## Output

CSV file with columns: `sample_index`, `python_code`, `triton_code`, `reasoning_trace`, and optional `task_id`/`source_id`.

