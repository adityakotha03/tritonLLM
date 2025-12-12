import os
import re
import json
import pandas as pd
import multiprocessing as mp
from string import Template
from utils import clean_markdown_code_gpt_oss

# Tinker imports
from tinker import ServiceClient
from tinker.types import SamplingParams
from tinker_cookbook import renderers, tokenizer_utils

# --- PROMPT TEMPLATES ---

BASE_SYSTEM_PROMPT_STRING = """You are going to generate a reasoning trace of how a PyTorch function has been translated into
an equivalent Triton kernel.

I will give you:
1. The original PyTorch implementation.
2. A target Triton kernel that the PyTorch maps to.

Your task:
- Produce ONLY a step-by-step reasoning trace explaining how to translate the PyTorch code into Triton.
- DO NOT output any code, pseudo-code, or edited snippets — only reasoning in natural language.
- Assume the reader already knows PyTorch and Triton; focus on the mapping details, not basic introductions.

CONSTRAINTS:
- Your entire answer MUST stay within $max_tokens tokens.
- Be concise but precise. Prefer fewer, information-dense steps over verbose explanations.
- If you run out of budget, prioritize:
  1. tensor shapes & indexing logic
  2. parallelization strategy (program ids, block sizes, grid)
  3. memory layout and loads/stores
  4. numerics / edge cases (broadcasting, dtype, padding, masks)

Structure your answer as:

1. High-level goal
2. Tensor shapes and indexing
3. Parallelization & launch configuration
4. Memory access pattern
5. Numerics & correctness details
6. Summary checklist

Again: DO NOT output any Python or Triton code. ONLY the reasoning trace.

--- PYTORCH IMPLEMENTATION ---
$python_code

--- TRITON IMPLEMENTATION OR SKELETON (OPTIONAL) ---
$triton_code
"""

QC_MODEL_SYSTEM_PROMPT_STRING = """You are a strict but concise reviewer.

You are given ONLY a reasoning trace that explains how to translate a PyTorch implementation into a Triton kernel. You DO NOT see the original code.

Your job:
- Check the reasoning trace for logical errors, contradictions, or incomplete text.
- Output format must be a JSON with "result" (PASS/FAIL).

Constraints:
- Be brief and surgical.
- Do NOT rewrite the reasoning trace.
- Only diagnose issues.

Output format:
1. Verdict: PASS or FAIL.
2. Issues: Bullet points if FAIL.

Give your output in the form of a JSON that follows the given schema:
{
    "result": "" # one of "PASS" or "FAIL"
}

Here is the reasoning trace to review:

$reasoning_trace
"""

# --- HELPERS ---

def clean_harmony(raw_text):
    """Extracts JSON from the QC model output using regex."""
    pattern = r"<\|channel\|>final(?:.*?)<\|message\|>\s*(\{[\s\S]*\})(?:<\|end\|>|$)"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

# --- WORKER SETUP ---

# Global context for worker processes
worker_ctx = {}

def init_worker(cfg):
    """Initialize clients once per process."""
    try:
        client = ServiceClient()
        worker_ctx['base_client'] = client.create_sampling_client(base_model=cfg["base_model"])
        worker_ctx['qc_client'] = client.create_sampling_client(base_model=cfg["qc_model"])
        
        # Setup renderers/tokenizers
        base_tok = tokenizer_utils.get_tokenizer(model_name=cfg["base_model"])
        worker_ctx['base_renderer'] = renderers.get_renderer(cfg["renderer_name"], base_tok)
        worker_ctx['base_params'] = SamplingParams(
            max_tokens=cfg["max_tokens"], 
            temperature=cfg.get("base_model_temperature", 0.5), 
            stop=worker_ctx['base_renderer'].get_stop_sequences()
        )

        qc_tok = tokenizer_utils.get_tokenizer(model_name=cfg["qc_model"])
        worker_ctx['qc_renderer'] = renderers.get_renderer(cfg["renderer_name"], qc_tok)
        worker_ctx['qc_params'] = SamplingParams(
            max_tokens=cfg["max_tokens"], 
            temperature=cfg.get("qc_model_temperature", 0.5), 
            stop=worker_ctx['qc_renderer'].get_stop_sequences()
        )
        
        # Store config and templates in context
        worker_ctx['cfg'] = cfg
        worker_ctx['base_tmpl'] = Template(BASE_SYSTEM_PROMPT_STRING)
        worker_ctx['qc_tmpl'] = Template(QC_MODEL_SYSTEM_PROMPT_STRING)
        
    except Exception as e:
        print(f"Worker init failed: {e}")

def process_row(row):
    """Run generation and QC for a single row."""
    idx, python_code, triton_code = row
    cfg = worker_ctx.get('cfg')
    result = {"id": idx, "trace": None, "qc": None, "error": None}

    if not cfg:
        result["error"] = "Worker context not initialized"
        return None

    try:
        # 1. Base Model Generation
        base_prompt_data = [{
            "role": "user", 
            "content": worker_ctx['base_tmpl'].substitute(
                python_code=python_code, 
                triton_code=triton_code, 
                max_tokens=cfg["max_tokens"]
            )
        }]
        base_prompt = worker_ctx['base_renderer'].build_generation_prompt(base_prompt_data)
        base_out = worker_ctx['base_client'].sample(
            prompt=base_prompt, 
            sampling_params=worker_ctx['base_params'], 
            num_samples=1
        ).result()
        
        trace_obj, _ = worker_ctx['base_renderer'].parse_response(base_out.sequences[0].tokens)
        
        # Extract and clean trace
        raw_trace = trace_obj.get("content", "")
        cleaned_trace = clean_markdown_code_gpt_oss(raw_trace)
        result["trace"] = cleaned_trace

        # 2. QC Model Generation
        qc_prompt_data = [{
            "role": "user", 
            "content": worker_ctx['qc_tmpl'].substitute(reasoning_trace=cleaned_trace)
        }]
        qc_prompt = worker_ctx['qc_renderer'].build_generation_prompt(qc_prompt_data)
        qc_out = worker_ctx['qc_client'].sample(
            prompt=qc_prompt, 
            sampling_params=worker_ctx['qc_params'], 
            num_samples=1
        ).result()
        
        qc_obj, _ = worker_ctx['qc_renderer'].parse_response(qc_out.sequences[0].tokens)
        
        # Extract and parse QC JSON
        raw_qc = qc_obj.get("content", "")
        # The prompt might wrap the JSON in markdown code blocks, clean it first
        cleaned_qc_text = clean_markdown_code_gpt_oss(raw_qc) 
        qc_data = clean_harmony(cleaned_qc_text)

        if qc_data and qc_data.get("result") == "PASS":
            result["qc"] = "PASS"
            return result
        else:
            # QC Failed or Parse Failed -> Skip this sample
            return None

    except Exception as e:
        # Silently fail on error to keep logs clean, or print if debugging
        # print(f"Sample {idx} error: {e}")
        return None

# --- MAIN ---

def main():
    # 1. Load Config
    with open("config.json", "r") as f:
        cfg = json.load(f)

    output_file = "results.jsonl"
    target_count = cfg.get("num_samples", 1000)
    
    # 2. Identify Existing Completed IDs
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        existing_ids.add(str(data["id"]))
                except json.JSONDecodeError:
                    pass
    
    current_count = len(existing_ids)
    needed = target_count - current_count
    
    print(f"Status: {current_count}/{target_count} samples completed.")
    if needed <= 0:
        print("Target reached. Exiting.")
        return

    # 3. Load and Filter Data
    print("Loading dataset...")
    df = pd.read_parquet("hf://datasets/GPUMODE/KernelBook/dataset_permissive.parquet")
    
    # Filter by license
    df = df[df["licenses"].apply(lambda x: "MIT" in x or "Apache-2.0" in x)]
    
    # Filter out already processed IDs
    # Ensure index is treated as string for comparison
    df["id"] = df.index.astype(str)
    df = df[~df["id"].isin(existing_ids)]
    
    available_samples = len(df)
    if available_samples == 0:
        print("No new samples available in dataset.")
        return

    # 4. Select Tasks (Oversample to account for QC failures)
    # We grab 2x what we need, or all remaining if less than that.
    sample_size = min(available_samples, needed * 2)
    # If the remaining pool is very large, just cap it to ensure we don't queue forever
    if sample_size < needed: 
        print(f"Warning: Only {available_samples} samples left, but {needed} needed.")
    
    print(f"Selecting {sample_size} candidates to generate {needed} passing samples...")
    sampled_df = df.sample(n=sample_size)
    
    tasks = [(str(i), r["python_code"], r["triton_code"]) for i, r in sampled_df.iterrows()]

    # 5. Run Pool
    print(f"Starting pool with {cfg.get('num_workers', 4)} workers...")
    
    with mp.Pool(processes=cfg.get("num_workers", 4), initializer=init_worker, initargs=(cfg,)) as pool:
        with open(output_file, "a", encoding="utf-8") as f:
            
            # imap_unordered yields results as they finish
            for res in pool.imap_unordered(process_row, tasks):
                
                if res is not None:
                    # Write result
                    f.write(json.dumps(res) + "\n")
                    f.flush()
                    current_count += 1
                    
                    print(f"Saved sample {res['id']} | Progress: {current_count}/{target_count}")
                    
                    # Stop if we hit the target
                    if current_count >= target_count:
                        print("Target count reached.")
                        pool.terminate()
                        break

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()