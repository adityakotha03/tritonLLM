import os
import pandas as pd
import json
import tinker
from string import Template
from tinker.types import SamplingParams
from tinker_cookbook import renderers, tokenizer_utils

BASE_SYSTEM_PROMPT_STRING = """You are helping translate a PyTorch implementation into an equivalent Triton kernel.

I will give you:
1. The original PyTorch implementation.
2. (Optional) A draft or target Triton kernel that the PyTorch should map to.

Your task:
- Produce ONLY a step-by-step reasoning trace explaining how to translate the PyTorch code into Triton.
- DO NOT output any code, pseudo-code, or edited snippets — only reasoning in natural language.
- Assume the reader already knows PyTorch and Triton; focus on the mapping details, not basic introductions.

CONSTRAINTS:
- Your entire answer MUST stay within $max_tokens tokens.
- Be concise but precise. Prefer fewer, information-dense steps over verbose explanations.
- If you run out of budget, prioritize:
  1) tensor shapes & indexing logic
  2) parallelization strategy (program ids, block sizes, grid)
  3) memory layout and loads/stores
  4) numerics / edge cases (broadcasting, dtype, padding, masks)

Structure your answer as:

1. High-level goal
   - One or two sentences.

2. Tensor shapes and indexing
   - Describe input/output tensor shapes.
   - Explain how PyTorch indexing maps to Triton block/program IDs.
   - Call out any non-trivial broadcasting or view/reshape logic.

3. Parallelization & launch configuration
   - Describe which dimensions are parallelized via `program_id` and how.
   - Explain the choice of block sizes and how they cover the full tensor.
   - Note any tail-handling or masking required.

4. Memory access pattern
   - Explain how each load/store in Triton corresponds to PyTorch reads/writes.
   - Note any need for contiguous views, striding, or reordering.

5. Numerics & correctness details
   - Mention any corner cases (e.g., padding, boundary checks, type casts).
   - Call out reductions, atomic operations, or accumulation patterns if present.

6. Summary checklist
   - 4–6 bullet points that someone can quickly scan to verify the Triton matches the PyTorch.

Again: DO NOT output any Python or Triton code. ONLY the reasoning trace.

--- PYTORCH IMPLEMENTATION ---
$python_code

--- TRITON IMPLEMENTATION OR SKELETON (OPTIONAL) ---
$triton_code
"""

BASE_SYSTEM_PROMPT = Template(BASE_SYSTEM_PROMPT_STRING)

QC_MODEL_SYSTEM_PROMPT_STRING = """You are a strict but concise reviewer.

You are given ONLY a reasoning trace that explains how to translate a PyTorch implementation into a Triton kernel. You DO NOT see the original code.

Your job:
- Check the reasoning trace for:
  - Clearly incorrect or self-contradictory logic (e.g., claims that conflict with each other).
  - Steps that are impossible or nonsensical for Triton/PyTorch (e.g., using non-existent Triton concepts).
  - Incomplete or abruptly cut-off sentences and bullet points.
- You are NOT asked to validate the actual correctness of the kernel, only to catch obviously broken reasoning or text.

Constraints:
- Be brief and surgical.
- Do NOT rewrite the reasoning trace.
- Do NOT propose an alternative solution.
- Only diagnose issues.

Output format:

1. Verdict:
   - One of: `PASS` or `FAIL`.

2. Issues (if any):
   - If `PASS`: write "No obvious logical errors or incomplete sentences found."
   - If `FAIL`: list each problem as a short bullet point, e.g.:
     - Logical: <short description>
     - Incomplete sentence: <short description>
     - Contradiction: <short description>

Give your output in the form of a JSON that follows the given schema:

\{
    "result": "" # one of "PASS" or "FAIL",
    "analysis": "" # details on issues as described earlier
\}

Here is the reasoning trace to review:

$reasoning_trace

"""

QC_MODEL_SYSTEM_PROMPT = Template(QC_MODEL_SYSTEM_PROMPT_STRING)

def main():
    cfg = json.load("config.json")
    service_client = tinker.ServiceClient()

    # Get base model (GPT-OSS-120B) objects
    base_sampling_client = service_client.create_sampling_client(
        base_model=cfg["base_model"]
    )
    base_tokenizer = tokenizer_utils.get_tokenizer(
        base_model=cfg["base_model"]
    )
    base_renderer = renderers.get_renderer(cfg["renderer_name"], base_tokenizer)
    base_stop_sequences = base_renderer.get_stop_sequences()
    base_sampling_params = SamplingParams(max_tokens=4096, temperature=0.5, stop=base_stop_sequences)

    # Get base model (GPT-OSS-20B) objects
    qc_sampling_client = service_client.create_sampling_client(
        base_model=cfg["qc_model"]
    )
    qc_tokenizer = tokenizer_utils.get_tokenizer(
        base_model=cfg["qc_model"]
    )
    qc_renderer = renderers.get_renderer(cfg["renderer_name"], qc_tokenizer)
    qc_stop_sequences = qc_renderer.get_stop_sequences()
    qc_sampling_params = SamplingParams(max_tokens=4096, temperature=0.5, stop=qc_stop_sequences)

    df = pd.read_parquet("hf://datasets/GPUMODE/KernelBook/dataset_permissive.parquet")
    sampled_df = df.sample(n=cfg["num_samples"])

    license_code_samples_condition = "MIT" in sampled_df["licenses"] or "Apache-2.0" in sampled_df["licenses"]
    triton_code_samples = sampled_df.loc["triton_code", license_code_samples_condition]
    
    python_code_samples = sampled_df.loc["python_code", license_code_samples_condition]

    reasoning_traces = []
    qc_analyses = []

    def get_base_renderer_prompt(
        python_code,
        triton_code
    ):
        return [{
            "role": "user",
            "content": BASE_SYSTEM_PROMPT.substitute({
                "python_code": python_code,
                "triton_code": triton_code,
                "max_tokens": cfg["max_tokens"]
            })
        }]
    
    def get_qc_renderer_prompt(
        reasoning_trace
    ):
        return [{
            "role": "user",
            "content": QC_MODEL_SYSTEM_PROMPT.substitute({
                "reasoning_trace": reasoning_trace
            })
        }]
    
    def call_base_model(
        python_code,
        triton_code
    ):
        base_prompt = base_renderer.build_generation_prompt(
            get_base_renderer_prompt(
                python_code=python_code,
                triton_code=triton_code
            )
        )
        base_output = base_sampling_client.sample(
            prompt=base_prompt,
            sampling_params=base_sampling_params,
            num_samples=1
        ).result()
        base_response, base_response_parse_status = base_renderer.parse_response(
            base_output.sequences[0].tokens
        )
        
        if not base_response_parse_status:
            print(f"BASE_NULL: Base model response could not be parsed correctly.")
        
        qc_prompt = qc_renderer.build_generation_prompt(
            get_qc_renderer_prompt(
                reasoning_trace=base_response
            )
        )
        qc_output = qc_sampling_client.sample(
            prompt=qc_prompt,
            sampling_params=qc_sampling_params,
            num_samples=1
        ).result()
        qc_response, qc_response_parse_status = qc_renderer.parse_response(
            qc_output.sequences[0].tokens
        )

        if not qc_response_parse_status:
            print(f"QC_NULL: QC model response could not be parsed correctly.")
        
        try:
            qc_verdict = json.loads(qc_response)
            return {
                "qc_result": qc_verdict["result"],
                "trace": base_response,
                "analysis": qc_verdict["analysis"]
            }
        except json.JSONDecodeError as err:
            return {
                "qc_result": qc_verdict["result"],
                "trace": base_response,
                "analysis": qc_verdict["analysis"]
            }
        
    