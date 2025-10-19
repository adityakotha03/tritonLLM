import runpod
from dotenv import load_dotenv
import os
from src.LLMs.client_openai import generate_triton_code_zero_shot
from src.prompts import construct_prompt_zero_shot
from src.utils.common import read_file, clean_markdown_code

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))

ref_arch_src = read_file("src/examples/19_ReLU.py")

gpu_name = "A100-80GB"
prompt = construct_prompt_zero_shot(gpu_name=gpu_name, ref_arch_src=ref_arch_src)

result = generate_triton_code_zero_shot(prompt, model="gpt-5", max_completion_tokens=8192)
generated_code = clean_markdown_code(result)

try:
    run_request = endpoint.run_sync(
        {"ref_arch_src": ref_arch_src, "generated_code": generated_code, "num_correct_trials": 5, "num_perf_trials": 100, "compile_with_inductor": 0},
        timeout=600,  # Client timeout in seconds
    )
    print(run_request)
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Compiled:    {run_request.compiled}")
    print(f"Correctness: {run_request.correctness} {run_request.metadata.get('correctness_trials', '')}")
    print(f"\nPerformance Comparison:")
    print(f"  Reference PyTorch: {run_request.ref_runtime:.4f} ms (avg)")
    print(f"  Custom Triton:     {run_request.runtime:.4f} ms (avg)")
    print(f"  Speedup:           {run_request.speedup:.2f}x")
    print("="*60)
except TimeoutError:
    print("Job timed out.")