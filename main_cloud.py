import runpod
from dotenv import load_dotenv
import os
from src.LLMs.client_openai import generate_triton_code_zero_shot
from src.prompts import construct_prompt_zero_shot
from src.utils.common import read_file, clean_markdown_code

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))

# ref_arch_src = read_file("src/examples/19_ReLU.py")
ref_arch_src = read_file("src/examples/12_Gemm_Multiply_LeakyReLU.py")

gpu_name = "A100-80GB"
prompt = construct_prompt_zero_shot(gpu_name=gpu_name, ref_arch_src=ref_arch_src)

print("Generating Triton code...")
result = generate_triton_code_zero_shot(prompt, model="gpt-5", max_completion_tokens=8192)
generated_code = clean_markdown_code(result)

output_path = 'output/generated_triton_code.py'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(generated_code)
print(f"Generated Triton code saved to {output_path}")

print("Evaluating the correctness and performance of the Triton code...")
try:
    result = endpoint.run_sync(
        {
            "ref_arch_src": ref_arch_src, 
            "generated_code": generated_code, 
            "num_correct_trials": 5, 
            "num_perf_trials": 100
        },
        timeout=600,
    )
    
    print(result)
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Hardware:    {result['metadata']['hardware']}")
    print(f"Device:    {result['metadata']['device']}")
    print(f"Compiled:    {result['compiled']}")
    print(f"Correctness: {result['correctness']} {result['metadata'].get('correctness_trials', '')}")
    print(f"\nPerformance Comparison:")
    print(f"  Reference PyTorch: {result['ref_runtime']:.4f} ms (avg)")
    print(f"  Reference PyTorch (compiled): {result['ref_runtime_compiled']:.4f} ms (avg)")
    print(f"  Custom Triton:     {result['runtime']:.4f} ms (avg)")
    print(f"  Speedup:           {result['speedup']:.2f}x")
    print(f"  Speedup vs compiled reference: {result['speedup_vs_compiled']:.2f}x")
    print("="*60)
except TimeoutError:
    print("Job timed out.")