from src.LLMs.client_openai import generate_triton_code_zero_shot
from src.prompts import construct_prompt_zero_shot
from src.utils.common import read_file, clean_markdown_code
from src.eval import eval_kernel_against_ref
import os

if __name__ == "__main__":
    ref_arch_src = read_file("src/examples/19_ReLU.py")
    gpu_name = "RTX_4070_Laptop"
    prompt = construct_prompt_zero_shot(gpu_name=gpu_name, ref_arch_src=ref_arch_src)
    
    print("Generating optimized Triton code...")
    result = generate_triton_code_zero_shot(prompt, model="gpt-5", max_completion_tokens=8192)
    
    
    output_path = 'output/generated_triton_code.py'
    generated_code = clean_markdown_code(result)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(generated_code)
    print(f"Generated Triton code saved to {output_path}")
    
    print("\nEvaluating the custom kernel against the original model...")
    eval_result = eval_kernel_against_ref(
        ref_arch_src,
        generated_code,
        verbose=False,
        measure_performance=True,
        num_correct_trials=5,
        num_perf_trials=100,
        backend="triton",
        compile_with_inductor=False # whether to compile the original model with inductor backend
    )
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Compiled:    {eval_result.compiled}")
    print(f"Correctness: {eval_result.correctness} {eval_result.metadata.get('correctness_trials', '')}")
    print(f"\nPerformance Comparison:")
    print(f"  Reference PyTorch: {eval_result.ref_runtime:.4f} ms (avg)")
    print(f"  Custom Triton:     {eval_result.runtime:.4f} ms (avg)")
    print(f"  Speedup:           {eval_result.speedup:.2f}x")
    print("="*60)