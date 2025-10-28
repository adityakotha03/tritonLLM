import runpod
from dotenv import load_dotenv
import os
import csv
from datetime import datetime
from src.LLMs.client_openai import generate_triton_code_zero_shot
from src.LLMs.client_gemini import generate_triton_code_zero_shot
from src.prompts import construct_prompt_zero_shot
from src.utils.common import read_file, clean_markdown_code
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))

problem_set_paths = [
    # Level 1 kernels
    "KernelBench/level1/6_Matmul_with_large_K_dimension_.py",
    "KernelBench/level1/23_Softmax.py",
    "KernelBench/level1/40_LayerNorm.py",
    "KernelBench/level1/50_conv_standard_2D__square_input__square_kernel.py",
    "KernelBench/level1/90_cumprod.py",
    # Level 2 kernels
    "KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py",
    "KernelBench/level2/3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py",
    "KernelBench/level2/35_Conv2d_Subtract_HardSwish_MaxPool_Mish.py",
    "KernelBench/level2/66_Matmul_Dropout_Softmax.py",
    "KernelBench/level2/97_Matmul_BatchNorm_BiasAdd_Divide_Swish.py",
]

def extract_problem_name(file_path):
    """Extract problem name from file path (e.g., '23_Softmax')"""
    basename = os.path.basename(file_path)
    return basename.replace('.py', '')

def save_generated_code(problem_name, generated_code):
    """Save generated code to output/zeroshot folder"""
    output_dir = 'output/zeroshot/gemini-2.5-pro'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{problem_name}.py")
    with open(output_path, "w") as f:
        f.write(generated_code)
    return output_path

def initialize_csv(csv_path):
    """Initialize CSV file with headers"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Problem',
            'Hardware',
            'Device',
            'Compiled',
            'Correctness',
            'Ref_PyTorch_Runtime_ms',
            'Ref_PyTorch_Compiled_Runtime_ms',
            'Triton_Runtime_ms',
            'Speedup',
            'Speedup_vs_Compiled',
            'Status',
            'Error_Message',
            'Timestamp'
        ])

def append_result_to_csv(csv_path, problem_name, result=None, error=None):
    """Append a single result to the CSV file"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if error:
            writer.writerow([
                problem_name,
                'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                'FAILED',
                str(error),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
        else:
            writer.writerow([
                problem_name,
                result.get('metadata', {}).get('hardware', 'N/A'),
                result.get('metadata', {}).get('device', 'N/A'),
                result.get('compiled', 'N/A'),
                result.get('correctness', 'N/A'),
                f"{result.get('ref_runtime', 0):.4f}",
                f"{result.get('ref_runtime_compiled', 0):.4f}",
                f"{result.get('runtime', 0):.4f}",
                f"{result.get('speedup', 0):.2f}",
                f"{result.get('speedup_vs_compiled', 0):.2f}",
                'SUCCESS',
                '',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])

def run_benchmark():
    """Run benchmarks for all problems and save results to CSV"""
    gpu_name = "A100-80GB"
    csv_path = 'results/zeroshot_gemini-2.5-pro.csv'

    initialize_csv(csv_path)
    print(f"Initialized results CSV at: {csv_path}")
    print(f"Using GPU: {gpu_name}")
    print(f"Total problems to benchmark: {len(problem_set_paths)}")
    print("="*80)
    
    for idx, problem_path in enumerate(problem_set_paths, 1):
        problem_name = extract_problem_name(problem_path)
        print(f"\n[{idx}/{len(problem_set_paths)}] Processing: {problem_name}")
        print("-"*80)
        
        try:
            print(f"Reading reference architecture from: {problem_path}")
            ref_arch_src = read_file(problem_path)
            
            print("Constructing prompt...")
            prompt = construct_prompt_zero_shot(gpu_name=gpu_name, ref_arch_src=ref_arch_src)
            
            print("Generating Triton code with LLM...")
            # llm_result = generate_triton_code_zero_shot(prompt, model="gpt-5", max_completion_tokens=16384)
            llm_result = generate_triton_code_zero_shot(client, prompt, model="gemini-2.5-pro", max_completion_tokens=16384)
            generated_code = clean_markdown_code(llm_result)
            
            code_path = save_generated_code(problem_name, generated_code)
            print(f"Generated code saved to: {code_path}")
            
            print("Evaluating correctness and performance on RunPod GPU...")
            result = endpoint.run_sync(
                {
                    "ref_arch_src": ref_arch_src, 
                    "generated_code": generated_code, 
                    "num_correct_trials": 5, 
                    "num_perf_trials": 100
                },
                timeout=120,
            )
            print(result)
            
            print("\n" + "="*60)
            print("BENCHMARK RESULTS")
            print("="*60)
            print(f"Problem:     {problem_name}")
            print(f"Hardware:    {result['metadata']['hardware']}")
            print(f"Device:      {result['metadata']['device']}")
            print(f"Compiled:    {result['compiled']}")
            print(f"Correctness: {result['correctness']}")
            print(f"\nPerformance Comparison:")
            print(f"  Reference PyTorch:            {result['ref_runtime']:.4f} ms")
            print(f"  Reference PyTorch (compiled): {result['ref_runtime_compiled']:.4f} ms")
            print(f"  Custom Triton:                {result['runtime']:.4f} ms")
            print(f"  Speedup:                      {result['speedup']:.2f}x")
            print(f"  Speedup vs compiled ref:      {result['speedup_vs_compiled']:.2f}x")
            print("="*60)
            
            append_result_to_csv(csv_path, problem_name, result=result)
            print(f"Results saved to CSV")
            
        except TimeoutError as e:
            error_msg = "Job timed out"
            print(f"ERROR: {error_msg}")
            append_result_to_csv(csv_path, problem_name, error=error_msg)
            
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            append_result_to_csv(csv_path, problem_name, error=error_msg)
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"All results saved to: {csv_path}")
    print(f"Generated codes saved to: output/zeroshot/")

if __name__ == "__main__":
    run_benchmark()