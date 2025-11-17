import os
import runpod
from dotenv import load_dotenv
import csv
from datetime import datetime
from src.search_config import SearchConfig
from src.LLMs.client_openai import OpenAIClient
from src.LLMs.client_gemini import GeminiClient
from src.search.search_orchestrator import SearchOrchestrator
from src.search.results_manager import ResultsManager
from src.utils.common import read_file

load_dotenv()

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


def create_llm_client(config: SearchConfig):
    """
    Create LLM client based on configuration.
    
    Args:
        config: Search configuration
        
    Returns:
        LLM client instance
    """
    if config.model_provider == "openai":
        return OpenAIClient(model=config.model_name)
    elif config.model_provider == "gemini":
        return GeminiClient(model=config.model_name)
    else:
        raise ValueError(f"Unknown model provider: {config.model_provider}")


def extract_problem_name(file_path):
    """Extract problem name from file path (e.g., '23_Softmax')"""
    basename = os.path.basename(file_path)
    return basename.replace('.py', '')


def get_completed_problems(csv_path):
    """Read CSV and return set of completed problem names"""
    if not os.path.exists(csv_path):
        return set()
    
    completed = set()
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row['Problem'])
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
    return completed


def save_generated_code(problem_name, generated_code, model_name):
    """Save generated code to output/search folder"""
    output_dir = f'output/search/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{problem_name}.py")
    with open(output_path, "w", encoding='utf-8') as f:
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


def append_result_to_csv(csv_path, problem_name, best_kernel=None, error=None):
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
        elif best_kernel:
            result = best_kernel.eval_result
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
        else:
            writer.writerow([
                problem_name,
                'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                'FAILED',
                'No valid kernel found',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])


def run_benchmark():
    """Run search-based benchmarks for all problems and save results to CSV"""
    
    # Configuration
    config = SearchConfig(
        # Search parameters
        num_rounds=3,   # Round 0 (baseline) + Round 1-2 (optimization)
        width_per_round=[3, 3],
        depth_per_round=[1, 1],
        top_k_bank=5,
        
        # LLM parameters
        model_name="gpt-5",
        model_provider="openai",  # Change to "gemini" for Gemini models or "openai" for OpenAI models
        temperature=1.0,
        max_completion_tokens=16384,
        
        # Problem parameters
        gpu_name="A100-80GB",
        problem_name="",  # Will be set per problem
        
        # Evaluation parameters
        num_correct_trials=5,
        num_perf_trials=100,
        evaluation_timeout=600
    )
    
    # Initialize RunPod
    runpod.api_key = os.getenv("RUNPOD_API_KEY")
    endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))
    
    # Create LLM client
    llm_client = create_llm_client(config)
    
    # Set up CSV path
    model_identifier = f"{config.model_provider}-{config.model_name}"
    csv_path = f'results/search_{model_identifier}.csv'
    
    # Check if CSV exists and get completed problems
    completed_problems = get_completed_problems(csv_path)
    
    if not os.path.exists(csv_path):
        initialize_csv(csv_path)
        print(f"Initialized new results CSV at: {csv_path}")
    else:
        print(f"Resuming benchmark - appending to existing CSV: {csv_path}")
        if completed_problems:
            print(f"Found {len(completed_problems)} already completed problem(s)")
    
    print(f"Using GPU: {config.gpu_name}")
    print(f"Model: {model_identifier}")
    print(f"Search config: {config.num_rounds} rounds, width={config.width_per_round}, depth={config.depth_per_round}")
    
    # Filter out already completed problems
    problems_to_run = [p for p in problem_set_paths if extract_problem_name(p) not in completed_problems]
    
    if len(problems_to_run) == 0:
        print(f"\nAll {len(problem_set_paths)} problems already completed!")
        print("="*80)
        return
    
    print(f"Total problems: {len(problem_set_paths)}")
    print(f"Remaining to benchmark: {len(problems_to_run)}")
    if completed_problems:
        print(f"Completed problems: {', '.join(sorted(completed_problems))}")
    print("="*80)
    
    for idx, problem_path in enumerate(problems_to_run, 1):
        problem_name = extract_problem_name(problem_path)
        print(f"\n[{idx}/{len(problems_to_run)}] Processing: {problem_name}")
        print("-"*80)
        
        try:
            print(f"Reading reference architecture from: {problem_path}")
            ref_arch_src = read_file(problem_path)
            
            # Update config with current problem name
            config.problem_name = problem_name
            
            print("Initializing search orchestrator...")
            results_manager = ResultsManager(output_dir=f"output/search_results/{model_identifier}/{problem_name}")
            orchestrator = SearchOrchestrator(
                config=config,
                llm_client=llm_client,
                endpoint=endpoint,
                ref_arch_src=ref_arch_src,
                results_manager=results_manager
            )
            
            print("Running search process...")
            results = orchestrator.run_search()
            
            best_kernel = results['best_kernel']
            
            if best_kernel:
                print("\n" + "="*60)
                print("BENCHMARK RESULTS")
                print("="*60)
                print(f"Problem:     {problem_name}")
                print(f"Kernel ID:   {best_kernel.kernel_id}")
                print(f"Round:       {best_kernel.round_number}")
                
                result = best_kernel.eval_result
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
                print(f"\nOptimization Idea: {best_kernel.metadata.get('idea_text', 'N/A')}")
                print("="*60)
                
                # Save generated code
                code_path = save_generated_code(problem_name, best_kernel.code, model_identifier)
                print(f"Best kernel code saved to: {code_path}")
                
                # Append to CSV
                append_result_to_csv(csv_path, problem_name, best_kernel=best_kernel)
                print(f"Results saved to CSV")
            else:
                print("\nNo valid kernel found during search!")
                append_result_to_csv(csv_path, problem_name, error="No valid kernel found")
            
        except TimeoutError as e:
            error_msg = "Job timed out"
            print(f"ERROR: {error_msg}")
            append_result_to_csv(csv_path, problem_name, error=error_msg)
            
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            append_result_to_csv(csv_path, problem_name, error=error_msg)
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"All results saved to: {csv_path}")
    print(f"Generated codes saved to: output/search/{model_identifier}/")


if __name__ == "__main__":
    run_benchmark()

