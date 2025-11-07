import os
import csv
from datetime import datetime
from src.eval import eval_kernel_against_ref
from src.utils.common import read_file

def eval_kernel_against_ref_qwen3_30b(ref_arch_src, generated_code, num_correct_trials=5, num_perf_trials=100, backend="triton"):
    return eval_kernel_against_ref(
        ref_arch_src,
        generated_code,
        verbose=False,
        measure_performance=True,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        backend=backend
    )

def extract_problem_name(file_path):
    """Extract problem name from file path (e.g., '23_Softmax')"""
    basename = os.path.basename(file_path)
    return basename.replace('.py', '')

def initialize_csv(csv_path):
    """Initialize CSV file with headers"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Problem',
            'Level',
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

def append_result_to_csv(csv_path, problem_name, level, result=None, error=None):
    """Append a single result to the CSV file"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if error:
            writer.writerow([
                problem_name,
                level,
                'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                'FAILED',
                str(error),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
        else:
            writer.writerow([
                problem_name,
                level,
                result.metadata.get('hardware', 'N/A'),
                result.metadata.get('device', 'N/A'),
                result.compiled,
                result.correctness,
                f"{result.ref_runtime:.4f}",
                f"{result.ref_runtime_compiled:.4f}",
                f"{result.runtime:.4f}",
                f"{result.speedup:.2f}",
                f"{result.speedup_vs_compiled:.2f}",
                'SUCCESS',
                '',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])

def get_matching_files_for_level(level):
    """Get all matching files between KernelBench and generated code for a specific level"""
    kernelbench_dir = f"KernelBench/{level}"
    generated_dir = f"output/zeroshot/qwen3_30b/{level}"
    
    if not os.path.exists(kernelbench_dir):
        print(f"Warning: KernelBench directory not found: {kernelbench_dir}")
        return []
    
    if not os.path.exists(generated_dir):
        print(f"Warning: Generated code directory not found: {generated_dir}")
        return []
    
    kernelbench_files = sorted([f for f in os.listdir(kernelbench_dir) if f.endswith('.py')])
    
    generated_files = sorted([f for f in os.listdir(generated_dir) if f.endswith('.py')])
    
    # Find matching files
    matching_files = []
    for filename in kernelbench_files:
        if filename in generated_files:
            matching_files.append({
                'filename': filename,
                'ref_path': os.path.join(kernelbench_dir, filename),
                'generated_path': os.path.join(generated_dir, filename),
                'level': level
            })
    
    return matching_files

def run_benchmark():
    """Run benchmarks for all levels and save results to CSV"""
    csv_path = 'results/zeroshot_qwen3_30b.csv'
    
    # Initialize CSV
    initialize_csv(csv_path)
    print(f"Initialized results CSV at: {csv_path}")
    print("="*80)
    
    # Get all matching files across all levels
    all_files = []
    levels = ['level1', 'level2', 'level3']
    
    for level in levels:
        level_files = get_matching_files_for_level(level)
        all_files.extend(level_files)
        print(f"Found {len(level_files)} matching files in {level}")
    
    print(f"\nTotal problems to benchmark: {len(all_files)}")
    print("="*80)
    
    # Process each file
    for idx, file_info in enumerate(all_files, 1):
        problem_name = extract_problem_name(file_info['filename'])
        level = file_info['level']
        
        print(f"\n[{idx}/{len(all_files)}] Processing: {problem_name} ({level})")
        print("-"*80)
        
        try:
            print(f"Reading reference from: {file_info['ref_path']}")
            ref_arch_src = read_file(file_info['ref_path'])
            
            print(f"Reading generated code from: {file_info['generated_path']}")
            generated_code = read_file(file_info['generated_path'])

            print("Evaluating correctness and performance on Qwen3-30B...")
            result = eval_kernel_against_ref_qwen3_30b(
                ref_arch_src=ref_arch_src,
                generated_code=generated_code,
                num_correct_trials=5,
                num_perf_trials=100,
                backend="triton"
            )

            print("\n" + "="*60)
            print("BENCHMARK RESULTS")
            print("="*60)
            print(f"Problem:     {problem_name}")
            print(f"Level:       {level}")
            print(f"Hardware:    {result.metadata['hardware']}")
            print(f"Device:      {result.metadata['device']}")
            print(f"Compiled:    {result.compiled}")
            print(f"Correctness: {result.correctness}")
            print(f"\nPerformance Comparison:")
            print(f"  Reference PyTorch:            {result.ref_runtime:.4f} ms")
            print(f"  Reference PyTorch (compiled): {result.ref_runtime_compiled:.4f} ms")
            print(f"  Custom Triton:                {result.runtime:.4f} ms")
            print(f"  Speedup:                      {result.speedup:.2f}x")
            print(f"  Speedup vs compiled ref:      {result.speedup_vs_compiled:.2f}x")
            print("="*60)
            
            append_result_to_csv(csv_path, problem_name, level, result=result)
            print(f"Results saved to CSV")
            
        except Exception as e:
            error_msg = str(e)
            print(f"ERROR: {error_msg}")
            append_result_to_csv(csv_path, problem_name, level, error=error_msg)
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"All results saved to: {csv_path}")

if __name__ == "__main__":
    run_benchmark()