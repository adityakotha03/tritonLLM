import os
import csv
import subprocess
import sys
from datetime import datetime

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
                result.get('hardware', 'N/A'),
                result.get('device', 'N/A'),
                result.get('compiled', 'N/A'),
                result.get('correctness', 'N/A'),
                f"{result.get('ref_runtime', -1):.4f}",
                f"{result.get('ref_runtime_compiled', -1):.4f}",
                f"{result.get('runtime', -1):.4f}",
                f"{result.get('speedup', -1):.2f}",
                f"{result.get('speedup_vs_compiled', -1):.2f}",
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

def run_single_benchmark_subprocess(ref_path, generated_path, problem_name, level):
    """
    Run a single benchmark in a completely isolated subprocess.
    This ensures CUDA context is fresh for each evaluation.
    """
    # Create a temporary Python script to run the evaluation
    script_content = f'''
import sys
import json
from src.eval import eval_kernel_against_ref
from src.utils.common import read_file

try:
    ref_arch_src = read_file("{ref_path}")
    generated_code = read_file("{generated_path}")
    
    result = eval_kernel_against_ref(
        ref_arch_src,
        generated_code,
        verbose=False,
        measure_performance=True,
        num_correct_trials=5,
        num_perf_trials=100,
        backend="triton"
    )
    
    # Convert result to dict and print as JSON
    result_dict = {{
        'hardware': result.metadata.get('hardware', 'N/A'),
        'device': result.metadata.get('device', 'N/A'),
        'compiled': result.compiled,
        'correctness': result.correctness,
        'ref_runtime': result.ref_runtime,
        'ref_runtime_compiled': result.ref_runtime_compiled,
        'runtime': result.runtime,
        'speedup': result.speedup,
        'speedup_vs_compiled': result.speedup_vs_compiled,
    }}
    
    print("__RESULT_START__")
    print(json.dumps(result_dict))
    print("__RESULT_END__")
    
except Exception as e:
    print("__ERROR__", str(e), file=sys.stderr)
    sys.exit(1)
'''
    
    # Write script to temporary file
    temp_script = f'temp_eval_{problem_name}.py'
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        # Run the script in a subprocess
        # Using subprocess ensures complete isolation and fresh CUDA context
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per benchmark
        )
        
        # Clean up temp script
        if os.path.exists(temp_script):
            os.remove(temp_script)
        
        if result.returncode == 0:
            output = result.stdout
            if "__RESULT_START__" in output and "__RESULT_END__" in output:
                start = output.find("__RESULT_START__") + len("__RESULT_START__")
                end = output.find("__RESULT_END__")
                json_str = output[start:end].strip()
                
                import json
                result_dict = json.loads(json_str)
                return result_dict, None
            else:
                return None, "Could not parse result from subprocess"
        else:
            error_msg = result.stderr
            if "__ERROR__" in error_msg:
                error_msg = error_msg.split("__ERROR__")[1].strip()
            return None, error_msg
            
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return None, "Evaluation timeout (>5 minutes)"
    except Exception as e:
        if os.path.exists(temp_script):
            os.remove(temp_script)
        return None, str(e)

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
    print("\nRunning each evaluation in isolated subprocess to prevent CUDA context corruption...")
    print("="*80)
    
    # Process each file
    for idx, file_info in enumerate(all_files, 1):
        problem_name = extract_problem_name(file_info['filename'])
        level = file_info['level']
        
        print(f"\n[{idx}/{len(all_files)}] Processing: {problem_name} ({level})")
        print("-"*80)
        
        # Run in isolated subprocess
        result_dict, error = run_single_benchmark_subprocess(
            file_info['ref_path'],
            file_info['generated_path'],
            problem_name,
            level
        )
        
        if error:
            print(f"ERROR: {error}")
            append_result_to_csv(csv_path, problem_name, level, error=error)
        else:
            print("\n" + "="*60)
            print("BENCHMARK RESULTS")
            print("="*60)
            print(f"Problem:     {problem_name}")
            print(f"Level:       {level}")
            print(f"Hardware:    {result_dict['hardware']}")
            print(f"Device:      {result_dict['device']}")
            print(f"Compiled:    {result_dict['compiled']}")
            print(f"Correctness: {result_dict['correctness']}")
            print(f"\nPerformance Comparison:")
            print(f"  Reference PyTorch:            {result_dict['ref_runtime']:.4f} ms")
            print(f"  Reference PyTorch (compiled): {result_dict['ref_runtime_compiled']:.4f} ms")
            print(f"  Custom Triton:                {result_dict['runtime']:.4f} ms")
            print(f"  Speedup:                      {result_dict['speedup']:.2f}x")
            print(f"  Speedup vs compiled ref:      {result_dict['speedup_vs_compiled']:.2f}x")
            print("="*60)
            
            append_result_to_csv(csv_path, problem_name, level, result=result_dict)
            print(f"Results saved to CSV")
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"All results saved to: {csv_path}")

if __name__ == "__main__":
    run_benchmark()

