from vllm import LLM, SamplingParams
from src.prompts import construct_prompt_zero_shot
from src.utils.common import read_file, clean_markdown_code
import os
import glob

def get_all_kernelbench_problems():
    """
    Recursively find all .py files in KernelBench directory.
    Returns list of tuples: (full_path, relative_path_from_kernelbench)
    """
    problems = []
    kernelbench_dir = "KernelBench"
    pattern = os.path.join(kernelbench_dir, "**", "*.py")
    for full_path in glob.glob(pattern, recursive=True):
        rel_path = os.path.relpath(full_path, kernelbench_dir)
        problems.append((full_path, rel_path))
    
    return sorted(problems)

def save_generated_code(rel_path, generated_code, base_output_dir="outputs/qwen3_30b"):
    """
    Save generated code maintaining the same folder structure.
    rel_path: e.g., 'level1/23_Softmax.py'
    """
    output_path = os.path.join(base_output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(generated_code)
    
    return output_path

def main():
    # Initialize the vLLM model
    print("Loading Qwen/Qwen3-30B-A3B-Instruct-2507 model...")
    llm = LLM("Qwen/Qwen3-30B-A3B-Instruct-2507")
    sampling_params = SamplingParams(
        temperature=1, 
        top_p=0.9, 
        top_k=40, 
        max_tokens=16384
    )
    print("Model loaded successfully!\n")
    
    # Get all problems
    problems = get_all_kernelbench_problems()
    print(f"Found {len(problems)} problems to process")
    print("="*80)
    
    gpu_name = "A100-80GB"
    
    for idx, (full_path, rel_path) in enumerate(problems, 1):
        print(f"\n[{idx}/{len(problems)}] Processing: {rel_path}")
        print("-"*80)
        
        try:
            print(f"Reading reference architecture from: {full_path}")
            ref_arch_src = read_file(full_path)
            
            print("Constructing prompt...")
            prompt = construct_prompt_zero_shot(gpu_name=gpu_name, ref_arch_src=ref_arch_src)
            
            # Generate code with vLLM
            print("Generating Triton code with Qwen model...")
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            generated_code = clean_markdown_code(generated_text)
            output_path = save_generated_code(rel_path, generated_code)
            print(f"Generated code saved to: {output_path}")
            print("Successfully generated code")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"All generated codes saved to: outputs/qwen_30b/")

if __name__ == "__main__":
    main()