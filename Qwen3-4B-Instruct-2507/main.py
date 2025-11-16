import os
import glob
import logging
import dotenv
import sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "tinker"))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from src.utils.common import read_file, clean_markdown_code_qwen

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_PATH = "tinker://c428e4d9-7c11-50b3-9663-d6717065928a:train:0/sampler_weights/final"


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


def save_generated_code(rel_path, generated_code, base_output_dir="output/zeroshot/qwen3_4b_finetuned"):
    """
    Save generated code maintaining the same folder structure.
    rel_path: e.g., 'level1/23_Softmax.py'
    """
    output_path = os.path.join(base_output_dir, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(generated_code)
    
    return output_path


def construct_messages(ref_arch_src: str):
    """
    Construct messages in the format expected by the fine-tuned model.
    Uses the same format as in the training data (load_dataset.ipynb).
    """
    messages = [
        {
            'role': 'system',
            'content': 'You are an expert in optimizing PyTorch code to Triton GPU kernels. Convert the provided PyTorch implementations to efficient Triton code.'
        },
        {
            'role': 'user',
            'content': f'Convert the following PyTorch code to Triton:\n\n```python\n{ref_arch_src}\n```'
        }
    ]
    
    return messages


def main(output_dir="output/zeroshot/qwen3_4b_finetuned"):
    """Load trained model and generate code for all KernelBench problems."""
    logger.info("Starting KernelBench evaluation with fine-tuned Qwen3-4B model")
    
    # Initialize Tinker client
    service_client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY"))
    
    logger.info(f"Loading trained model path: {MODEL_PATH}")
    sampling_client = service_client.create_sampling_client(
        model_path=MODEL_PATH,
        base_model=BASE_MODEL
    )
    
    # Get tokenizer and renderer for the base model
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")
    logger.info("Model loaded successfully!\n")
    
    # Get all problems
    problems = get_all_kernelbench_problems()
    print(f"Found {len(problems)} problems to process")
    print("="*80)

    
    for idx, (full_path, rel_path) in enumerate(problems, 1):
        print(f"\n[{idx}/{len(problems)}] Processing: {rel_path}")
        print("-"*80)
        
        # Check if output file already exists
        output_path = os.path.join(output_dir, rel_path)
        if os.path.exists(output_path):
            print(f"SKIPPING: Output file already exists at {output_path}")
            print("-"*80)
            continue
        
        try:
            print(f"Reading reference architecture from: {full_path}")
            ref_arch_src = read_file(full_path)
            
            print("Constructing messages...")
            messages = construct_messages(ref_arch_src=ref_arch_src)
            
            # Build prompt for generation using renderer
            prompt = renderer.build_generation_prompt(messages, role="assistant")
            
            # Generate code with Tinker sampling client
            print("Generating Triton code with fine-tuned Qwen model...")
            params = tinker.SamplingParams(
                max_tokens=8192,
                temperature=1.0,
                top_p=0.9,
                top_k=40,
                stop=renderer.get_stop_sequences()
            )
            
            future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
            result = future.result()
            
            # Decode and clean the generated text
            generated_text = tokenizer.decode(result.sequences[0].tokens)
            generated_code = clean_markdown_code_qwen(generated_text)
            
            # Save the generated code
            output_path = save_generated_code(rel_path, generated_code, output_dir)
            print(f"Generated code saved to: {output_path}")
            print("Successfully generated code")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            logger.exception(e)
        
        print("-"*80)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"All generated codes saved to: {output_dir}/")


if __name__ == "__main__":
    main()

