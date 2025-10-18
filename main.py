from src.LLMs.client_openai import generate_triton_code_zero_shot
from src.prompts import construct_prompt_zero_shot
from src.utils.common import read_file, clean_markdown_code
import os

if __name__ == "__main__":
    ref_arch_src = read_file("src/examples/19_ReLU.py")
    prompt = construct_prompt_zero_shot(gpu_name="L40S", ref_arch_src=ref_arch_src)
    
    print("Generating optimized Triton code...")
    result = generate_triton_code_zero_shot(prompt, model="gpt-5", max_completion_tokens=8192)
    
    
    output_path = 'output/generated_triton_code.py'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(clean_markdown_code(result))