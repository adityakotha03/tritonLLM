import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_triton_code_zero_shot(
    prompt: str,
    model: str = "gpt-5",
    max_completion_tokens: int = 8192,
    temperature: float = 1.0
) -> str:
    """
    Generate Triton kernel code using OpenAI chat completion API.
    
    Args:
        prompt: The input prompt describing the optimization task
        model: OpenAI model name (default: gpt-5)
        max_completion_tokens: Maximum tokens in response (default: 4096)
        temperature: Sampling temperature (default: 1.0)
    
    Returns:
        Generated code as string
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in writing optimized Triton kernels for GPU computing."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=max_completion_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content


if __name__ == "__main__":
    from src.prompts import construct_prompt_zero_shot
    from src.utils.common import read_file, clean_markdown_code
    REPO_TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

    ref_arch_src = read_file(os.path.join(REPO_TOP_PATH, "examples", "model_ex_add.py"))
    prompt = construct_prompt_zero_shot(gpu_name="L40S", ref_arch_src=ref_arch_src)
    
    print("Generating optimized Triton code...")
    
    result = generate_triton_code_zero_shot(prompt, model="gpt-5", max_completion_tokens=8192)
    
    output_path = os.path.join(REPO_TOP_PATH, "examples", "generated_triton_code.py")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(clean_markdown_code(result))
    
    print(f"\nGenerated code saved to: {output_path}")