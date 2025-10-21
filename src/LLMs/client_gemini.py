import os 
from google import genai
from google.genai import types

def generate_triton_code_zero_shot(
    client: genai.Client,
    prompt: str,
    model: str = "gemini-2.5-pro",
    max_completion_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 40
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=[
                "You are an expert in writing optimized Triton kernels for GPU computing."
            ],
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
            ),
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_completion_tokens,
            temperature=temperature,
        )
    )
    return response.text

if __name__ == "__main__":
    from src.prompts import construct_prompt_zero_shot
    from src.utils.common import read_file, clean_markdown_code
    REPO_TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    ref_arch_src = read_file(os.path.join(REPO_TOP_PATH, "examples", "model_ex_add.py"))
    prompt = construct_prompt_zero_shot(gpu_name="L40S", ref_arch_src=ref_arch_src)
    
    print("Generating optimized Triton code...")

    result = generate_triton_code_zero_shot(client, prompt)

    output_path = os.path.join(REPO_TOP_PATH, "examples", "generated_triton_code.py")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(clean_markdown_code(result))
    
    print(f"\nGenerated code saved to: {output_path}")