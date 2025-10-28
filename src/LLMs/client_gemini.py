import os 
from google import genai
from google.genai import types
from src.LLMs.base_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Gemini client implementing the BaseLLMClient interface."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-pro"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Default model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
    
    def generate_text(
        self,
        prompt: str,
        max_completion_tokens: int = 8192,
        temperature: float = 1.0,
        system_message: str = "You are an expert in writing optimized Triton kernels for GPU computing.",
        top_p: float = 0.9,
        top_k: int = 40,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: Input prompt
            max_completion_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_message: System instruction for the model
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Generated text as string
        """
        model = kwargs.pop("model", self.model)
        
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=[system_message],
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


# Backward compatibility: keep the original function
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