import os
from dotenv import load_dotenv
from openai import OpenAI
from src.LLMs.base_client import BaseLLMClient

load_dotenv()


class OpenAIClient(BaseLLMClient):
    """OpenAI client implementing the BaseLLMClient interface."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Default model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def generate_text(
        self,
        prompt: str,
        max_completion_tokens: int = 8192,
        temperature: float = 1.0,
        system_message: str = "You are an expert in writing optimized Triton kernels for GPU computing.",
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI chat completion API.
        
        Args:
            prompt: Input prompt
            max_completion_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_message: System message for the chat
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Generated text as string
        """
        model = kwargs.pop("model", self.model)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content


# Backward compatibility: keep the original function
_default_client = None

def get_default_client():
    """Get or create the default OpenAI client."""
    global _default_client
    if _default_client is None:
        _default_client = OpenAIClient()
    return _default_client


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
    client = get_default_client()
    return client.generate_text(
        prompt=prompt,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        model=model
    )


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