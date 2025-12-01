import os
import json
import ast
import logging
import dotenv
import pandas as pd
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_MODEL = "Qwen/Qwen3-8B"
MODEL_PATH = "tinker://72fb9baa-da81-50c7-94ba-9f16b7218cff:train:0/sampler_weights/final"


def main():
    """Load trained model and generate sample output with reasoning traces."""
    logger.info("Starting Kernelbook reasoning sampling script")

    service_client = tinker.ServiceClient(api_key=os.getenv("TINKER_API_KEY"))

    logger.info(f"Loading trained model path: {MODEL_PATH}")
    sampling_client = service_client.create_sampling_client(
        model_path=MODEL_PATH,
        base_model=BASE_MODEL
    )
    
    # Get tokenizer and renderer for the base model
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer_name = "qwen3"  # Use qwen3 renderer for reasoning with <think> tags
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading first example from reasoning dataset...")
    dataset = pd.read_csv("processed_reasoning_traces.csv")
    dataset['messages'] = dataset['messages'].apply(ast.literal_eval)
    first_example = dataset['messages'].iloc[0]
    
    # Extract system + user messages (remove assistant for generation)
    test_messages = [msg for msg in first_example if msg['role'] != 'assistant']

    print("\n" + "="*80)
    print("INPUT PROMPT:")
    print("="*80)
    for msg in test_messages:
        print(f"\n[{msg['role'].upper()}]:")
        content = msg['content']
        if len(content) > 500:
            print(content[:500] + "\n... (truncated)")
        else:
            print(content)
    
    # Build prompt for generation
    prompt = renderer.build_generation_prompt(test_messages, role="assistant")
    
    # Sample from the model
    logger.info("Generating Triton code with trained model...")
    params = tinker.SamplingParams(
        max_tokens=4096,
        temperature=0.0,  # Greedy sampling
        stop=renderer.get_stop_sequences()
    )
    
    future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    result = future.result()
    
    # Decode and display result
    generated_text = tokenizer.decode(result.sequences[0].tokens)
    
    print("\n" + "="*80)
    print("GENERATED TRITON CODE:")
    print("="*80)
    print(generated_text)
    print("="*80)
    
    # Also show the ground truth for comparison
    ground_truth = [msg for msg in first_example if msg['role'] == 'assistant'][0]['content']
    print("\n" + "="*80)
    print("GROUND TRUTH (for comparison):")
    print("="*80)
    # Truncate ground truth if too long (reasoning can be verbose)
    if len(ground_truth) > 2000:
        print(ground_truth[:2000] + "\n... (truncated)")
    else:
        print(ground_truth)
    print("="*80)
    
    logger.info("\nSampling complete!")


if __name__ == "__main__":
    main()

