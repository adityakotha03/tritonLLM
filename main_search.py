import os
import runpod
from dotenv import load_dotenv
from src.search_config import SearchConfig
from src.LLMs.client_openai import OpenAIClient
from src.LLMs.client_gemini import GeminiClient
from src.search.search_orchestrator import SearchOrchestrator
from src.search.results_manager import ResultsManager
from src.utils.common import read_file

load_dotenv()


def create_llm_client(config: SearchConfig):
    """
    Create LLM client based on configuration.
    
    Args:
        config: Search configuration
        
    Returns:
        LLM client instance
    """
    if config.model_provider == "openai":
        return OpenAIClient(model=config.model_name)
    elif config.model_provider == "gemini":
        return GeminiClient(model=config.model_name)
    else:
        raise ValueError(f"Unknown model provider: {config.model_provider}")


def main():
    """Main function to run the search."""

    config = SearchConfig(
        # Search parameters
        num_rounds=4,   # Round 0 (baseline) + Round 1-3 (optimization), baseline is just one idea and one implementation
        width_per_round=[3, 3, 3],
        depth_per_round=[1, 1, 1],
        top_k_bank=5,
        
        # LLM parameters
        model_name="gpt-5",
        model_provider="openai",  # "openai" or "gemini"
        temperature=1.0,
        max_completion_tokens=16384,
        
        # Problem parameters
        gpu_name="A100-80GB",
        problem_name="conv2d_relu_biasadd",
        
        # Evaluation parameters
        num_correct_trials=5,
        num_perf_trials=100,
        evaluation_timeout=600
    )
    

    ref_arch_path = "KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py"
    
    ref_arch_src = read_file(ref_arch_path)
    runpod.api_key = os.getenv("RUNPOD_API_KEY")
    endpoint = runpod.Endpoint(os.getenv("ENDPOINT_ID"))
    

    llm_client = create_llm_client(config)
    results_manager = ResultsManager(output_dir="output/search_results")
    orchestrator = SearchOrchestrator(
        config=config,
        llm_client=llm_client,
        endpoint=endpoint,
        ref_arch_src=ref_arch_src,
        results_manager=results_manager
    )
    results = orchestrator.run_search()
    

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    best_kernel = results['best_kernel']
    if best_kernel:
        print(f"\nBest Kernel:")
        print(f"  Kernel ID: {best_kernel.kernel_id}")
        print(f"  Round: {best_kernel.round_number}")
        print(f"  Speedup: {best_kernel.get_speedup():.2f}x")
        print(f"  Runtime: {best_kernel.get_runtime():.4f} ms")
        print(f"  Optimization Idea: {best_kernel.metadata.get('idea_text', 'N/A')}")
    else:
        print("\nNo valid kernel found!")
    
    print(f"\nResults saved to: {results['results_path']}")
    print("\n" + "=" * 80)
    print("Search completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

