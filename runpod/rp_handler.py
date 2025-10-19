import runpod
from eval.eval import eval_kernel_against_ref

def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata
        
    Returns:
        dict: Evaluation results with compilation, correctness, and performance metrics
    """
    print(f"Worker Start")
    input_data = event.get('input', event)
    
    ref_arch_src = input_data['ref_arch_src']
    generated_code = input_data['generated_code']
    num_correct_trials = input_data['num_correct_trials']
    num_perf_trials = input_data['num_perf_trials']
    compile_with_inductor = True if input_data['compile_with_inductor'] == 1 else False
    
    eval_result = eval_kernel_against_ref(
        ref_arch_src,
        generated_code,
        verbose=False,
        measure_performance=True,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        backend="triton",
        compile_with_inductor=compile_with_inductor
    )
    
    print(eval_result)
    # Convert Pydantic model to dict for JSON serialization
    return eval_result.model_dump() if eval_result else {"error": "Evaluation returned None"}

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })