"""
**This file is modified from KernelBench**
Construct Prompt

# Design principles: 
# - To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
# - However, we do not do extensive prompt engineering or few-shot example in the LLM to steer behaviour. 
"""

import os
from src.utils.common import read_file

REPO_TOP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROBLEM_STATEMENT = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. \n
You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

def construct_prompt_zero_shot(gpu_name: str, ref_arch_src: str) -> str:
    """
    Construct a zero-shot prompt with GPU hardware information and best practices if provided.
    
    Args:
        gpu_name: Name of the GPU (e.g., "H100", "A100", "L40S")
        ref_arch_src: Source code of the reference architecture to optimize
    
    Returns:
        Complete prompt string with hardware info and task instructions
    """
    
    prompt = PROBLEM_STATEMENT
    
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/examples/model_ex_add.py")
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/examples/model_new_ex_add_triton.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    
    if gpu_name is not None:
        gpu_specs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_specs.py")
        
        if not os.path.exists(gpu_specs_path):
            raise FileNotFoundError(f"GPU specs file not found: {gpu_specs_path}")
        
        # Read and execute gpu_specs.py to get the dictionaries
        gpu_spec_info_src = read_file(gpu_specs_path)
        local_dict = {}
        exec(gpu_spec_info_src, {}, local_dict)
        
        GPU_SPEC_INFO = local_dict.get("GPU_SPEC_INFO")
        GPU_DEFINITIONS = local_dict.get("GPU_DEFINITIONS")
        GPU_BEST_PRACTICES = local_dict.get("GPU_BEST_PRACTICES")
        
        if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
            raise ValueError(
                "GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_specs.py"
            )
        
        if gpu_name not in GPU_SPEC_INFO:
            raise ValueError(
                f"GPU name '{gpu_name}' not found in GPU_SPEC_INFO. Available GPUs: {list(GPU_SPEC_INFO.keys())}"
            )
    
        # Add hardware information
        curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]
        gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
        
        prompt += f"""
Here is some information about the underlying hardware that you should keep in mind.

The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

    """
        
        # Add GPU specifications
        for key, value in curr_gpu_spec_info.items():
            if key == "GPU Architecture":
                continue
            prompt += f"- We have {value} of {key}.\n"
        
        # Add GPU concept definitions
        prompt += """

Here are some concepts about the GPU architecture that could be helpful:

    """
        for key, value in GPU_DEFINITIONS.items():
            prompt += f"- {key}: {value}\n"
        
        # Add best practices
        prompt += """

Here are some best practices for writing Triton kernels on GPU:

    """
        for best_practice in GPU_BEST_PRACTICES:
            prompt += f"- {best_practice}\n"


    if example_arch != "" and example_new_arch != "":
        prompt += f"""
Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: \n
``` \n
{example_arch}
``` \n
The example new arch with custom Triton kernels looks like this: \n
```
{example_new_arch}
``` \n
        """
    
    prompt += f"""
You are given the following architecture:

```python
{ref_arch_src}
```
"""
    prompt += PROBLEM_INSTRUCTION
    
    return prompt



def construct_idea_generation_prompt(
    gpu_name: str,
    ref_arch_src: str,
    current_best_kernels: list,
    previous_ideas: list,
    round_number: int,
    num_ideas: int = 3
) -> str:
    """
    Construct a prompt for generating optimization ideas.
    
    Args:
        gpu_name: Name of the GPU
        ref_arch_src: Source code of the reference architecture
        current_best_kernels: List of current best kernels with their performance
        previous_ideas: List of previously attempted ideas
        round_number: Current round number
        num_ideas: Number of ideas to generate
    
    Returns:
        Prompt string for idea generation
    """
    prompt = f"""You are an expert in GPU kernel optimization. Your task is to generate {num_ideas} diverse and creative optimization ideas for improving Triton kernel performance.

The target GPU is NVIDIA {gpu_name}.

"""
    
    # Add GPU specs if available
    if gpu_name is not None:
        gpu_specs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_specs.py")
        if os.path.exists(gpu_specs_path):
            gpu_spec_info_src = read_file(gpu_specs_path)
            local_dict = {}
            exec(gpu_spec_info_src, {}, local_dict)
            GPU_SPEC_INFO = local_dict.get("GPU_SPEC_INFO", {})
            
            if gpu_name in GPU_SPEC_INFO:
                curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]
                prompt += f"GPU Architecture: {curr_gpu_spec_info.get('GPU Architecture', 'Unknown')}\n"
                for key, value in curr_gpu_spec_info.items():
                    if key != "GPU Architecture":
                        prompt += f"- {key}: {value}\n"
                prompt += "\n"
    
    prompt += f"""This is ROUND {round_number} of optimization.

Here is the reference PyTorch code we are trying to optimize:

```python
{ref_arch_src}
```

"""
    
    if current_best_kernels:
        prompt += f"""Current best kernels and their performance:

"""
        for i, kernel_info in enumerate(current_best_kernels[:5], 1):
            speedup = kernel_info.get('speedup', 'N/A')
            runtime = kernel_info.get('runtime', 'N/A')
            idea = kernel_info.get('idea', 'Initial implementation')
            prompt += f"""{i}. Speedup: {speedup}x | Runtime: {runtime}ms
   Optimization: {idea}

"""
    
    if previous_ideas:
        prompt += f"""Previously attempted optimization ideas (avoid repeating these):

"""
        for i, idea in enumerate(previous_ideas[-10:], 1):  # Last 10 ideas
            prompt += f"{i}. {idea}\n"
        prompt += "\n"
    
    prompt += f"""Generate {num_ideas} NEW and DIVERSE optimization ideas that have NOT been tried before. Focus on different optimization categories:

1. Memory Access Optimization (coalescing, banking, prefetching)
2. Asynchronous Operations & Latency Hiding (async copies, double buffering)
3. Data Type & Precision Optimization (FP16, BF16, mixed precision)
4. Compute & Instruction Optimization (FMA, special instructions)
5. Parallelism & Occupancy Enhancement (block size, grid size, warps)
6. Control Flow & Loop Optimization (unrolling, pipeline)

For each idea, provide:
- A clear, concise description of the optimization strategy
- Why it might improve performance on this specific hardware
- What specific aspect it targets (memory, compute, parallelism, etc.)

Format your response as a numbered list:

1. [Optimization Idea 1]
2. [Optimization Idea 2]
3. [Optimization Idea 3]

Be specific and concrete in your ideas. Avoid vague suggestions.
"""
    
    return prompt


def construct_implementation_prompt(
    gpu_name: str,
    ref_arch_src: str,
    optimization_idea: str,
    round_number: int
) -> str:
    """
    Construct a prompt for implementing a specific optimization idea.
    
    Args:
        gpu_name: Name of the GPU
        ref_arch_src: Source code of the reference architecture
        optimization_idea: The optimization idea to implement
        round_number: Current round number
    
    Returns:
        Prompt string for code implementation
    """
    
    prompt = PROBLEM_STATEMENT
    
    # Add example if available
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/examples/model_ex_add.py")
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/examples/model_new_ex_add_triton.py"
    )

    if os.path.exists(example_arch_path) and os.path.exists(example_new_arch_path):
        example_arch = read_file(example_arch_path)
        example_new_arch = read_file(example_new_arch_path)
        
        prompt += f"""
Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: \n
``` \n
{example_arch}
``` \n
The example new arch with custom Triton kernels looks like this: \n
```
{example_new_arch}
``` \n
        """
    
    if gpu_name is not None:
        gpu_specs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_specs.py")
        
        if os.path.exists(gpu_specs_path):
            gpu_spec_info_src = read_file(gpu_specs_path)
            local_dict = {}
            exec(gpu_spec_info_src, {}, local_dict)
            
            GPU_SPEC_INFO = local_dict.get("GPU_SPEC_INFO")
            GPU_DEFINITIONS = local_dict.get("GPU_DEFINITIONS")
            GPU_BEST_PRACTICES = local_dict.get("GPU_BEST_PRACTICES")
            
            if gpu_name in GPU_SPEC_INFO:
                curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]
                gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
                
                prompt += f"""
Here is some information about the underlying hardware that you should keep in mind.

The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.

    """
                
                for key, value in curr_gpu_spec_info.items():
                    if key == "GPU Architecture":
                        continue
                    prompt += f"- We have {value} of {key}.\n"
                
                prompt += """

Here are some concepts about the GPU architecture that could be helpful:

    """
                for key, value in GPU_DEFINITIONS.items():
                    prompt += f"- {key}: {value}\n"
                
                prompt += """

Here are some best practices for writing Triton kernels on GPU:

    """
                for best_practice in GPU_BEST_PRACTICES:
                    prompt += f"- {best_practice}\n"
    
    prompt += f"""

This is ROUND {round_number} of optimization.

OPTIMIZATION IDEA TO IMPLEMENT:
{optimization_idea}

You are given the following reference architecture:

```python
{ref_arch_src}
```

YOUR TASK:
Implement the optimization idea described above by creating a ModelNew class with custom Triton kernels.
- Focus specifically on implementing the suggested optimization strategy
- Generate REAL, FUNCTIONAL code that compiles and runs correctly
- Name your optimized architecture ModelNew
- Output ONLY the new model code in a code block
- NO pseudocode, NO testing code, NO explanations outside the code block

Generate the optimized code now:
"""
    
    return prompt


if __name__ == "__main__":
    gpu = "L40S"
    ref_arch_src = read_file(os.path.join(REPO_TOP_PATH, "src", "examples", "19_ReLU.py"))
    prompt = construct_prompt_zero_shot(gpu, ref_arch_src)
    
    output_path = os.path.join(REPO_TOP_PATH, "src", "examples", f"prompt_{gpu}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(prompt)