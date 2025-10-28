# TritonLLM: LLM-Powered Triton Kernel Optimization

An automated system for generating optimized Triton GPU kernels from PyTorch code using Large Language Models, with comprehensive benchmarking and evaluation.

## Quick Start

### Installation

```bash
# Set up your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```bash
# Run the main script to generate and evaluate Triton kernels
python main.py
```

This will:
1. Read a reference PyTorch model from `src/examples/`
2. Generate an optimized Triton kernel using GPT-5
3. Evaluate correctness and performance
4. Display benchmark results comparing against PyTorch baseline

## Example Output

```
============================================================
BENCHMARK RESULTS
============================================================
Compiled:    True
Correctness: True (5 / 5)

Performance Comparison:
  Reference PyTorch: 0.5123 ms (avg)
  Custom Triton:     0.2290 ms (avg)
  Speedup:           2.24x
============================================================
```

## Project Structure

```
tritonLLM/
├── main.py                          # Main entry point
├── src/
│   ├── LLMs/
│   │   └── client_openai.py        # OpenAI API client
│   ├── examples/                    # Example PyTorch models
│   │   ├── model_ex_add.py
│   │   ├── model_new_ex_add_triton.py
│   │   └── 19_ReLU.py
│   ├── prompts.py                   # Prompt construction with GPU specs
│   ├── gpu_specs.py                 # GPU hardware specifications
│   ├── eval.py                      # Evaluation and benchmarking
│   └── utils/
│       ├── common.py                # Common utilities
│       └── eval_helpers.py          # Evaluation helper functions
├── output/                          # Generated Triton kernels
└── README.md
```

## Configuration

### Supported GPUs

The system includes detailed specifications for:
- NVIDIA RTX 4070 Laptop (Ada Lovelace)
- NVIDIA L40S (Ada Lovelace)
- NVIDIA H100 (Hopper)
- NVIDIA A100 (Ampere)
- NVIDIA L4 (Ada Lovelace)
- NVIDIA T4 (Turing)
- NVIDIA A10G (Ampere)

### Customizing GPU Target

Edit `main.py` to change the target GPU:

```python
gpu_name = "RTX_4070_Laptop"  # Change to your GPU
prompt = construct_prompt_zero_shot(gpu_name=gpu_name, ref_arch_src=ref_arch_src)
```

### Evaluation Parameters

Adjust evaluation settings in `main.py`:

```python
eval_result = eval_kernel_against_ref(
    ref_arch_src,
    generated_code,
    verbose=False,              # Set to True for detailed logs
    measure_performance=True,
    num_correct_trials=5,       # Number of correctness checks
    num_perf_trials=100,        # Number of performance trials
    backend="triton",           # "triton" or "cuda"
)
```

## Prompt Engineering

The system constructs prompts with:
- **Problem Statement**: Task description for kernel optimization
- **GPU Hardware Info**: Architecture, memory, bandwidth, compute specs
- **GPU Architecture Concepts**: Definitions of warps, shared memory, tensor cores, etc.
- **Best Practices**: Triton-specific optimization guidelines
- **Few-Shot Examples** (optional): Example transformations

See `src/prompts.py` for prompt construction details.

## Evaluation Methodology

### Correctness Verification
- Runs multiple trials with randomized inputs
- Compares outputs against PyTorch reference using `torch.allclose`
- Reports pass rate (e.g., "5 / 5" trials passed)

### Performance Benchmarking
- Measures both reference and custom kernel runtimes
- Uses CUDA events for precise GPU timing
- Includes warmup runs to avoid cold-start effects
- Reports mean, std, min, max across multiple trials
- Calculates speedup: `reference_time / custom_time`

**Note**: Parts of the evaluation framework were adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench), an excellent benchmark for evaluating LLMs' ability to generate efficient GPU kernels.

## How It Works

1. **Input**: PyTorch model with standard operators (Conv2d, ReLU, LayerNorm, etc.)
2. **Prompt Construction**: Build context-rich prompt with GPU specs and best practices
3. **LLM Generation**: Query GPT-5 to generate optimized Triton implementation
4. **Code Extraction**: Parse and clean generated code from markdown
5. **Compilation**: Compile Triton kernel with `@triton.jit` decorator
6. **Evaluation**:
   - Load both reference PyTorch and custom Triton models
   - Verify correctness across multiple random inputs
   - Benchmark performance with CUDA timing events
   - Calculate speedup and statistics

## Advanced Usage

### Custom Examples

Add your own PyTorch models to `src/examples/`:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model definition
    
    def forward(self, x):
        # Your forward pass
        return output

def get_init_inputs():
    # Return inputs for model initialization
    return []

def get_inputs():
    # Return inputs for forward pass
    return [torch.randn(size, device='cuda')]
```

### Adding New GPUs

Edit `src/gpu_specs.py` to add specifications:

```python
GPU_SPEC_INFO = {
    "YOUR_GPU": {
        "GPU Architecture": "Architecture Name",
        "GPU Memory": "Memory Size",
        "Memory Bandwidth": "Bandwidth",
        "FP32 TFLOPS": "Compute Power",
        # ... other specs
    }
}
```

## Search-Based Optimization (NEW!)

We now support **search-based kernel optimization** inspired by [Stanford's CRFM blog post](https://crfm.stanford.edu/2025/05/28/fast-kernels.html). This method uses natural language optimization ideas with parallel branching to explore diverse optimization strategies.

### Key Features
- **Natural Language Ideas**: Generates optimization strategies before implementing them
- **Branching Search**: Multiple ideas per round × multiple implementations per idea
- **Model Agnostic**: Supports both OpenAI and Gemini
- **Complete Tracking**: Saves all generated code, ideas, and results to JSON

### Quick Start

```bash
# Run search-based optimization
python main_search.py
```

## Roadmap

- [x] Zero-shot Triton kernel generation
- [x] GPU-aware prompting with hardware specs
- [x] Automated correctness and performance evaluation
- [x] Reference baseline benchmarking
- [x] Added Support for Cloud eval with Runpod serverless
- [x] **Search-based optimization with branching and natural language ideas**
- [ ] Tree-of-thought reasoning for multi-step optimization
- [ ] Multi-turn iterative refinement
- [ ] Support for more LLM providers (Anthropic, DeepSeek, etc.)

## Acknowledgments

- **Evaluation Framework**: Parts of the evaluation methodology were adapted from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) by Scaling Intelligence at Stanford
- **GPU Specifications**: Adapted from KernelBench's GPU specs database
- **Prompt Design Principles**: Inspired by KernelBench's approach to LLM evaluation for kernel generation

---

**Note**: This is a research project. Generated kernels should be thoroughly tested before production use.
