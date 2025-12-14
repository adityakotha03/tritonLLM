# Using LLMs to Generate High-Performance Triton Kernels for PyTorch Workloads

GPU resource availability, as it stands, serves as the major bottleneck for many artificial intelligence (AI) applications that are currently trying to scale. As the demand for these resources grows, there is a need for more optimal GPU kernels, but writing them requires high domain expertise. Hence, we try to tackle the question: **"What is the most reliable way to generate high-performant Triton kernels using LLMs?"**

To answer this, we explore 3 different approaches:

1. **Zero-shot kernel generation** with closed source LLMs
2. **Tree of thought search** with closed source LLMs
3. **LoRA fine-tuning** with open source LLMs


## How to Use This Codebase

### Environment Setup

Create a `.env` file in the root directory with the following API keys:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `RUNPOD_API_KEY`
- `ENDPOINT_ID`
- `TINKER_API_KEY`

**Note:** `RUNPOD_API_KEY` and `ENDPOINT_ID` can be created by deploying the code in the `runpod/` folder to RunPod as a serverless function.

### Running Evaluations

#### 1. Zero-Shot Evaluation
Run `benchmark_zeroshot.py` in the root directory. Swap the `client` variable to switch between Gemini and OpenAI:

```bash
python benchmark_zeroshot.py
```

#### 2. Tree of Thought Search
Run `benchmark_search.py` in the root directory to automatically execute the tree of thought search and generate results:

```bash
python benchmark_search.py
```

#### 3. Open Source LLM Code Generation
Each open source model has its own folder. To generate code:
1. Navigate to the model's directory
2. Run `main_base.py` to generate code for each problem in KernelBench
3. Run `eval_isolated.py` to evaluate results on a RunPod instance (A100 SXM with 80GB VRAM)

### LoRA Fine-Tuning

Navigate to the `tinker/` folder and run:
- **Non-reasoning fine-tuning:** `main.py`
- **Reasoning traces fine-tuning:** `main_reasoning.py`

### Reasoning Trace Generation

The reasoning trace generation code is located in `dataset_gen_reasoning/dataset_gpt_oss_tinker/`. This generates reasoning traces from a subset of 1,500 samples out of ~18k samples from KernelBook.

**Additional setup:** Create a new `.env` file in the `tinker/` folder with:
- `TINKER_API_KEY`
- `WANDB_API_KEY`