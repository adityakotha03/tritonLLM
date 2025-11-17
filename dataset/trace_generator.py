from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

try:
    from .config_loader import PipelineConfig, load_config
except ImportError:
    from config_loader import PipelineConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class ModelGenerator:
    def __init__(self, config: PipelineConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        logger.info(f"Loading vLLM model: {config.model.model_id}")
        self.llm = LLM(
            model=config.model.model_id,
            tensor_parallel_size=config.model.tensor_parallel_size,
            dtype=config.model.dtype,
            max_model_len=config.model.max_model_len,
            enforce_eager=config.model.enforce_eager,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )
        
        gen = config.generation
        self.sampling_params = SamplingParams(
            max_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            presence_penalty=gen.presence_penalty,
            frequency_penalty=gen.frequency_penalty,
            stop=gen.stop_sequences or None,
        )
    
    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]


def run_pipeline(config_path: Path | str = "dataset/config.json") -> Path:
    config = load_config(config_path)
    
    logger.info(f"Loading dataset: {config.dataset.name}")
    ds_kwargs = {"name": config.dataset.config} if config.dataset.config else {}
    dataset = load_dataset(config.dataset.name, split=config.dataset.split, **ds_kwargs)
    
    if config.dataset.shuffle:
        dataset = dataset.shuffle(seed=config.dataset.seed)
    if config.dataset.limit:
        dataset = dataset.select(range(config.dataset.limit))
    
    records = list(dataset)
    logger.info(f"Loaded {len(records)} records")
    
    generator = ModelGenerator(config)
    template = config.runtime.prompt_template
    batch_size = config.runtime.batch_size
    save_interval = config.runtime.save_interval
    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results: List[Dict] = []
    start_idx = 0
    
    if output_path.exists() and not config.runtime.overwrite:
        existing_df = pd.read_csv(output_path)
        if len(existing_df) > 0:
            results = existing_df.to_dict("records")
            start_idx = int(existing_df["sample_index"].max()) + 1
            logger.info(f"Resuming from index {start_idx}")
    
    logger.info(f"Starting generation: batch_size={batch_size}, save_interval={save_interval}")
    
    with tqdm(total=len(records), initial=start_idx, desc="Generating") as pbar:
        for i in range(start_idx, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch_prompts = []
            batch_indices = []
            
            for idx, record in enumerate(batch_records):
                if "python_code" not in record or "triton_code" not in record:
                    continue
                prompt = template.format(
                    python_code=record["python_code"].strip(),
                    triton_code=record["triton_code"].strip(),
                )
                batch_prompts.append(prompt)
                batch_indices.append(i + idx)
            
            if not batch_prompts:
                pbar.update(len(batch_records))
                continue
            
            traces = generator.generate(batch_prompts)
            
            for trace, idx, record in zip(traces, batch_indices, batch_records):
                row = {
                    "sample_index": idx,
                    "python_code": record["python_code"],
                    "triton_code": record["triton_code"],
                    "reasoning_trace": trace,
                }
                if "task_id" in record:
                    row["task_id"] = record["task_id"]
                if "id" in record:
                    row["source_id"] = record["id"]
                results.append(row)
            
            pbar.update(len(batch_records))
            
            if (i + batch_size) % save_interval < batch_size or (i + batch_size) >= len(records):
                logger.info(f"Saving checkpoint: {len(results)} samples")
                pd.DataFrame(results).to_csv(output_path, index=False)
    
    logger.info(f"Completed. Results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    config_path = os.environ.get("KERNEL_TRACE_CONFIG") or Path(__file__).parent / "config.json"
    run_pipeline(config_path)
