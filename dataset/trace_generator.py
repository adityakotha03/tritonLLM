from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

try:
    from .config_loader import PipelineConfig, load_config
except ImportError:
    from config_loader import PipelineConfig, load_config

logger = logging.getLogger(__name__)


def apply_stop_sequences(text: str, stop_sequences: List[str]) -> str:
    for stop in stop_sequences or []:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


class ModelGenerator:
    def __init__(self, config: PipelineConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        self.config = config
        model_id = config.model.model_id
        dtype = torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16
        
        logger.info(f"Loading model {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        if hasattr(torch, "compile"):
            try:
                import torch._dynamo as dynamo
                dynamo.config.dynamic_shapes = True
                dynamo.config.automatic_dynamic_shapes = True
            except (ImportError, AttributeError):
                pass
            
            try:
                logger.info("Compiling model with torch.compile")
                self.model = torch.compile(self.model, dynamic=True)
            except Exception as exc:
                logger.warning(f"Compilation failed: {exc}, using eager")
        
        gen = config.generation
        self.max_new_tokens = gen.max_new_tokens
        self.temperature = gen.temperature
        self.top_p = gen.top_p
        self.stop_sequences = gen.stop_sequences
    
    def generate(self, prompts: List[str]) -> List[str]:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_model_len or 4096,
        ).to(device)
        
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        prompt_lengths = inputs["input_ids"].shape[1]
        completions = outputs[:, prompt_lengths:]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        return [apply_stop_sequences(text, self.stop_sequences) for text in texts]


def run_pipeline(config_path: Path | str = "dataset/config.json") -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info(f"Loading configuration from {config_path}")
    
    config = load_config(config_path)
    
    logger.info(f"Loading dataset: {config.dataset.name} ({config.dataset.split})")
    ds_kwargs = {}
    if config.dataset.config:
        ds_kwargs["name"] = config.dataset.config
    
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
        logger.info(f"Loading existing results from {output_path}")
        existing_df = pd.read_csv(output_path)
        if len(existing_df) > 0:
            results = existing_df.to_dict("records")
            max_idx = int(existing_df["sample_index"].max())
            start_idx = max_idx + 1
            logger.info(f"Resuming from index {start_idx} ({len(results)} samples already processed)")
    
    logger.info(f"Starting generation with batch_size={batch_size}, save_interval={save_interval}")
    
    with tqdm(total=len(records), initial=start_idx, desc="Generating") as pbar:
        for i in range(start_idx, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch_prompts = []
            batch_indices = []
            
            for idx, record in enumerate(batch_records):
                if "python_code" not in record or "triton_code" not in record:
                    logger.warning(f"Skipping sample {i + idx}: missing fields")
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
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
    
    logger.info(f"Pipeline completed. Final results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    from pathlib import Path
    
    if "KERNEL_TRACE_CONFIG" in os.environ:
        config_path = os.environ["KERNEL_TRACE_CONFIG"]
    else:
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.json"
    
    run_pipeline(config_path)
