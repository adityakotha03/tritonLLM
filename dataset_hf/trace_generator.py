from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def load_config(config_path: Path | str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_prompt(python_code: str, triton_code: str, template: str) -> str:
    return template.format(
        python_code=python_code.strip(),
        triton_code=triton_code.strip(),
    )


def apply_stop_sequences(text: str, stop_sequences: List[str]) -> str:
    for stop in stop_sequences or []:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


class HFModelGenerator:
    def __init__(self, config: Dict):
        self.config = config
        model_id = config["model"]["model_id"]
        dtype_str = config["model"].get("dtype", "bfloat16")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        
        self.device = self._select_device()
        logger.info(f"Loading model {model_id} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        use_device_map = torch.cuda.is_available()
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if use_device_map:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not use_device_map:
            self.model.to(self.device)
        self.model.eval()
        
        self._compile_model()
        
        gen_config = config.get("generation", {})
        self.max_new_tokens = gen_config.get("max_new_tokens", 768)
        self.temperature = gen_config.get("temperature", 0.2)
        self.top_p = gen_config.get("top_p", 0.95)
        self.stop_sequences = gen_config.get("stop_sequences", [])
        self._generate_lock = threading.Lock()
    
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
        
    def _compile_model(self):
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile unavailable")
            return
        
        is_mps = self.device.type == "mps"
        if is_mps:
            logger.info("MPS detected: using regional compilation with dynamic shapes")
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
            except Exception as exc:
                logger.warning(f"MPS compilation failed: {exc}, using eager execution")
            return
        
        try:
            logger.info("Compiling model with torch.compile fullgraph=True")
            self.model = torch.compile(self.model, fullgraph=True, dynamic=True)
        except Exception as exc:
            logger.warning(f"Fullgraph compilation failed: {exc}, using regional compilation")
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    dynamic=True,
                )
            except Exception as exc2:
                logger.warning(f"Regional compilation failed: {exc2}, using eager execution")
    
    def generate(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["model"].get("max_model_len", 4096),
        ).to(self.device)
        
        do_sample = self.temperature > 0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        
        with self._generate_lock, torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        prompt_lengths = inputs["input_ids"].shape[1]
        completions = outputs[:, prompt_lengths:]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        return [apply_stop_sequences(text, self.stop_sequences) for text in texts]


def generate_trace(
    generator: HFModelGenerator,
    idx: int,
    record: Dict,
    template: str,
    lock: threading.Lock,
    results: List[Dict],
) -> None:
    python_code = record.get("python_code", "")
    triton_code = record.get("triton_code", "")
    
    if not python_code or not triton_code:
        logger.warning(f"Skipping sample {idx}: missing python_code or triton_code")
        return
    
    prompt = prepare_prompt(python_code, triton_code, template)
    
    try:
        traces = generator.generate([prompt])
        if not traces:
            logger.warning(f"Empty trace for sample {idx}")
            return
        
        trace = traces[0]
        row = {
            "sample_index": idx,
            "python_code": python_code,
            "triton_code": triton_code,
            "reasoning_trace": trace,
        }
        
        if "task_id" in record:
            row["task_id"] = record["task_id"]
        if "id" in record:
            row["source_id"] = record["id"]
        
        with lock:
            results.append(row)
            
    except Exception as e:
        logger.error(f"Error generating trace for sample {idx}: {e}")


def run_pipeline(config_path: Path | str = "dataset_hf/config.json") -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info(f"Loading configuration from {config_path}")
    
    config = load_config(config_path)
    
    dataset_name = config["dataset"]["name"]
    dataset_config = config["dataset"].get("config")
    split = config["dataset"].get("split", "train")
    limit = config["dataset"].get("limit")
    
    logger.info(f"Loading dataset: {dataset_name} ({split})")
    ds_kwargs = {}
    if dataset_config:
        ds_kwargs["name"] = dataset_config
    
    dataset = load_dataset(dataset_name, split=split, **ds_kwargs)
    
    if config["dataset"].get("shuffle", False):
        seed = config["dataset"].get("seed", 42)
        dataset = dataset.shuffle(seed=seed)
    
    if limit:
        dataset = dataset.select(range(limit))
    
    records = list(dataset)
    logger.info(f"Loaded {len(records)} records")
    
    generator = HFModelGenerator(config)
    template = config["runtime"]["prompt_template"]
    
    num_threads = config["runtime"].get("num_threads", 4)
    if generator.device.type == "mps":
        num_threads = 1
        logger.info("MPS detected: limiting to 1 thread to avoid MPS runtime contention")
    output_path = Path(config["runtime"]["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not config["runtime"].get("overwrite", False):
        raise FileExistsError(f"Output file {output_path} exists. Set overwrite=True to replace.")
    
    lock = threading.Lock()
    results: List[Dict] = []
    
    logger.info(f"Starting generation with {num_threads} worker threads")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(generate_trace, generator, idx, record, template, lock, results)
            for idx, record in enumerate(records)
        ]
        
        with tqdm(total=len(futures), desc="Generating traces", unit="sample") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
    
    results.sort(key=lambda x: x["sample_index"])
    
    logger.info(f"Writing {len(results)} rows to {output_path}")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    logger.info("Pipeline completed")
    return output_path


if __name__ == "__main__":
    config_path = os.environ.get("KERNEL_TRACE_CONFIG", "dataset_hf/config.json")
    run_pipeline(config_path)

