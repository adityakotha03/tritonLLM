from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path | str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class GeminiGenerator:
    def __init__(self, config: Dict):
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in config or environment")
        
        genai.configure(api_key=api_key)
        
        model_name = config.get("model_name", "gemini-2.0-flash-exp")
        logger.info(f"Initializing Gemini model: {model_name}")
        
        generation_config = config.get("generation", {})
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": generation_config.get("temperature", 0.2),
                "top_p": generation_config.get("top_p", 0.95),
                "max_output_tokens": generation_config.get("max_new_tokens", 768),
            },
        )
        self.stop_sequences = generation_config.get("stop_sequences", [])
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            for stop in self.stop_sequences:
                if stop and stop in text:
                    text = text[:text.find(stop)].strip()
            
            return text
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""


def generate_trace(
    generator: GeminiGenerator,
    idx: int,
    record: Dict,
    template: str,
    lock,
    results: List[Dict],
) -> None:
    if "python_code" not in record or "triton_code" not in record:
        return
    
    prompt = template.format(
        python_code=record["python_code"].strip(),
        triton_code=record["triton_code"].strip(),
    )
    
    trace = generator.generate(prompt)
    
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
    
    with lock:
        results.append(row)


def run_pipeline(config_path: Path | str = "dataset_gemini/config.json") -> Path:
    config = load_config(config_path)
    
    dataset_name = config["dataset"]["name"]
    dataset_config = config["dataset"].get("config")
    split = config["dataset"].get("split", "train")
    limit = config["dataset"].get("limit")
    
    logger.info(f"Loading dataset: {dataset_name}")
    ds_kwargs = {"name": dataset_config} if dataset_config else {}
    dataset = load_dataset(dataset_name, split=split, **ds_kwargs)
    
    if config["dataset"].get("shuffle", False):
        dataset = dataset.shuffle(seed=config["dataset"].get("seed", 42))
    if limit:
        dataset = dataset.select(range(limit))
    
    records = list(dataset)
    logger.info(f"Loaded {len(records)} records")
    
    generator = GeminiGenerator(config)
    template = config["runtime"]["prompt_template"]
    num_threads = config["runtime"].get("num_threads", 8)
    save_interval = config["runtime"].get("save_interval", 200)
    output_path = Path(config["runtime"]["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results: List[Dict] = []
    start_idx = 0
    
    if output_path.exists() and not config["runtime"].get("overwrite", False):
        existing_df = pd.read_csv(output_path)
        if len(existing_df) > 0:
            results = existing_df.to_dict("records")
            start_idx = int(existing_df["sample_index"].max()) + 1
            logger.info(f"Resuming from index {start_idx}")
    
    logger.info(f"Starting generation: num_threads={num_threads}, save_interval={save_interval}")
    
    lock = threading.Lock()
    processed_count = start_idx
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(generate_trace, generator, idx, record, template, lock, results): idx
            for idx, record in enumerate(records)
            if idx >= start_idx
        }
        
        with tqdm(total=len(futures), initial=start_idx, desc="Generating") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    processed_count += 1
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Error processing sample {idx}: {e}")
                finally:
                    pbar.update(1)
                    
                    if processed_count % save_interval == 0:
                        logger.info(f"Saving checkpoint: {len(results)} samples")
                        df = pd.DataFrame(results)
                        df.to_csv(output_path, index=False)
    
    logger.info(f"Saving final results: {len(results)} samples")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Completed. Results saved to {output_path}")
    return output_path


if __name__ == "__main__":
    config_path = os.environ.get("GEMINI_TRACE_CONFIG") or Path(__file__).parent / "config.json"
    run_pipeline(config_path)

