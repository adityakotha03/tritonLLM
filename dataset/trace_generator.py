from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

from dataset.config_loader import PipelineConfig, load_config

logger = logging.getLogger(__name__)


def _prepare_prompts(config: PipelineConfig, records: Iterable[Dict]) -> List[Tuple[int, Dict, str]]:
    prepared: List[Tuple[int, Dict, str]] = []
    template = config.runtime.prompt_template

    for idx, record in enumerate(records):
        if "python_code" not in record or "triton_code" not in record:
            raise KeyError("Each dataset sample must include 'python_code' and 'triton_code' fields.")

        prompt = template.format(
            python_code=record["python_code"].strip(),
            triton_code=record["triton_code"].strip(),
        )
        prepared.append((idx, record, prompt))

    return prepared


def _load_dataset_records(config: PipelineConfig) -> List[Dict]:
    ds_kwargs = {}
    if config.dataset.config is not None:
        ds_kwargs["name"] = config.dataset.config

    dataset = load_dataset(
        path=config.dataset.name,
        split=config.dataset.split,
        **ds_kwargs,
    )

    if config.dataset.shuffle:
        dataset = dataset.shuffle(seed=config.dataset.seed)

    if config.dataset.limit is not None:
        dataset = dataset.select(range(config.dataset.limit))

    return list(dataset)


def _ensure_output_path(config: PipelineConfig) -> Path:
    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not config.runtime.overwrite:
        raise FileExistsError(
            f"Output file {output_path} exists. Enable overwrite in config.runtime.overwrite to replace it."
        )
    return output_path


def _torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.float32)


class GenerationBackend:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.max_workers = config.runtime.num_threads

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError


class CUDABackend(GenerationBackend):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        logger.info("Using CUDA backend with vLLM")
        self.llm = LLM(
            model=config.model.model_id,
            tensor_parallel_size=config.model.tensor_parallel_size,
            dtype=config.model.dtype,
            enforce_eager=config.model.enforce_eager,
            max_model_len=config.model.max_model_len,
            trust_remote_code=True,
        )
        gen = config.generation
        self.sampling_params = SamplingParams(
            max_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            presence_penalty=gen.presence_penalty,
            frequency_penalty=gen.frequency_penalty,
            stop=gen.stop_sequences,
        )

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        if not outputs or len(outputs) != len(prompts):
            raise RuntimeError("Mismatch between prompts and vLLM outputs")
        results: List[str] = []
        for output in outputs:
            if not output.outputs:
                raise RuntimeError("vLLM returned empty completion")
            results.append(output.outputs[0].text.strip())
        return results


class MPSBackend(GenerationBackend):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        logger.info("Using MPS backend with transformers AutoModel")
        self.max_workers = 1  # avoid contention on MPS runtime
        self.device = torch.device("mps")
        dtype = _torch_dtype_from_str(config.model.dtype)

        tokenizer_kwargs = {"use_fast": True}
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, **tokenizer_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model.model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("trust_remote_code")
            self.model = AutoModelForCausalLM.from_pretrained(config.model.model_id, **model_kwargs)

        self.model.to(self.device)
        self.model.eval()

        self._compile_model()

    def _compile_model(self) -> None:
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile unavailable; using eager execution on MPS")
            return

        try:
            logger.info("Compiling model with torch.compile fullgraph=True")
            self.model = torch.compile(self.model, fullgraph=True, dynamic=True)
        except Exception as exc:
            logger.warning("Fullgraph torch.compile failed on MPS (%s); retrying with defaults", exc)
            try:
                self.model = torch.compile(self.model, dynamic=True)
            except Exception as exc2:
                logger.warning("torch.compile fallback failed (%s); using eager execution", exc2)

    def _apply_stop_sequences(self, text: str) -> str:
        stop_sequences = self.config.generation.stop_sequences or []
        for stop in stop_sequences:
            if not stop:
                continue
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        return text.strip()

    def generate(self, prompts: List[str]) -> List[str]:
        gen = self.config.generation
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        do_sample = gen.temperature > 0
        generation_kwargs = {
            "max_new_tokens": gen.max_new_tokens,
            "temperature": gen.temperature if do_sample else None,
            "top_p": gen.top_p if do_sample else None,
            "do_sample": do_sample,
        }

        if gen.presence_penalty or gen.frequency_penalty:
            # Approximate presence/frequency penalties via repetition penalty
            generation_kwargs["repetition_penalty"] = max(
                1.0, 1.0 + max(gen.presence_penalty, gen.frequency_penalty)
            )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **{k: v for k, v in generation_kwargs.items() if v is not None})

        prompt_lengths = inputs["input_ids"].shape[1]
        completions = outputs[:, prompt_lengths:]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        return [self._apply_stop_sequences(text) for text in texts]


def _select_backend(config: PipelineConfig) -> GenerationBackend:
    if torch.cuda.is_available():
        return CUDABackend(config)

    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_available:
        return MPSBackend(config)

    raise EnvironmentError("Neither CUDA nor MPS backends are available for accelerated inference.")


def run_pipeline(config_path: Path | str = "dataset/config.json") -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Loading configuration from %s", config_path)

    config = load_config(config_path)
    logger.info("Configuration loaded: %s", asdict(config.model))

    logger.info("Loading dataset: %s (%s)", config.dataset.name, config.dataset.split)
    records = _load_dataset_records(config)
    logger.info("Loaded %d records", len(records))

    prompts = _prepare_prompts(config, records)
    logger.info("Prepared prompts for generation")

    backend = _select_backend(config)
    batch_size = max(1, config.runtime.batch_size)

    lock = threading.Lock()
    results: List[Dict[str, str]] = []

    def _batched(sequence: List[Tuple[int, Dict, str]], size: int) -> Iterable[List[Tuple[int, Dict, str]]]:
        for start in range(0, len(sequence), size):
            yield sequence[start : start + size]

    def _generate(batch: List[Tuple[int, Dict, str]]) -> Tuple[int, int]:
        batch_indices = [entry[0] for entry in batch]
        prompts_only = [entry[2] for entry in batch]
        texts = backend.generate(prompts_only)

        if len(texts) != len(batch):
            raise RuntimeError("Mismatch between generated texts and batch size")

        batch_rows: List[Dict[str, str]] = []
        for text, (index, sample, _) in zip(texts, batch):
            row = {
                "sample_index": index,
                "python_code": sample["python_code"],
                "triton_code": sample["triton_code"],
                "reasoning_trace": text,
            }
            if "task_id" in sample:
                row["task_id"] = sample["task_id"]
            if "id" in sample:
                row["source_id"] = sample["id"]
            batch_rows.append(row)

        with lock:
            results.extend(batch_rows)

        return batch_indices[0], batch_indices[-1]

    logger.info("Starting generation with up to %d worker threads", backend.max_workers)
    batches = list(_batched(prompts, batch_size))

    with ThreadPoolExecutor(max_workers=backend.max_workers) as executor:
        futures = [executor.submit(_generate, batch) for batch in batches]
        with tqdm(total=len(futures), desc="Generating", unit="batch") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    results.sort(key=lambda row: row["sample_index"])
    output_path = _ensure_output_path(config)

    logger.info("Writing %d rows to %s", len(results), output_path)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    logger.info("Pipeline completed")
    return output_path


if __name__ == "__main__":
    config_path = os.environ.get("KERNEL_TRACE_CONFIG", "dataset/config.json")
    run_pipeline(config_path)

