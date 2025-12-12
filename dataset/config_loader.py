from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_id: str
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    max_model_len: Optional[int] = None
    enforce_eager: bool = False


@dataclass
class DatasetConfig:
    name: str
    config: Optional[str] = None
    split: str = "train"
    limit: Optional[int] = None
    shuffle: bool = False
    seed: Optional[int] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None


@dataclass
class RuntimeConfig:
    batch_size: int = 1
    num_threads: int = 1
    prompt_template: str = (
        "Given the reference PyTorch and Triton implementations, explain how the Triton"
        " kernel matches the PyTorch semantics."
    )
    output_csv: str = "dataset/kernel_traces.csv"
    overwrite: bool = False
    save_interval: int = 100


@dataclass
class PipelineConfig:
    model: ModelConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    runtime: RuntimeConfig

    @property
    def output_path(self) -> Path:
        return Path(self.runtime.output_csv)


def _expect_section(raw: Dict[str, Any], section: str) -> Dict[str, Any]:
    if section not in raw:
        raise KeyError(f"Missing '{section}' section in config.json")
    block = raw[section]
    if not isinstance(block, dict):
        raise TypeError(f"Expected '{section}' section to be an object, got {type(block)}")
    return block


def _load_model(model_block: Dict[str, Any]) -> ModelConfig:
    if "model_id" not in model_block:
        raise KeyError("Model configuration requires 'model_id'")
    return ModelConfig(
        model_id=model_block["model_id"],
        tensor_parallel_size=int(model_block.get("tensor_parallel_size", 1)),
        dtype=str(model_block.get("dtype", "bfloat16")),
        max_model_len=model_block.get("max_model_len"),
        enforce_eager=bool(model_block.get("enforce_eager", False)),
    )


def _load_dataset(dataset_block: Dict[str, Any]) -> DatasetConfig:
    if "name" not in dataset_block:
        raise KeyError("Dataset configuration requires 'name'")
    return DatasetConfig(
        name=str(dataset_block["name"]),
        config=dataset_block.get("config"),
        split=str(dataset_block.get("split", "train")),
        limit=dataset_block.get("limit"),
        shuffle=bool(dataset_block.get("shuffle", False)),
        seed=dataset_block.get("seed"),
    )


def _load_generation(gen_block: Dict[str, Any]) -> GenerationConfig:
    stop_sequences = gen_block.get("stop_sequences")
    if stop_sequences is not None and not isinstance(stop_sequences, list):
        raise TypeError("Generation 'stop_sequences' must be a list of strings")
    return GenerationConfig(
        max_new_tokens=int(gen_block.get("max_new_tokens", 512)),
        temperature=float(gen_block.get("temperature", 0.0)),
        top_p=float(gen_block.get("top_p", 0.95)),
        presence_penalty=float(gen_block.get("presence_penalty", 0.0)),
        frequency_penalty=float(gen_block.get("frequency_penalty", 0.0)),
        stop_sequences=stop_sequences,
    )


def _load_runtime(runtime_block: Dict[str, Any]) -> RuntimeConfig:
    prompt_template = runtime_block.get("prompt_template")
    if prompt_template is None:
        raise KeyError("Runtime configuration requires 'prompt_template'")
    return RuntimeConfig(
        batch_size=int(runtime_block.get("batch_size", 1)),
        num_threads=int(runtime_block.get("num_threads", 1)),
        prompt_template=str(prompt_template),
        output_csv=str(runtime_block.get("output_csv", "dataset/kernel_traces.csv")),
        overwrite=bool(runtime_block.get("overwrite", False)),
        save_interval=int(runtime_block.get("save_interval", 100)),
    )


def load_config(path: Path | str) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object")

    model_block = _expect_section(raw, "model")
    dataset_block = _expect_section(raw, "dataset")
    generation_block = _expect_section(raw, "generation")
    runtime_block = _expect_section(raw, "runtime")

    pipeline_config = PipelineConfig(
        model=_load_model(model_block),
        dataset=_load_dataset(dataset_block),
        generation=_load_generation(generation_block),
        runtime=_load_runtime(runtime_block),
    )

    return pipeline_config

