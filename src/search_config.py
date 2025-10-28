"""
Search configuration for the kernel optimization search algorithm.
Supports configurable width (ideas per round) and depth (implementations per idea).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SearchConfig:
    """
    Configuration for the search-based kernel optimization.
    
    Attributes:
        num_rounds: Total number of rounds (includes Round 0 baseline + optimization rounds)
        width_per_round: Number of optimization ideas per optimization round (length = num_rounds - 1)
                        Round 0 is baseline with 1 idea, so array should have (num_rounds - 1) values
        depth_per_round: Number of implementations per idea for optimization rounds (length = num_rounds - 1)
                        Round 0 baseline generates 1 implementation, so array should have (num_rounds - 1) values
        top_k_bank: Number of top kernels to keep in the bank
        model_name: LLM model to use (e.g., "gpt-5", "gemini-2.5-pro")
        model_provider: LLM provider ("openai" or "gemini")
        temperature: Sampling temperature for LLM generation
        max_completion_tokens: Maximum tokens for LLM response
        gpu_name: Target GPU name for optimization
        problem_name: Name of the problem being optimized
        num_correct_trials: Number of correctness trials for evaluation
        num_perf_trials: Number of performance trials for benchmarking
        evaluation_timeout: Timeout for each evaluation job (seconds)
    """
    
    # Search parameters
    num_rounds: int = 5
    width_per_round: List[int] = field(default_factory=lambda: [3, 4, 4, 5])  # 4 values for rounds 1-4 (Round 0 is baseline)
    depth_per_round: List[int] = field(default_factory=lambda: [2, 3, 3, 3])  # 4 values for rounds 1-4 (Round 0 is baseline)
    top_k_bank: int = 5
    
    # LLM parameters
    model_name: str = "gpt-5"
    model_provider: str = "openai"  # "openai" or "gemini"
    temperature: float = 1.0
    max_completion_tokens: int = 8192
    
    # Problem parameters
    gpu_name: str = "A100-80GB"
    problem_name: str = "unknown"
    
    # Evaluation parameters
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    evaluation_timeout: int = 600
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # width_per_round and depth_per_round should have length (num_rounds - 1)
        # since Round 0 is hardcoded baseline and doesn't use these arrays
        expected_length = self.num_rounds - 1
        if len(self.width_per_round) != expected_length:
            raise ValueError(
                f"width_per_round must have length {expected_length} (num_rounds - 1 for optimization rounds), got {len(self.width_per_round)}"
            )
        if len(self.depth_per_round) != expected_length:
            raise ValueError(
                f"depth_per_round must have length {expected_length} (num_rounds - 1 for optimization rounds), got {len(self.depth_per_round)}"
            )
        if self.model_provider not in ["openai", "gemini"]:
            raise ValueError(
                f"model_provider must be 'openai' or 'gemini', got {self.model_provider}"
            )
    
    def get_width(self, round_num: int) -> int:
        """Get number of ideas for a specific round.
        
        Note: Round 0 (baseline) doesn't use this, so index is (round_num - 1).
        """
        if round_num == 0:
            raise ValueError("Round 0 (baseline) doesn't use width_per_round")
        return self.width_per_round[round_num - 1]
    
    def get_depth(self, round_num: int) -> int:
        """Get number of implementations per idea for a specific round.
        
        Note: Round 0 (baseline) doesn't use this, so index is (round_num - 1).
        """
        if round_num == 0:
            raise ValueError("Round 0 (baseline) doesn't use depth_per_round")
        return self.depth_per_round[round_num - 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            "num_rounds": self.num_rounds,
            "width_per_round": self.width_per_round,
            "depth_per_round": self.depth_per_round,
            "top_k_bank": self.top_k_bank,
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "gpu_name": self.gpu_name,
            "problem_name": self.problem_name,
            "num_correct_trials": self.num_correct_trials,
            "num_perf_trials": self.num_perf_trials,
            "evaluation_timeout": self.evaluation_timeout,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SearchConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class OptimizationIdea:
    """
    Represents a natural language optimization idea.
    
    Attributes:
        idea_text: The optimization idea in natural language
        round_number: Which round this idea was generated in
        idea_id: Unique identifier for this idea
        parent_kernel_ids: IDs of kernels that inspired this idea
        metadata: Additional metadata about the idea
    """
    idea_text: str
    round_number: int
    idea_id: str
    parent_kernel_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "idea_text": self.idea_text,
            "round_number": self.round_number,
            "idea_id": self.idea_id,
            "parent_kernel_ids": self.parent_kernel_ids,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationIdea":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class KernelResult:
    """
    Represents a generated kernel and its evaluation results.
    
    Attributes:
        code: The generated kernel code
        kernel_id: Unique identifier for this kernel
        idea_id: ID of the optimization idea that generated this kernel
        round_number: Which round this kernel was generated in
        eval_result: Evaluation results (compiled, correctness, performance)
        code_hash: Hash of the code for duplicate detection
        metadata: Additional metadata
    """
    code: str
    kernel_id: str
    idea_id: str
    round_number: int
    eval_result: Optional[Dict[str, Any]] = None
    code_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "kernel_id": self.kernel_id,
            "idea_id": self.idea_id,
            "round_number": self.round_number,
            "eval_result": self.eval_result,
            "code_hash": self.code_hash,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KernelResult":
        """Create from dictionary."""
        return cls(**data)
    
    def get_runtime(self) -> float:
        """Get runtime from eval result, returns -1 if not available."""
        if self.eval_result and "runtime" in self.eval_result:
            return self.eval_result["runtime"]
        return -1.0
    
    def get_speedup(self) -> float:
        """Get speedup from eval result, returns -1 if not available."""
        if self.eval_result and "speedup" in self.eval_result:
            return self.eval_result["speedup"]
        return -1.0
    
    def is_correct(self) -> bool:
        """Check if kernel is correct."""
        if self.eval_result and "correctness" in self.eval_result:
            return self.eval_result["correctness"]
        return False
    
    def is_compiled(self) -> bool:
        """Check if kernel compiled successfully."""
        if self.eval_result and "compiled" in self.eval_result:
            return self.eval_result["compiled"]
        return False

