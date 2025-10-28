"""
Cloud Evaluation Manager.
Manages batch evaluation of kernels on RunPod serverless endpoints.
"""

import time
from typing import List, Dict, Optional
from src.search_config import KernelResult


class CloudEvaluator:
    """
    Manages batch evaluation of kernels using RunPod.
    """
    
    def __init__(
        self,
        endpoint,
        ref_arch_src: str,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timeout: int = 600,
        max_retries: int = 2
    ):
        """
        Initialize the cloud evaluator.
        
        Args:
            endpoint: RunPod endpoint object
            ref_arch_src: Reference architecture source code
            num_correct_trials: Number of correctness trials
            num_perf_trials: Number of performance trials
            timeout: Timeout for each evaluation (seconds)
            max_retries: Maximum number of retries for failed evaluations
        """
        self.endpoint = endpoint
        self.ref_arch_src = ref_arch_src
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.timeout = timeout
        self.max_retries = max_retries
    
    def evaluate_kernel(self, kernel: KernelResult) -> KernelResult:
        """
        Evaluate a single kernel on the cloud endpoint.
        
        Args:
            kernel: KernelResult to evaluate
            
        Returns:
            Updated KernelResult with evaluation results
        """
        for attempt in range(self.max_retries + 1):
            try:
                result = self.endpoint.run_sync(
                    {
                        "ref_arch_src": self.ref_arch_src,
                        "generated_code": kernel.code,
                        "num_correct_trials": self.num_correct_trials,
                        "num_perf_trials": self.num_perf_trials
                    },
                    timeout=self.timeout
                )
                
                # Update kernel with evaluation result
                kernel.eval_result = {
                    "compiled": result.get("compiled", False),
                    "correctness": result.get("correctness", False),
                    "runtime": result.get("runtime", -1.0),
                    "runtime_stats": result.get("runtime_stats", {}),
                    "ref_runtime": result.get("ref_runtime", -1.0),
                    "ref_runtime_compiled": result.get("ref_runtime_compiled", -1.0),
                    "speedup": result.get("speedup", -1.0),
                    "speedup_vs_compiled": result.get("speedup_vs_compiled", -1.0),
                    "metadata": result.get("metadata", {})
                }
                
                return kernel
                
            except TimeoutError:
                print(f"Evaluation timeout for kernel {kernel.kernel_id} (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt == self.max_retries:
                    kernel.eval_result = {
                        "compiled": False,
                        "correctness": False,
                        "runtime": -1.0,
                        "speedup": -1.0,
                        "error": "Evaluation timeout"
                    }
                    return kernel
                time.sleep(2)  # Brief pause before retry
                
            except Exception as e:
                print(f"Error evaluating kernel {kernel.kernel_id}: {e}")
                if attempt == self.max_retries:
                    kernel.eval_result = {
                        "compiled": False,
                        "correctness": False,
                        "runtime": -1.0,
                        "speedup": -1.0,
                        "error": str(e)
                    }
                    return kernel
                time.sleep(2)
        
        return kernel
    
    def evaluate_kernels_batch(
        self,
        kernels: List[KernelResult],
        show_progress: bool = True
    ) -> List[KernelResult]:
        """
        Evaluate a batch of kernels sequentially.
        
        Note: RunPod serverless handles concurrency automatically.
        We submit sequentially but RunPod can process in parallel.
        
        Args:
            kernels: List of KernelResult objects to evaluate
            show_progress: Whether to show progress messages
            
        Returns:
            List of evaluated KernelResult objects
        """
        evaluated_kernels = []
        
        for i, kernel in enumerate(kernels):
            if show_progress:
                print(f"Evaluating kernel {i + 1}/{len(kernels)}: {kernel.kernel_id}")
            
            evaluated_kernel = self.evaluate_kernel(kernel)
            evaluated_kernels.append(evaluated_kernel)
            
            if show_progress:
                if evaluated_kernel.is_correct():
                    speedup = evaluated_kernel.get_speedup()
                    print(f"  ✓ Correct | Speedup: {speedup:.2f}x")
                elif evaluated_kernel.is_compiled():
                    print(f"  ✗ Compiled but incorrect")
                else:
                    print(f"  ✗ Compilation failed")
        
        return evaluated_kernels
    
    def get_evaluation_summary(self, kernels: List[KernelResult]) -> Dict:
        """
        Get summary statistics for a batch of evaluated kernels.
        
        Args:
            kernels: List of evaluated KernelResult objects
            
        Returns:
            Dictionary with summary statistics
        """
        total = len(kernels)
        compiled = sum(1 for k in kernels if k.is_compiled())
        correct = sum(1 for k in kernels if k.is_correct())
        
        speedups = [k.get_speedup() for k in kernels if k.is_correct()]
        best_speedup = max(speedups) if speedups else 0.0
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        
        return {
            "total_evaluated": total,
            "compiled": compiled,
            "correct": correct,
            "compilation_rate": compiled / total if total > 0 else 0.0,
            "correctness_rate": correct / total if total > 0 else 0.0,
            "best_speedup": best_speedup,
            "average_speedup": avg_speedup,
            "num_with_speedup": len(speedups)
        }

