"""
Results Manager.
Handles saving and loading search results to/from JSON files.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from src.search_config import SearchConfig, OptimizationIdea, KernelResult


class ResultsManager:
    """
    Manages saving and loading search results.
    """
    
    def __init__(self, output_dir: str = "output/search_results"):
        """
        Initialize the results manager.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_search_results(
        self,
        config: SearchConfig,
        all_ideas: List[OptimizationIdea],
        all_kernels: List[KernelResult],
        best_kernel: Optional[KernelResult],
        statistics: Dict,
        round_summaries: List[Dict]
    ) -> str:
        """
        Save complete search results to a JSON file.
        
        Args:
            config: Search configuration used
            all_ideas: All optimization ideas generated
            all_kernels: All kernels generated and evaluated
            best_kernel: Best performing kernel
            statistics: Overall statistics
            round_summaries: Per-round summaries
            
        Returns:
            Path to the saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{config.problem_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        results = {
            "metadata": {
                "timestamp": timestamp,
                "problem_name": config.problem_name,
                "gpu_name": config.gpu_name,
                "model_name": config.model_name,
                "model_provider": config.model_provider
            },
            "config": config.to_dict(),
            "statistics": statistics,
            "round_summaries": round_summaries,
            "best_kernel": best_kernel.to_dict() if best_kernel else None,
            "all_ideas": [idea.to_dict() for idea in all_ideas],
            "all_kernels": [kernel.to_dict() for kernel in all_kernels]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def save_round_results(
        self,
        config: SearchConfig,
        round_number: int,
        ideas: List[OptimizationIdea],
        kernels: List[KernelResult],
        round_summary: Dict
    ) -> str:
        """
        Save results for a specific round.
        
        Args:
            config: Search configuration
            round_number: Round number
            ideas: Ideas generated in this round
            kernels: Kernels generated in this round
            round_summary: Summary statistics for the round
            
        Returns:
            Path to the saved round results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"round_{round_number}_{config.problem_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        round_results = {
            "metadata": {
                "timestamp": timestamp,
                "problem_name": config.problem_name,
                "round_number": round_number
            },
            "round_summary": round_summary,
            "ideas": [idea.to_dict() for idea in ideas],
            "kernels": [kernel.to_dict() for kernel in kernels]
        }
        
        with open(filepath, 'w') as f:
            json.dump(round_results, f, indent=2)
        
        return filepath
    
    def load_search_results(self, filepath: str) -> Dict:
        """
        Load search results from a JSON file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Dictionary with search results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results
    
    def save_best_kernel_code(
        self,
        kernel: KernelResult,
        config: SearchConfig
    ) -> str:
        """
        Save the best kernel code to a separate file for easy access.
        
        Args:
            kernel: Best kernel result
            config: Search configuration
            
        Returns:
            Path to the saved code file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_kernel_{config.problem_name}_{timestamp}.py"
        filepath = os.path.join(self.output_dir, filename)
        
        header = f"""# Best Kernel for {config.problem_name}
# Generated: {timestamp}
# Speedup: {kernel.get_speedup():.2f}x
# Runtime: {kernel.get_runtime():.4f} ms
# Round: {kernel.round_number}
# Idea: {kernel.metadata.get('idea_text', 'N/A')}

"""
        
        with open(filepath, 'w') as f:
            f.write(header)
            f.write(kernel.code)
        
        print(f"Best kernel code saved to: {filepath}")
        return filepath
    
    def list_saved_results(self) -> List[str]:
        """
        List all saved result files.
        
        Returns:
            List of result file paths
        """
        if not os.path.exists(self.output_dir):
            return []
        
        files = [
            os.path.join(self.output_dir, f)
            for f in os.listdir(self.output_dir)
            if f.endswith('.json')
        ]
        return sorted(files, reverse=True)  # Most recent first
    
    def generate_summary_report(
        self,
        config: SearchConfig,
        statistics: Dict,
        round_summaries: List[Dict],
        best_kernel: Optional[KernelResult]
    ) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            config: Search configuration
            statistics: Overall statistics
            round_summaries: Per-round summaries
            best_kernel: Best performing kernel
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 80)
        report.append("KERNEL OPTIMIZATION SEARCH RESULTS")
        report.append("=" * 80)
        report.append(f"Problem: {config.problem_name}")
        report.append(f"GPU: {config.gpu_name}")
        report.append(f"Model: {config.model_name} ({config.model_provider})")
        report.append(f"Rounds: {config.num_rounds}")
        report.append("")
        
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Kernels Generated: {statistics.get('total_kernels', 0)}")
        report.append(f"Compiled: {statistics.get('compiled_kernels', 0)}")
        report.append(f"Correct: {statistics.get('correct_kernels', 0)}")
        report.append(f"Best Speedup: {statistics.get('best_speedup', 0):.2f}x")
        report.append("")
        
        report.append("PER-ROUND SUMMARY")
        report.append("-" * 80)
        for summary in round_summaries:
            round_num = summary.get('round', 0)
            total = summary.get('total', 0)
            correct = summary.get('correct', 0)
            best_speedup = summary.get('best_speedup', 0)
            report.append(f"Round {round_num}: {total} kernels | {correct} correct | Best: {best_speedup:.2f}x")
        report.append("")
        
        if best_kernel:
            report.append("BEST KERNEL")
            report.append("-" * 80)
            report.append(f"Kernel ID: {best_kernel.kernel_id}")
            report.append(f"Round: {best_kernel.round_number}")
            report.append(f"Speedup: {best_kernel.get_speedup():.2f}x")
            report.append(f"Runtime: {best_kernel.get_runtime():.4f} ms")
            report.append(f"Idea: {best_kernel.metadata.get('idea_text', 'N/A')}")
        else:
            report.append("No valid kernel found")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_summary_report(
        self,
        config: SearchConfig,
        statistics: Dict,
        round_summaries: List[Dict],
        best_kernel: Optional[KernelResult]
    ) -> str:
        """
        Generate and save a summary report to a text file.
        
        Args:
            config: Search configuration
            statistics: Overall statistics
            round_summaries: Per-round summaries
            best_kernel: Best performing kernel
            
        Returns:
            Path to the saved report file
        """
        report = self.generate_summary_report(config, statistics, round_summaries, best_kernel)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{config.problem_name}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to: {filepath}")
        return filepath

