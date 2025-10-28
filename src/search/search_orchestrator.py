"""
Search Orchestrator.
Coordinates the multi-round optimization search process.
"""

from typing import List, Dict, Optional
from src.search_config import SearchConfig, OptimizationIdea, KernelResult
from src.LLMs.base_client import BaseLLMClient
from src.search.idea_generator import IdeaGenerator
from src.search.code_generator import CodeGenerator
from src.search.kernel_bank import KernelBank
from src.search.cloud_evaluator import CloudEvaluator
from src.search.results_manager import ResultsManager


class SearchOrchestrator:
    """
    Orchestrates the search-based kernel optimization process.
    """
    
    def __init__(
        self,
        config: SearchConfig,
        llm_client: BaseLLMClient,
        endpoint,
        ref_arch_src: str,
        results_manager: Optional[ResultsManager] = None
    ):
        """
        Initialize the search orchestrator.
        
        Args:
            config: Search configuration
            llm_client: LLM client for generating ideas and code
            endpoint: RunPod endpoint for evaluation
            ref_arch_src: Reference architecture source code
            results_manager: Results manager (creates default if None)
        """
        self.config = config
        self.ref_arch_src = ref_arch_src
        
        # Initialize components
        self.idea_generator = IdeaGenerator(llm_client, config)
        self.code_generator = CodeGenerator(llm_client, config)
        self.kernel_bank = KernelBank(top_k=config.top_k_bank)
        self.cloud_evaluator = CloudEvaluator(
            endpoint=endpoint,
            ref_arch_src=ref_arch_src,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            timeout=config.evaluation_timeout
        )
        self.results_manager = results_manager or ResultsManager()
        
        # Track all ideas and kernels
        self.all_ideas: List[OptimizationIdea] = []
        self.all_kernels: List[KernelResult] = []
        self.round_summaries: List[Dict] = []
    
    def run_search(self) -> Dict:
        """
        Run the complete search process.
        
        Returns:
            Dictionary with search results
        """
        print("=" * 80)
        print("STARTING KERNEL OPTIMIZATION SEARCH")
        print("=" * 80)
        print(f"Problem: {self.config.problem_name}")
        print(f"GPU: {self.config.gpu_name}")
        print(f"Model: {self.config.model_name}")
        print(f"Rounds: {self.config.num_rounds}")
        print(f"Width per round: {self.config.width_per_round}")
        print(f"Depth per round: {self.config.depth_per_round}")
        print("=" * 80)
        print()
        
        # Round 0: Generate baseline
        self._run_round_0()
        
        # Rounds 1-N: Iterative optimization
        for round_num in range(1, self.config.num_rounds):
            self._run_round(round_num)
        
        # Finalize and save results
        return self._finalize_search()
    
    def _run_round_0(self):
        """Run Round 0: Generate and evaluate baseline implementation."""
        print("\n" + "=" * 80)
        print("ROUND 0: Baseline Implementation")
        print("=" * 80)
        
        # Generate baseline idea
        baseline_idea = self.idea_generator.generate_initial_idea(round_number=0)
        self.all_ideas.append(baseline_idea)
        
        print(f"Baseline idea: {baseline_idea.idea_text}")
        
        # Generate baseline implementation
        print(f"\nGenerating baseline implementation...")
        baseline_kernel = self.code_generator.generate_baseline_implementation(
            ref_arch_src=self.ref_arch_src,
            idea=baseline_idea
        )
        
        # Evaluate baseline
        print(f"Evaluating baseline kernel...")
        evaluated_kernel = self.cloud_evaluator.evaluate_kernel(baseline_kernel)
        
        # Add to collections
        self.all_kernels.append(evaluated_kernel)
        self.kernel_bank.add_kernel(evaluated_kernel)
        
        # Print results
        if evaluated_kernel.is_correct():
            print(f"✓ Baseline kernel: Speedup {evaluated_kernel.get_speedup():.2f}x")
        elif evaluated_kernel.is_compiled():
            print(f"✗ Baseline kernel compiled but incorrect")
        else:
            print(f"✗ Baseline kernel failed to compile")
        
        # Save round summary
        round_summary = self.kernel_bank.get_round_summary(0)
        self.round_summaries.append(round_summary)
        
        print(f"\nRound 0 complete: {round_summary['correct']}/{round_summary['total']} correct")
    
    def _run_round(self, round_num: int):
        """
        Run a single optimization round.
        
        Args:
            round_num: Round number
        """
        print("\n" + "=" * 80)
        print(f"ROUND {round_num}: Optimization Search")
        print("=" * 80)
        
        # Get current best kernels for context
        top_kernels_info = self.kernel_bank.get_top_kernels_info()
        previous_idea_texts = [idea.idea_text for idea in self.all_ideas]
        
        # Generate optimization ideas
        width = self.config.get_width(round_num)
        print(f"\nGenerating {width} optimization ideas...")
        
        ideas = self.idea_generator.generate_ideas(
            ref_arch_src=self.ref_arch_src,
            current_best_kernels=top_kernels_info,
            previous_ideas=previous_idea_texts,
            round_number=round_num
        )
        
        self.all_ideas.extend(ideas)
        
        print(f"Generated {len(ideas)} ideas:")
        for i, idea in enumerate(ideas, 1):
            print(f"  {i}. {idea.idea_text[:100]}{'...' if len(idea.idea_text) > 100 else ''}")
        
        # Generate implementations for each idea
        depth = self.config.get_depth(round_num)
        print(f"\nGenerating {depth} implementation(s) per idea...")
        
        round_kernels = []
        for i, idea in enumerate(ideas, 1):
            print(f"\nIdea {i}/{len(ideas)}: Generating {depth} implementation(s)...")
            implementations = self.code_generator.generate_implementations(
                ref_arch_src=self.ref_arch_src,
                idea=idea,
                num_implementations=depth
            )
            round_kernels.extend(implementations)
        
        print(f"\nGenerated {len(round_kernels)} total implementations for round {round_num}")
        
        # Evaluate all kernels
        print(f"\nEvaluating {len(round_kernels)} kernels...")
        evaluated_kernels = self.cloud_evaluator.evaluate_kernels_batch(
            round_kernels,
            show_progress=True
        )
        
        # Add to collections
        self.all_kernels.extend(evaluated_kernels)
        num_added = self.kernel_bank.add_kernels(evaluated_kernels)
        
        # Print evaluation summary
        eval_summary = self.cloud_evaluator.get_evaluation_summary(evaluated_kernels)
        print(f"\nRound {round_num} Evaluation Summary:")
        print(f"  Total: {eval_summary['total_evaluated']}")
        print(f"  Compiled: {eval_summary['compiled']} ({eval_summary['compilation_rate']:.1%})")
        print(f"  Correct: {eval_summary['correct']} ({eval_summary['correctness_rate']:.1%})")
        print(f"  Best Speedup: {eval_summary['best_speedup']:.2f}x")
        print(f"  Avg Speedup: {eval_summary['average_speedup']:.2f}x")
        print(f"  Unique kernels added to bank: {num_added}")
        
        # Save round summary
        round_summary = self.kernel_bank.get_round_summary(round_num)
        self.round_summaries.append(round_summary)
        
        # Save round results
        self.results_manager.save_round_results(
            config=self.config,
            round_number=round_num,
            ideas=ideas,
            kernels=evaluated_kernels,
            round_summary=round_summary
        )
        
        # Show current top kernels
        print(f"\nCurrent Top {min(3, self.config.top_k_bank)} Kernels:")
        for i, kernel in enumerate(self.kernel_bank.get_top_k_kernels(k=3), 1):
            print(f"  {i}. Round {kernel.round_number} | Speedup: {kernel.get_speedup():.2f}x | Runtime: {kernel.get_runtime():.4f}ms")
    
    def _finalize_search(self) -> Dict:
        """
        Finalize the search and save all results.
        
        Returns:
            Dictionary with final results
        """
        print("\n" + "=" * 80)
        print("SEARCH COMPLETE")
        print("=" * 80)
        
        # Get best kernel
        best_kernel = self.kernel_bank.get_best_kernel()
        
        # Get statistics
        statistics = self.kernel_bank.get_statistics()
        
        # Print summary
        summary_report = self.results_manager.generate_summary_report(
            config=self.config,
            statistics=statistics,
            round_summaries=self.round_summaries,
            best_kernel=best_kernel
        )
        print("\n" + summary_report)
        
        # Save complete results
        results_path = self.results_manager.save_search_results(
            config=self.config,
            all_ideas=self.all_ideas,
            all_kernels=self.all_kernels,
            best_kernel=best_kernel,
            statistics=statistics,
            round_summaries=self.round_summaries
        )
        
        # Save best kernel code
        if best_kernel:
            self.results_manager.save_best_kernel_code(best_kernel, self.config)
        
        # Save summary report
        self.results_manager.save_summary_report(
            config=self.config,
            statistics=statistics,
            round_summaries=self.round_summaries,
            best_kernel=best_kernel
        )
        
        return {
            "config": self.config,
            "best_kernel": best_kernel,
            "statistics": statistics,
            "round_summaries": self.round_summaries,
            "all_ideas": self.all_ideas,
            "all_kernels": self.all_kernels,
            "results_path": results_path
        }

