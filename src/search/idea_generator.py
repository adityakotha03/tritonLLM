"""
Optimization Idea Generator.
Generates natural language optimization ideas based on current performance and previous attempts.
"""

import uuid
from typing import List
from src.search_config import OptimizationIdea, SearchConfig
from src.LLMs.base_client import BaseLLMClient
from src.prompts import construct_idea_generation_prompt


class IdeaGenerator:
    """
    Generates optimization ideas using an LLM.
    """
    
    def __init__(self, llm_client: BaseLLMClient, config: SearchConfig):
        """
        Initialize the idea generator.
        
        Args:
            llm_client: LLM client for generating ideas
            config: Search configuration
        """
        self.llm_client = llm_client
        self.config = config
    
    def generate_ideas(
        self,
        ref_arch_src: str,
        current_best_kernels: List[dict],
        previous_ideas: List[str],
        round_number: int
    ) -> List[OptimizationIdea]:
        """
        Generate optimization ideas for the current round.
        
        Args:
            ref_arch_src: Reference architecture source code
            current_best_kernels: List of current best kernels with performance info
            previous_ideas: List of previously attempted ideas
            round_number: Current round number
            
        Returns:
            List of OptimizationIdea objects
        """
        num_ideas = self.config.get_width(round_number)
        
        # Construct the prompt
        prompt = construct_idea_generation_prompt(
            gpu_name=self.config.gpu_name,
            ref_arch_src=ref_arch_src,
            current_best_kernels=current_best_kernels,
            previous_ideas=previous_ideas,
            round_number=round_number,
            num_ideas=num_ideas
        )
        
        # Generate ideas using LLM
        idea_texts = self.llm_client.generate_optimization_ideas(
            prompt=prompt,
            num_ideas=num_ideas,
            max_completion_tokens=self.config.max_completion_tokens,
            temperature=self.config.temperature
        )
        
        # Create OptimizationIdea objects
        ideas = []
        parent_kernel_ids = [k.get('kernel_id', '') for k in current_best_kernels[:3]]
        
        for idea_text in idea_texts:
            idea = OptimizationIdea(
                idea_text=idea_text,
                round_number=round_number,
                idea_id=f"idea_r{round_number}_{uuid.uuid4().hex[:8]}",
                parent_kernel_ids=parent_kernel_ids,
                metadata={
                    "num_best_kernels": len(current_best_kernels),
                    "num_previous_ideas": len(previous_ideas)
                }
            )
            ideas.append(idea)
        
        return ideas
    
    def generate_initial_idea(self, round_number: int = 0) -> OptimizationIdea:
        """
        Generate the initial baseline idea for round 0.
        
        Args:
            round_number: Round number (typically 0 for initial)
            
        Returns:
            OptimizationIdea for the baseline implementation
        """
        idea = OptimizationIdea(
            idea_text="Given the PyTorch code, replace the operation with a custom Triton kernel",
            round_number=round_number,
            idea_id=f"idea_r{round_number}_baseline",
            parent_kernel_ids=[],
            metadata={"is_baseline": True}
        )
        return idea

