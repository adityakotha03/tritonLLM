"""
Code Generator.
Converts optimization ideas into actual kernel implementations.
"""

import uuid
import hashlib
from typing import List
from src.search_config import OptimizationIdea, KernelResult, SearchConfig
from src.LLMs.base_client import BaseLLMClient
from src.prompts import construct_implementation_prompt, construct_prompt_zero_shot
from src.utils.common import clean_markdown_code


class CodeGenerator:
    """
    Generates code implementations from optimization ideas.
    """
    
    def __init__(self, llm_client: BaseLLMClient, config: SearchConfig):
        """
        Initialize the code generator.
        
        Args:
            llm_client: LLM client for generating code
            config: Search configuration
        """
        self.llm_client = llm_client
        self.config = config
    
    def generate_implementations(
        self,
        ref_arch_src: str,
        idea: OptimizationIdea,
        num_implementations: int = None
    ) -> List[KernelResult]:
        """
        Generate multiple implementations of an optimization idea.
        
        Args:
            ref_arch_src: Reference architecture source code
            idea: Optimization idea to implement
            num_implementations: Number of implementations to generate (uses config if None)
            
        Returns:
            List of KernelResult objects with generated code
        """
        if num_implementations is None:
            num_implementations = self.config.get_depth(idea.round_number)
        
        # Construct the implementation prompt
        prompt = construct_implementation_prompt(
            gpu_name=self.config.gpu_name,
            ref_arch_src=ref_arch_src,
            optimization_idea=idea.idea_text,
            round_number=idea.round_number
        )
        
        implementations = []
        
        for i in range(num_implementations):
            # Generate code using LLM
            # Use higher temperature for diversity if generating multiple implementations
            # Note: Some models (like gpt-5/o1) only support temperature=1.0
            temperature = self.config.temperature
            
            generated_code = self.llm_client.generate_implementation(
                prompt=prompt,
                max_completion_tokens=self.config.max_completion_tokens,
                temperature=temperature
            )
            
            # Clean the generated code (remove markdown formatting)
            cleaned_code = clean_markdown_code(generated_code)
            
            # Create KernelResult object
            kernel_result = KernelResult(
                code=cleaned_code,
                kernel_id=f"kernel_r{idea.round_number}_{idea.idea_id}_{i}_{uuid.uuid4().hex[:6]}",
                idea_id=idea.idea_id,
                round_number=idea.round_number,
                code_hash=self._hash_code(cleaned_code),
                metadata={
                    "implementation_index": i,
                    "idea_text": idea.idea_text,
                    "temperature": temperature
                }
            )
            
            implementations.append(kernel_result)
        
        return implementations
    
    def generate_baseline_implementation(
        self,
        ref_arch_src: str,
        idea: OptimizationIdea
    ) -> KernelResult:
        """
        Generate the initial baseline implementation (Round 0).
        
        Args:
            ref_arch_src: Reference architecture source code
            idea: Baseline optimization idea
            
        Returns:
            KernelResult with baseline implementation
        """
        # Use the zero-shot prompt for baseline
        prompt = construct_prompt_zero_shot(
            gpu_name=self.config.gpu_name,
            ref_arch_src=ref_arch_src
        )
        
        generated_code = self.llm_client.generate_implementation(
            prompt=prompt,
            max_completion_tokens=self.config.max_completion_tokens,
            temperature=self.config.temperature
        )
        
        cleaned_code = clean_markdown_code(generated_code)
        
        kernel_result = KernelResult(
            code=cleaned_code,
            kernel_id=f"kernel_r{idea.round_number}_baseline_{uuid.uuid4().hex[:6]}",
            idea_id=idea.idea_id,
            round_number=idea.round_number,
            code_hash=self._hash_code(cleaned_code),
            metadata={
                "is_baseline": True,
                "idea_text": idea.idea_text
            }
        )
        
        return kernel_result
    
    def _hash_code(self, code: str) -> str:
        """
        Generate a hash of the code for duplicate detection.
        
        Args:
            code: Source code string
            
        Returns:
            SHA256 hash of the code
        """
        return hashlib.sha256(code.encode()).hexdigest()

