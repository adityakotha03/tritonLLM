"""
Abstract base class for LLM clients.
Provides a model-agnostic interface for generating optimization ideas and implementations.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    All LLM providers (OpenAI, Gemini, etc.) should inherit from this class.
    """
    
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        max_completion_tokens: int = 8192,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate text completion from the LLM.
        
        Args:
            prompt: Input prompt
            max_completion_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text as string
        """
        pass
    
    def generate_optimization_ideas(
        self,
        prompt: str,
        num_ideas: int = 3,
        max_completion_tokens: int = 4096,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate natural language optimization ideas.
        
        Args:
            prompt: Prompt describing the context and asking for ideas
            num_ideas: Number of ideas to generate
            max_completion_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of optimization ideas as strings
        """
        # Default implementation generates one completion and expects ideas separated by newlines
        # Subclasses can override for more sophisticated approaches
        response = self.generate_text(
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Parse ideas from the response
        ideas = self._parse_ideas(response, num_ideas)
        return ideas
    
    def generate_implementation(
        self,
        prompt: str,
        max_completion_tokens: int = 8192,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate code implementation based on an optimization idea.
        
        Args:
            prompt: Prompt with the optimization idea and context
            max_completion_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated code as string
        """
        return self.generate_text(
            prompt=prompt,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def _parse_ideas(self, response: str, num_ideas: int) -> List[str]:
        """
        Parse optimization ideas from LLM response.
        Expects ideas to be numbered or separated by blank lines.
        
        Args:
            response: Raw LLM response
            num_ideas: Expected number of ideas
            
        Returns:
            List of parsed ideas
        """
        ideas = []
        lines = response.strip().split('\n')
        
        current_idea = []
        for line in lines:
            line = line.strip()
            
            # Check if this is a new idea (numbered like "1.", "2.", etc.)
            if line and (
                line[0].isdigit() and '.' in line[:5] or
                line.startswith('Idea') or
                line.startswith('**Idea')
            ):
                # Save previous idea if exists
                if current_idea:
                    idea_text = ' '.join(current_idea).strip()
                    if idea_text:
                        ideas.append(idea_text)
                    current_idea = []
                
                # Start new idea (remove numbering)
                if line[0].isdigit():
                    # Remove "1. " or "1) " prefix
                    line = line.split('.', 1)[-1].strip()
                    line = line.split(')', 1)[-1].strip()
                
                current_idea.append(line)
            elif line:
                # Continue current idea
                current_idea.append(line)
            elif current_idea:
                # Blank line might separate ideas
                idea_text = ' '.join(current_idea).strip()
                if idea_text:
                    ideas.append(idea_text)
                current_idea = []
        
        # Don't forget the last idea
        if current_idea:
            idea_text = ' '.join(current_idea).strip()
            if idea_text:
                ideas.append(idea_text)
        
        # Return up to num_ideas
        return ideas[:num_ideas] if ideas else [response]

