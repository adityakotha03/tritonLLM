"""
Kernel Bank.
Maintains a collection of top-performing kernels and prevents duplicate evaluations.
"""

from typing import List, Dict, Optional, Set
from src.search_config import KernelResult


class KernelBank:
    """
    Manages a collection of kernels, tracking the best performers and avoiding duplicates.
    """
    
    def __init__(self, top_k: int = 5):
        """
        Initialize the kernel bank.
        
        Args:
            top_k: Number of top kernels to maintain
        """
        self.top_k = top_k
        self.all_kernels: List[KernelResult] = []
        self.seen_hashes: Set[str] = set()
        self.kernels_by_round: Dict[int, List[KernelResult]] = {}
    
    def add_kernel(self, kernel: KernelResult) -> bool:
        """
        Add a kernel to the bank.
        
        Args:
            kernel: KernelResult to add
            
        Returns:
            True if added (not a duplicate), False if duplicate
        """
        # Check for duplicate
        if kernel.code_hash and kernel.code_hash in self.seen_hashes:
            return False
        
        # Add to collections
        self.all_kernels.append(kernel)
        if kernel.code_hash:
            self.seen_hashes.add(kernel.code_hash)
        
        # Track by round
        if kernel.round_number not in self.kernels_by_round:
            self.kernels_by_round[kernel.round_number] = []
        self.kernels_by_round[kernel.round_number].append(kernel)
        
        return True
    
    def add_kernels(self, kernels: List[KernelResult]) -> int:
        """
        Add multiple kernels to the bank.
        
        Args:
            kernels: List of KernelResult objects
            
        Returns:
            Number of kernels successfully added (excluding duplicates)
        """
        added = 0
        for kernel in kernels:
            if self.add_kernel(kernel):
                added += 1
        return added
    
    def get_top_k_kernels(self, k: Optional[int] = None) -> List[KernelResult]:
        """
        Get the top-k performing kernels.
        
        Args:
            k: Number of kernels to return (uses self.top_k if None)
            
        Returns:
            List of top-k kernels sorted by speedup (descending)
        """
        if k is None:
            k = self.top_k
        
        # Filter for correct and compiled kernels
        valid_kernels = [
            k for k in self.all_kernels
            if k.is_correct() and k.is_compiled()
        ]
        
        # Sort by speedup (descending)
        sorted_kernels = sorted(
            valid_kernels,
            key=lambda x: x.get_speedup(),
            reverse=True
        )
        
        return sorted_kernels[:k]
    
    def get_best_kernel(self) -> Optional[KernelResult]:
        """
        Get the single best kernel.
        
        Returns:
            Best performing kernel, or None if no valid kernels
        """
        top_kernels = self.get_top_k_kernels(k=1)
        return top_kernels[0] if top_kernels else None
    
    def get_kernels_by_round(self, round_number: int) -> List[KernelResult]:
        """
        Get all kernels from a specific round.
        
        Args:
            round_number: Round number
            
        Returns:
            List of kernels from that round
        """
        return self.kernels_by_round.get(round_number, [])
    
    def get_all_kernels(self) -> List[KernelResult]:
        """
        Get all kernels in the bank.
        
        Returns:
            List of all kernels
        """
        return self.all_kernels
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the kernel bank.
        
        Returns:
            Dictionary with statistics
        """
        total_kernels = len(self.all_kernels)
        compiled_kernels = sum(1 for k in self.all_kernels if k.is_compiled())
        correct_kernels = sum(1 for k in self.all_kernels if k.is_correct())
        
        best_kernel = self.get_best_kernel()
        best_speedup = best_kernel.get_speedup() if best_kernel else 0.0
        
        return {
            "total_kernels": total_kernels,
            "compiled_kernels": compiled_kernels,
            "correct_kernels": correct_kernels,
            "best_speedup": best_speedup,
            "unique_hashes": len(self.seen_hashes),
            "rounds_tracked": len(self.kernels_by_round)
        }
    
    def get_top_kernels_info(self, k: Optional[int] = None) -> List[Dict]:
        """
        Get information about top-k kernels for prompt generation.
        
        Args:
            k: Number of kernels (uses self.top_k if None)
            
        Returns:
            List of dictionaries with kernel info
        """
        top_kernels = self.get_top_k_kernels(k)
        
        kernel_infos = []
        for kernel in top_kernels:
            info = {
                "kernel_id": kernel.kernel_id,
                "speedup": f"{kernel.get_speedup():.2f}",
                "runtime": f"{kernel.get_runtime():.4f}",
                "idea": kernel.metadata.get("idea_text", "Unknown"),
                "round": kernel.round_number
            }
            kernel_infos.append(info)
        
        return kernel_infos
    
    def is_duplicate(self, code_hash: str) -> bool:
        """
        Check if a code hash has been seen before.
        
        Args:
            code_hash: Hash of the code
            
        Returns:
            True if duplicate, False otherwise
        """
        return code_hash in self.seen_hashes
    
    def get_round_summary(self, round_number: int) -> Dict:
        """
        Get summary statistics for a specific round.
        
        Args:
            round_number: Round number
            
        Returns:
            Dictionary with round statistics
        """
        round_kernels = self.get_kernels_by_round(round_number)
        
        if not round_kernels:
            return {
                "round": round_number,
                "total": 0,
                "compiled": 0,
                "correct": 0,
                "best_speedup": 0.0
            }
        
        compiled = sum(1 for k in round_kernels if k.is_compiled())
        correct = sum(1 for k in round_kernels if k.is_correct())
        
        valid_speedups = [k.get_speedup() for k in round_kernels if k.is_correct()]
        best_speedup = max(valid_speedups) if valid_speedups else 0.0
        
        return {
            "round": round_number,
            "total": len(round_kernels),
            "compiled": compiled,
            "correct": correct,
            "best_speedup": best_speedup
        }

