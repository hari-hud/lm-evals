"""Base benchmark interface for language model evaluation."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    metrics: Dict[str, float]
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of benchmark results."""
        metrics_str = "\n".join(f"{k}: {v:.4f}" for k, v in self.metrics.items())
        return f"Benchmark Results:\n{metrics_str}"

class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""
    num_samples: Optional[int] = None
    batch_size: int = 1
    seed: Optional[int] = None
    verbose: bool = False

class Benchmark(ABC):
    """Abstract base class for all evaluation benchmarks."""
    
    def __init__(self, config: Optional[Union[Dict[str, Any], BenchmarkConfig]] = None):
        """Initialize benchmark with optional configuration.
        
        Args:
            config: Configuration for benchmark execution
        """
        self.config = BenchmarkConfig(**(config or {}))
    
    @abstractmethod
    def run(self, model: "BaseModel") -> BenchmarkResult:
        """Run the benchmark evaluation on the given model.
        
        Args:
            model: Language model to evaluate
            
        Returns:
            Benchmark results including metrics and optional details
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this benchmark."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of what this benchmark evaluates."""
        pass
    
    @property
    def metrics(self) -> List[str]:
        """Get list of metrics produced by this benchmark."""
        return [] 
