"""Main evaluator class for running benchmarks."""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from lm_evals.core.benchmark import Benchmark, BenchmarkResult
from lm_evals.core.model import BaseModel as LMModel

class EvaluatorConfig(BaseModel):
    """Configuration for the evaluator."""
    parallel: bool = False
    num_workers: int = 1
    cache_results: bool = True
    output_path: Optional[str] = None

class Evaluator:
    """Main class for evaluating language models on benchmarks."""
    
    def __init__(
        self,
        model: LMModel,
        config: Optional[Union[Dict[str, Any], EvaluatorConfig]] = None
    ):
        """Initialize evaluator with model and optional configuration.
        
        Args:
            model: Language model to evaluate
            config: Evaluator configuration
        """
        self.model = model
        self.config = EvaluatorConfig(**(config or {}))
        self._results_cache = {}
        
    def evaluate(
        self,
        benchmark: Union[Benchmark, List[Benchmark]],
        **kwargs
    ) -> Union[BenchmarkResult, Dict[str, BenchmarkResult]]:
        """Run evaluation on specified benchmark(s).
        
        Args:
            benchmark: Single benchmark or list of benchmarks to run
            **kwargs: Additional arguments passed to benchmarks
            
        Returns:
            Evaluation results for each benchmark
        """
        if isinstance(benchmark, list):
            return self._evaluate_multiple(benchmark, **kwargs)
        return self._evaluate_single(benchmark, **kwargs)
    
    def _evaluate_single(
        self,
        benchmark: Benchmark,
        **kwargs
    ) -> BenchmarkResult:
        """Run evaluation on a single benchmark.
        
        Args:
            benchmark: Benchmark to evaluate
            **kwargs: Additional arguments passed to benchmark
            
        Returns:
            Benchmark results
        """
        # Check cache
        cache_key = (benchmark.name, str(kwargs))
        if self.config.cache_results and cache_key in self._results_cache:
            return self._results_cache[cache_key]
        
        # Run benchmark
        result = benchmark.run(self.model)
        
        # Cache results
        if self.config.cache_results:
            self._results_cache[cache_key] = result
            
        return result
    
    def _evaluate_multiple(
        self,
        benchmarks: List[Benchmark],
        **kwargs
    ) -> Dict[str, BenchmarkResult]:
        """Run evaluation on multiple benchmarks.
        
        Args:
            benchmarks: List of benchmarks to evaluate
            **kwargs: Additional arguments passed to benchmarks
            
        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}
        for benchmark in benchmarks:
            results[benchmark.name] = self._evaluate_single(benchmark, **kwargs)
        return results
    
    def summarize(
        self,
        results: Union[BenchmarkResult, Dict[str, BenchmarkResult]]
    ) -> str:
        """Generate human-readable summary of evaluation results.
        
        Args:
            results: Results from evaluate() to summarize
            
        Returns:
            Formatted summary string
        """
        if isinstance(results, dict):
            # Multiple benchmark results
            summary = []
            for name, result in results.items():
                summary.append(f"\n=== {name} ===")
                summary.append(str(result))
            return "\n".join(summary)
        
        # Single benchmark result
        return str(results) 
