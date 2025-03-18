"""
LM-Evals: A unified framework for evaluating language models, RAG systems, and AI agents.
"""

__version__ = "0.1.0"

from lm_evals.core.evaluator import Evaluator
from lm_evals.core.model import BaseModel
from lm_evals.core.benchmark import Benchmark
from lm_evals.core.metrics import Metric

__all__ = ["Evaluator", "BaseModel", "Benchmark", "Metric"] 
