"""Metrics for evaluating language model outputs."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute metric score for predictions against references.
        
        Args:
            predictions: Model generated outputs
            references: Ground truth references
            
        Returns:
            Metric score
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this metric."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Get a description of what this metric measures."""
        pass

class ExactMatch(Metric):
    """Exact string matching metric."""
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute exact match accuracy."""
        matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        return matches / len(predictions) if predictions else 0.0
    
    @property
    def name(self) -> str:
        return "exact_match"
    
    @property
    def description(self) -> str:
        return "Fraction of predictions that exactly match references"

class F1Score(Metric):
    """F1 score based on token overlap."""
    
    def _get_tokens(self, text: str) -> set:
        """Get set of tokens from text."""
        return set(text.lower().split())
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute average F1 score."""
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = self._get_tokens(pred)
            ref_tokens = self._get_tokens(ref)
            
            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
                continue
                
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue
            
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
                
        return np.mean(f1_scores)
    
    @property
    def name(self) -> str:
        return "f1_score"
    
    @property 
    def description(self) -> str:
        return "F1 score based on token overlap between predictions and references"

class ROUGE(Metric):
    """ROUGE-L metric for measuring sequence overlap."""
    
    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
    
    def compute(self, predictions: List[str], references: List[str]) -> float:
        """Compute average ROUGE-L score."""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
                
            lcs = self._lcs_length(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref_tokens)
            
            if precision + recall == 0:
                scores.append(0.0)
            else:
                rouge_l = 2 * (precision * recall) / (precision + recall)
                scores.append(rouge_l)
                
        return np.mean(scores)
    
    @property
    def name(self) -> str:
        return "rouge_l"
    
    @property
    def description(self) -> str:
        return "ROUGE-L score measuring longest common subsequence between predictions and references" 
