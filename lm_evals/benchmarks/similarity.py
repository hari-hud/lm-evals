"""Benchmark for evaluating text similarity metrics."""
from typing import Dict, List, Optional, Union
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from datasets import Dataset, load_dataset

from rouge_score import rouge_scorer
from bert_score import BERTScorer
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from ..core.benchmark import Benchmark, BenchmarkResult
from ..core.model import BaseModel as LMModel

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass


class SimilarityMetricsConfig(BaseModel):
    """Configuration for similarity metrics benchmark."""
    metrics: List[str] = Field(
        default=["rouge", "bleu", "bert_score", "exact_match"],
        description="List of metrics to compute"
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to custom dataset (CSV/JSON with 'input' and 'reference' columns)"
    )
    predictions_path: Optional[str] = Field(
        default=None,
        description="Path to pre-generated predictions (JSON with 'input' and 'output' fields)"
    )
    reference_path: Optional[str] = Field(
        default=None,
        description="Path to reference answers (JSON with 'input' and 'reference' fields)"
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for model inference"
    )


class SimilarityMetricsBenchmark(Benchmark):
    """Benchmark for evaluating text similarity between model outputs and references."""

    def __init__(
        self,
        config: Optional[SimilarityMetricsConfig] = None,
        **kwargs
    ):
        """Initialize the benchmark."""
        super().__init__(**kwargs)
        self.config = config or SimilarityMetricsConfig()
        
        # Initialize metric calculators
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.exact_match = evaluate.load("exact_match")
        self.smooth = SmoothingFunction().method1

    def load_data(self) -> Dataset:
        """Load evaluation data from specified source."""
        if self.config.dataset_path:
            path = Path(self.config.dataset_path)
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            elif path.suffix == '.json':
                df = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported dataset format: {path.suffix}")
            
            return Dataset.from_pandas(df)
        
        elif self.config.predictions_path and self.config.reference_path:
            with open(self.config.predictions_path) as f:
                predictions = json.load(f)
            with open(self.config.reference_path) as f:
                references = json.load(f)
                
            # Merge predictions and references
            data = []
            for pred, ref in zip(predictions, references):
                assert pred['input'] == ref['input'], "Mismatch between prediction and reference inputs"
                data.append({
                    'input': pred['input'],
                    'output': pred['output'],
                    'reference': ref['reference']
                })
            return Dataset.from_list(data)
        
        else:
            # Use default test dataset
            return load_dataset("hendrycks_ethics", "cm", split="test")

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute all specified similarity metrics."""
        metrics = {}
        
        if "rouge" in self.config.metrics:
            rouge_scores = [self.rouge.score(ref, pred) for ref, pred in zip(references, predictions)]
            metrics.update({
                'rouge1_f': sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores),
                'rouge2_f': sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores),
                'rougeL_f': sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
            })
        
        if "bleu" in self.config.metrics:
            bleu_scores = []
            for ref, pred in zip(references, predictions):
                ref_tokens = nltk.word_tokenize(ref)
                pred_tokens = nltk.word_tokenize(pred)
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smooth)
                bleu_scores.append(score)
            metrics['bleu'] = sum(bleu_scores) / len(bleu_scores)
        
        if "bert_score" in self.config.metrics:
            P, R, F1 = self.bert_scorer.score(predictions, references)
            metrics.update({
                'bert_score_precision': P.mean().item(),
                'bert_score_recall': R.mean().item(),
                'bert_score_f1': F1.mean().item()
            })
        
        if "exact_match" in self.config.metrics:
            em_results = self.exact_match.compute(
                predictions=predictions,
                references=references
            )
            metrics['exact_match'] = em_results['exact_match']
        
        return metrics

    def run(self, model: Optional[LMModel] = None) -> BenchmarkResult:
        """Run the benchmark on the specified model."""
        dataset = self.load_data()
        
        # If using pre-generated predictions
        if self.config.predictions_path and self.config.reference_path:
            predictions = dataset['output']
            references = dataset['reference']
        else:
            if model is None:
                raise ValueError("Model must be provided when not using pre-generated predictions")
            
            # Generate predictions
            predictions = []
            for i in range(0, len(dataset), self.config.batch_size):
                batch = dataset[i:i + self.config.batch_size]
                batch_predictions = [
                    model.generate(input_text) for input_text in batch['input']
                ]
                predictions.extend(batch_predictions)
            
            references = dataset['reference']
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, references)
        
        # Prepare detailed results
        details = {
            str(i): {
                'input': input_text,
                'prediction': pred,
                'reference': ref,
                'metrics': self.compute_metrics([pred], [ref])
            }
            for i, (input_text, pred, ref) in enumerate(
                zip(dataset['input'], predictions, references)
            )
        }
        
        return BenchmarkResult(
            name="similarity_metrics",
            metrics=metrics,
            details=details
        ) 
