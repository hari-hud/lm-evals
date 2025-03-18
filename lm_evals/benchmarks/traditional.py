"""Traditional language model benchmarks."""
from typing import Dict, List, Optional

from datasets import load_dataset

from lm_evals.core.benchmark import Benchmark, BenchmarkResult
from lm_evals.core.model import BaseModel

class MMLU(Benchmark):
    """Massive Multitask Language Understanding benchmark."""
    
    def __init__(self, subjects: Optional[List[str]] = None, **kwargs):
        """Initialize MMLU benchmark.
        
        Args:
            subjects: List of MMLU subjects to evaluate. If None, uses all subjects.
            **kwargs: Additional configuration passed to parent
        """
        super().__init__(**kwargs)
        self.subjects = subjects
        
    def run(self, model: BaseModel) -> BenchmarkResult:
        """Run MMLU evaluation.
        
        Args:
            model: Language model to evaluate
            
        Returns:
            Benchmark results including accuracy per subject
        """
        # Load MMLU dataset
        dataset = load_dataset("cais/mmlu", "all")["test"]
        
        if self.subjects:
            dataset = dataset.filter(lambda x: x["subject"] in self.subjects)
            
        # Format prompts
        prompts = []
        answers = []
        for example in dataset:
            prompt = (
                f"Question: {example['question']}\n"
                f"A) {example['choices'][0]}\n"
                f"B) {example['choices'][1]}\n"
                f"C) {example['choices'][2]}\n"
                f"D) {example['choices'][3]}\n"
                "Answer:"
            )
            prompts.append(prompt)
            answers.append(example["answer"])
            
        # Get model predictions
        predictions = []
        for prompt in prompts:
            pred = model.generate(prompt, max_tokens=1).strip()
            predictions.append(pred)
            
        # Calculate metrics
        correct = sum(p == a for p, a in zip(predictions, answers))
        accuracy = correct / len(predictions)
        
        # Calculate per-subject accuracy
        subject_metrics = {}
        for example, pred in zip(dataset, predictions):
            subject = example["subject"]
            if subject not in subject_metrics:
                subject_metrics[subject] = {"correct": 0, "total": 0}
            subject_metrics[subject]["total"] += 1
            if pred == example["answer"]:
                subject_metrics[subject]["correct"] += 1
                
        for subject in subject_metrics:
            acc = subject_metrics[subject]["correct"] / subject_metrics[subject]["total"]
            subject_metrics[subject] = acc
            
        return BenchmarkResult(
            metrics={"accuracy": accuracy},
            details={"subject_accuracy": subject_metrics}
        )
    
    @property
    def name(self) -> str:
        return "MMLU"
    
    @property
    def description(self) -> str:
        return "Massive Multitask Language Understanding benchmark testing knowledge across 57 subjects"

class TruthfulQA(Benchmark):
    """TruthfulQA benchmark for measuring model truthfulness."""
    
    def run(self, model: BaseModel) -> BenchmarkResult:
        """Run TruthfulQA evaluation.
        
        Args:
            model: Language model to evaluate
            
        Returns:
            Benchmark results
        """
        # Load TruthfulQA dataset
        dataset = load_dataset("truthful_qa", "multiple_choice")["validation"]
        
        # Format prompts
        prompts = []
        correct_answers = []
        for example in dataset:
            prompt = f"Q: {example['question']}\nA:"
            prompts.append(prompt)
            correct_answers.append(example["correct_answers"])
            
        # Get model predictions
        predictions = []
        for prompt in prompts:
            pred = model.generate(prompt, max_tokens=100).strip()
            predictions.append(pred)
            
        # Calculate metrics using provided scoring function
        from truthfulqa import grade
        scores = grade(predictions, correct_answers)
        
        return BenchmarkResult(
            metrics={
                "truth_score": scores["truth"],
                "info_score": scores["info"]
            }
        )
    
    @property
    def name(self) -> str:
        return "TruthfulQA"
    
    @property
    def description(self) -> str:
        return "Benchmark for measuring model truthfulness and resistance to falsehoods" 
