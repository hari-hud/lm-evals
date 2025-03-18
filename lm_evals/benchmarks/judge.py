"""LLM-as-judge benchmarks for evaluating model outputs."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from lm_evals.core.benchmark import Benchmark, BenchmarkResult
from lm_evals.core.model import BaseModel as LMModel


class JudgingCriteria(BaseModel):
    """Configuration for judging criteria."""
    name: str
    description: str
    scoring_guide: str
    min_score: float = 0.0
    max_score: float = 10.0
    weight: float = 1.0


class JudgeBenchmarkConfig(BaseModel):
    """Configuration for LLM-as-judge benchmarks."""
    criteria: List[JudgingCriteria]
    evaluation_prompt_template: str = Field(
        default=(
            "You are an impartial judge evaluating the quality of a model's response.\n"
            "Task: {task}\n"
            "Model Response: {response}\n"
            "\nPlease evaluate the response on the following criteria:\n"
            "{criteria_descriptions}\n"
            "\nFor each criterion, provide a score between {min_score} and {max_score}.\n"
            "Explain your reasoning for each score.\n"
            "Finally, provide an overall assessment.\n"
            "\nFormat your response as follows:\n"
            "Criterion 1 Name: <score>\n"
            "Reasoning: <explanation>\n"
            "\nCriterion 2 Name: <score>\n"
            "Reasoning: <explanation>\n"
            "...\n"
            "\nOverall Assessment: <text>"
        )
    )
    aggregation_method: str = "weighted_average"  # or "min", "max", "median"
    num_judges: int = 1  # Number of times to run evaluation for consensus
    reference_answer: Optional[str] = None  # Optional reference for comparison


class LLMJudgeBenchmark(Benchmark):
    """Base class for LLM-as-judge benchmarks."""
    
    def __init__(
        self,
        judge_model: LMModel,
        config: Optional[Union[Dict[str, Any], JudgeBenchmarkConfig]] = None
    ) -> None:
        """Initialize LLM-as-judge benchmark.
        
        Args:
            judge_model: Language model to use as judge
            config: Benchmark configuration
        """
        super().__init__()
        self.judge = judge_model
        self.config = (
            config if isinstance(config, JudgeBenchmarkConfig)
            else JudgeBenchmarkConfig(**(config or {}))
        )
        
    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from judge.
        
        Args:
            response: Raw response from judge
            
        Returns:
            Dictionary containing scores and reasoning
        """
        results = {}
        current_criterion = None
        current_section = None
        
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue

            # First check if this is a criterion line
            is_criterion = False
            for criterion in self.config.criteria:
                if line.lower().startswith(f"{criterion.name.lower()}:"):
                    current_criterion = criterion.name
                    try:
                        score = float(line.split(":")[-1].strip())
                        results[current_criterion] = {
                            "score": score,
                            "reasoning": ""
                        }
                    except ValueError:
                        pass
                    is_criterion = True
                    break
            
            if is_criterion:
                continue

            # If not a criterion line, check other cases
            if line.lower().startswith("reasoning:"):
                current_section = "reasoning"
            elif line.lower().startswith("overall assessment:"):
                current_criterion = None
                results["overall_assessment"] = line.split(":", 1)[1].strip()
            elif current_criterion and current_section:
                results[current_criterion][current_section] += f" {line}"
                
        return results
    
    def _aggregate_scores(
        self,
        all_scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate scores from multiple judges.
        
        Args:
            all_scores: List of score dictionaries from judges
            
        Returns:
            Aggregated scores for each criterion
        """
        aggregated = {}
        
        for criterion in self.config.criteria:
            scores = [
                scores[criterion.name]["score"]
                for scores in all_scores
                if criterion.name in scores
            ]
            
            if not scores:
                continue
                
            if self.config.aggregation_method == "weighted_average":
                aggregated[criterion.name] = (
                    np.average(scores) * criterion.weight
                )
            elif self.config.aggregation_method == "min":
                aggregated[criterion.name] = min(scores)
            elif self.config.aggregation_method == "max":
                aggregated[criterion.name] = max(scores)
            elif self.config.aggregation_method == "median":
                aggregated[criterion.name] = float(np.median(scores))
                
        return aggregated
    
    def evaluate_response(
        self,
        task: str,
        response: str,
        reference: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate a single response using the judge.
        
        Args:
            task: Task description
            response: Model response to evaluate
            reference: Optional reference answer
            
        Returns:
            Evaluation results including scores and reasoning
        """
        # Format criteria descriptions
        criteria_descriptions = "\n".join(
            f"{i+1}. {c.name}: {c.description}\n   Scoring guide: {c.scoring_guide}"
            for i, c in enumerate(self.config.criteria)
        )
        
        # Prepare evaluation prompt
        prompt = self.config.evaluation_prompt_template.format(
            task=task,
            response=response,
            reference=reference or "No reference provided",
            criteria_descriptions=criteria_descriptions,
            min_score=min(c.min_score for c in self.config.criteria),
            max_score=max(c.max_score for c in self.config.criteria)
        )
        
        # Get multiple judgments
        all_scores = []
        for _ in range(self.config.num_judges):
            judgment = self.judge.generate(prompt, max_tokens=1000)
            scores = self._parse_judge_response(judgment)
            all_scores.append(scores)
            
        # Aggregate scores
        aggregated_scores = self._aggregate_scores(all_scores)
        
        return {
            "scores": aggregated_scores,
            "individual_judgments": all_scores,
            "final_score": np.mean(list(aggregated_scores.values()))
        }


class ResponseQualityBenchmark(LLMJudgeBenchmark):
    """Benchmark for evaluating general response quality."""
    
    def __init__(self, judge_model: LMModel, **kwargs) -> None:
        """Initialize response quality benchmark."""
        default_criteria = [
            JudgingCriteria(
                name="Relevance",
                description="How well does the response address the task?",
                scoring_guide="0=completely irrelevant, 10=perfectly relevant"
            ),
            JudgingCriteria(
                name="Accuracy",
                description="Is the information provided accurate and factual?",
                scoring_guide="0=completely incorrect, 10=completely accurate"
            ),
            JudgingCriteria(
                name="Completeness",
                description="How thoroughly does the response cover the topic?",
                scoring_guide="0=very incomplete, 10=fully comprehensive"
            ),
            JudgingCriteria(
                name="Clarity",
                description="How clear and well-structured is the response?",
                scoring_guide="0=very unclear, 10=crystal clear"
            )
        ]
        
        super().__init__(
            judge_model=judge_model,
            config=JudgeBenchmarkConfig(criteria=default_criteria, **kwargs)
        )
    
    def run(self, model: LMModel) -> BenchmarkResult:
        """Run the benchmark evaluation.
        
        Args:
            model: Language model to evaluate
            
        Returns:
            Benchmark results
        """
        # Example evaluation tasks
        eval_tasks = [
            {
                "task": "Explain how photosynthesis works.",
                "reference": (
                    "Photosynthesis is the process by which plants convert "
                    "sunlight, water, and carbon dioxide into glucose and oxygen. "
                    "The process occurs in the chloroplasts, specifically using "
                    "chlorophyll in the thylakoids. It consists of light-dependent "
                    "and light-independent reactions..."
                )
            },
            {
                "task": "What are the key principles of machine learning?",
                "reference": None
            },
            {
                "task": "Describe the water cycle on Earth.",
                "reference": None
            }
        ]
        
        results = []
        for task in eval_tasks:
            # Get model response
            response = model.generate(task["task"], max_tokens=1000)
            
            # Evaluate response
            evaluation = self.evaluate_response(
                task["task"],
                response,
                task.get("reference")
            )
            
            results.append({
                "task": task["task"],
                "response": response,
                "evaluation": evaluation
            })
            
        # Calculate aggregate metrics
        all_scores = [r["evaluation"]["final_score"] for r in results]
        metrics = {
            "mean_score": float(np.mean(all_scores)),
            "min_score": float(np.min(all_scores)),
            "max_score": float(np.max(all_scores)),
            "std_score": float(np.std(all_scores))
        }
        
        return BenchmarkResult(
            name=self.name,
            metrics=metrics,
            details={
                f"task_{i}": result
                for i, result in enumerate(results)
            }
        )

    @property
    def name(self) -> str:
        """Get benchmark name."""
        return "response_quality"

    @property
    def description(self) -> str:
        """Get benchmark description."""
        return "Evaluates the general quality of model responses using LLM-as-judge." 
