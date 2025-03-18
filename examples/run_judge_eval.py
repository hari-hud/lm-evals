"""Example script demonstrating usage of LLM-as-judge evaluation."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from lm_evals import Evaluator
from lm_evals.benchmarks.judge import (
    JudgingCriteria,
    JudgeBenchmarkConfig,
    ResponseQualityBenchmark
)
from lm_evals.models.openai import OpenAIModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Optional job ID for tracking"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4",
        help="Model to use as judge"
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--num_judges",
        type=int,
        default=3,
        help="Number of judges for consensus"
    )
    return parser.parse_args()


def setup_output_dir(base_dir: str, job_id: str = None) -> Path:
    """Set up output directory for results.
    
    Args:
        base_dir: Base directory for results
        job_id: Optional job ID for tracking
        
    Returns:
        Path to output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"eval_{timestamp}"
    if job_id:
        dir_name += f"_job{job_id}"
    
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Run example evaluation using LLM-as-judge."""
    args = parse_args()
    
    # Set up output directory
    output_dir = setup_output_dir(args.output_dir, args.job_id)
    
    # Initialize models - one as judge, one to evaluate
    judge_model = OpenAIModel(
        model=args.judge_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3  # Lower temperature for more consistent judgments
    )
    
    model_to_evaluate = OpenAIModel(
        model=args.eval_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    
    # Create custom judging criteria
    criteria = [
        JudgingCriteria(
            name="Scientific Accuracy",
            description="Accuracy of scientific concepts and explanations",
            scoring_guide="0=completely incorrect, 10=scientifically accurate and precise",
            weight=1.5  # Giving higher weight to accuracy
        ),
        JudgingCriteria(
            name="Clarity",
            description="Clarity and accessibility of explanation",
            scoring_guide="0=very unclear, 10=crystal clear and easy to understand",
            weight=1.0
        ),
        JudgingCriteria(
            name="Completeness",
            description="Coverage of important aspects and details",
            scoring_guide="0=very incomplete, 10=comprehensive coverage",
            weight=1.0
        )
    ]
    
    # Configure the benchmark
    config = JudgeBenchmarkConfig(
        criteria=criteria,
        num_judges=args.num_judges,
        aggregation_method="weighted_average"
    )
    
    # Initialize benchmark
    benchmark = ResponseQualityBenchmark(
        judge_model=judge_model,
        config=config
    )
    
    # Run evaluation
    print("\n=== Running LLM-as-Judge Evaluation ===")
    print(f"Judge Model: {args.judge_model}")
    print(f"Evaluated Model: {args.eval_model}")
    print(f"Number of Judges: {args.num_judges}")
    print(f"Results will be saved to: {output_dir}\n")
    
    evaluator = Evaluator(model=model_to_evaluate)
    results = evaluator.evaluate(benchmark)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": {
                    "judge_model": args.judge_model,
                    "eval_model": args.eval_model,
                    "num_judges": args.num_judges
                },
                "metrics": results.metrics,
                "details": results.details
            },
            f,
            indent=2
        )
    
    # Print summary
    print("\nOverall Results:")
    print(f"Mean Score: {results.metrics['mean_score']:.2f}")
    print(f"Min Score: {results.metrics['min_score']:.2f}")
    print(f"Max Score: {results.metrics['max_score']:.2f}")
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main() 
