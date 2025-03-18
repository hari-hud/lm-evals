"""Example script demonstrating usage of NVIDIA NIM models with LM-Evals."""
import os
from typing import List

from lm_evals import Evaluator
from lm_evals.benchmarks import MMLU, TruthfulQA
from lm_evals.models.nvidia import NIMModel

def main():
    """Run example evaluation pipeline with NVIDIA NIM model."""
    # Initialize NVIDIA NIM model
    model = NIMModel(
        model_name="llama2-70b",  # Name of your deployed model
        endpoint_url="https://your-nim-endpoint.com",  # Your NIM endpoint URL
        api_key=os.getenv("NVIDIA_API_KEY"),  # Optional API key
        max_tokens=1000,
        temperature=0.7
    )
    
    # Initialize evaluator
    evaluator = Evaluator(model=model)
    
    # Run MMLU benchmark
    print("\n=== Running MMLU Benchmark ===")
    mmlu = MMLU(subjects=["mathematics", "physics", "chemistry"])
    mmlu_results = evaluator.evaluate(mmlu)
    print("\nMMLU Results:")
    print(mmlu_results)
    
    # Run TruthfulQA benchmark
    print("\n=== Running TruthfulQA Benchmark ===")
    truthfulqa = TruthfulQA()
    truthfulqa_results = evaluator.evaluate(truthfulqa)
    print("\nTruthfulQA Results:")
    print(truthfulqa_results)

if __name__ == "__main__":
    main() 
