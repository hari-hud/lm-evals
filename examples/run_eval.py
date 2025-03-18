"""Example script demonstrating usage of the LM-Evals framework."""
import os
from typing import Any, List

from lm_evals import Evaluator
from lm_evals.benchmarks import MMLU, TruthfulQA
from lm_evals.benchmarks.rag import RAGSystem, ContextRelevance, AnswerFaithfulness
from lm_evals.benchmarks.agent import AgentSystem, TaskCompletion, ToolUsage
from lm_evals.models.openai import OpenAIModel

def run_traditional_benchmarks(model: OpenAIModel):
    """Run traditional LLM benchmarks."""
    print("\n=== Running Traditional Benchmarks ===")
    
    # Initialize evaluator
    evaluator = Evaluator(model=model)
    
    # Run MMLU benchmark
    mmlu = MMLU(subjects=["mathematics", "physics", "chemistry"])
    mmlu_results = evaluator.evaluate(mmlu)
    print("\nMMLU Results:")
    print(mmlu_results)
    
    # Run TruthfulQA benchmark
    truthfulqa = TruthfulQA()
    truthfulqa_results = evaluator.evaluate(truthfulqa)
    print("\nTruthfulQA Results:")
    print(truthfulqa_results)

class ExampleRAGSystem(RAGSystem):
    """Example RAG system implementation."""
    
    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Mock retrieval implementation."""
        # In a real system, this would search a document store
        return [
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris.",
            "France is a country in Europe."
        ]
    
    def generate(self, query: str, contexts: List[str]) -> str:
        """Mock generation implementation."""
        # In a real system, this would use an LLM
        return "Paris is the capital of France and home to the Eiffel Tower."

def run_rag_benchmarks(model: OpenAIModel):
    """Run RAG system benchmarks."""
    print("\n=== Running RAG Benchmarks ===")
    
    # Initialize RAG system and evaluator
    rag = ExampleRAGSystem()
    evaluator = Evaluator(model=model)
    
    # Run context relevance benchmark
    relevance = ContextRelevance(rag_system=rag)
    relevance_results = evaluator.evaluate(relevance)
    print("\nContext Relevance Results:")
    print(relevance_results)
    
    # Run answer faithfulness benchmark
    faithfulness = AnswerFaithfulness(rag_system=rag)
    faithfulness_results = evaluator.evaluate(faithfulness)
    print("\nAnswer Faithfulness Results:")
    print(faithfulness_results)

class ExampleAgentSystem(AgentSystem):
    """Example agent system implementation."""
    
    def plan(self, task: str) -> List[str]:
        """Mock planning implementation."""
        return [
            "1. Understand the task",
            "2. Identify required tools",
            "3. Execute steps in sequence"
        ]
    
    def execute(self, task: str, plan: List[str]) -> dict:
        """Mock execution implementation."""
        return {
            "success": True,
            "used_tools": ["calculator"],
            "output": "Task completed successfully"
        }
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Mock tool usage implementation."""
        return {"status": "success", "result": "Tool output"}

def run_agent_benchmarks(model: OpenAIModel):
    """Run agent system benchmarks."""
    print("\n=== Running Agent Benchmarks ===")
    
    # Initialize agent system and evaluator
    agent = ExampleAgentSystem()
    evaluator = Evaluator(model=model)
    
    # Run task completion benchmark
    completion = TaskCompletion(agent_system=agent)
    completion_results = evaluator.evaluate(completion)
    print("\nTask Completion Results:")
    print(completion_results)
    
    # Run tool usage benchmark
    tool_usage = ToolUsage(agent_system=agent)
    tool_usage_results = evaluator.evaluate(tool_usage)
    print("\nTool Usage Results:")
    print(tool_usage_results)

def main():
    """Run example evaluation pipeline."""
    # Initialize OpenAI model
    model = OpenAIModel(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Run benchmarks
    run_traditional_benchmarks(model)
    run_rag_benchmarks(model)
    run_agent_benchmarks(model)

if __name__ == "__main__":
    main() 
