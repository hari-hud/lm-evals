"""Integration tests for end-to-end evaluation workflow."""
import os
import pytest
from unittest.mock import patch

from lm_evals import Evaluator
from lm_evals.models.openai import OpenAIModel
from lm_evals.benchmarks.judge import ResponseQualityBenchmark


@pytest.fixture
def mock_openai_responses():
    """Mock OpenAI API responses."""
    with patch("openai.OpenAI") as mock_client:
        # Mock response for model being evaluated
        eval_response = mock_client.return_value.chat.completions.create
        eval_response.return_value.choices = [
            mock_client.return_value.types.chat.ChatCompletionMessage(
                content="This is a test response that explains a concept clearly and accurately."
            )
        ]
        
        # Mock response for judge model
        judge_response = mock_client.return_value.chat.completions.create
        judge_response.return_value.choices = [
            mock_client.return_value.types.chat.ChatCompletionMessage(
                content="""
Relevance: 8
Reasoning: The response is directly relevant to the task.

Accuracy: 9
Reasoning: The information provided is accurate and well-explained.

Clarity: 8
Reasoning: The explanation is clear and easy to understand.

Overall Assessment: High quality response that effectively addresses the task.
"""
            )
        ]
        
        yield mock_client


@pytest.mark.integration
def test_end_to_end_evaluation(mock_openai_responses):
    """Test complete evaluation workflow."""
    # Initialize models
    judge_model = OpenAIModel(
        model="gpt-4",
        api_key="test-key",
        temperature=0.3
    )
    
    model_to_evaluate = OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="test-key",
        temperature=0.7
    )
    
    # Initialize benchmark
    benchmark = ResponseQualityBenchmark(judge_model=judge_model)
    
    # Run evaluation
    evaluator = Evaluator(model=model_to_evaluate)
    results = evaluator.evaluate(benchmark)
    
    # Verify results structure
    assert results.name == "response_quality"
    assert "mean_score" in results.metrics
    assert "min_score" in results.metrics
    assert "max_score" in results.metrics
    
    # Verify scores are in expected range
    assert 0 <= results.metrics["mean_score"] <= 10
    assert 0 <= results.metrics["min_score"] <= 10
    assert 0 <= results.metrics["max_score"] <= 10
    
    # Verify detailed results
    assert results.details is not None
    for task_id, task_results in results.details.items():
        assert "task" in task_results
        assert "response" in task_results
        assert "evaluation" in task_results
        
        evaluation = task_results["evaluation"]
        assert "scores" in evaluation
        assert "individual_judgments" in evaluation
        assert "final_score" in evaluation


@pytest.mark.integration
def test_evaluation_with_real_api():
    """Test evaluation with real OpenAI API (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    
    # Initialize models with real API key
    judge_model = OpenAIModel(
        model="gpt-4",
        api_key=api_key,
        temperature=0.3
    )
    
    model_to_evaluate = OpenAIModel(
        model="gpt-3.5-turbo",
        api_key=api_key,
        temperature=0.7
    )
    
    # Initialize benchmark with minimal configuration
    benchmark = ResponseQualityBenchmark(
        judge_model=judge_model,
        config={
            "num_judges": 1,  # Minimize API calls
            "criteria": [
                {
                    "name": "Overall Quality",
                    "description": "Overall quality of the response",
                    "scoring_guide": "0=poor, 10=excellent"
                }
            ]
        }
    )
    
    # Run evaluation
    evaluator = Evaluator(model=model_to_evaluate)
    results = evaluator.evaluate(benchmark)
    
    # Basic validation of results
    assert results.name == "response_quality"
    assert "mean_score" in results.metrics
    assert 0 <= results.metrics["mean_score"] <= 10


@pytest.mark.integration
def test_error_handling_integration(mock_openai_responses):
    """Test error handling in integration context."""
    # Simulate API error
    mock_openai_responses.return_value.chat.completions.create.side_effect = Exception(
        "API Error"
    )
    
    judge_model = OpenAIModel(
        model="gpt-4",
        api_key="test-key"
    )
    
    model_to_evaluate = OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="test-key"
    )
    
    benchmark = ResponseQualityBenchmark(judge_model=judge_model)
    evaluator = Evaluator(model=model_to_evaluate)
    
    with pytest.raises(Exception) as exc_info:
        evaluator.evaluate(benchmark)
    assert "API Error" in str(exc_info.value) 
