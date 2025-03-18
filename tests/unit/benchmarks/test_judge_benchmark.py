"""Unit tests for LLM-as-judge benchmark implementation."""
import pytest
from unittest.mock import Mock

from lm_evals.benchmarks.judge import (
    JudgingCriteria,
    JudgeBenchmarkConfig,
    LLMJudgeBenchmark,
    ResponseQualityBenchmark
)


@pytest.fixture
def mock_judge_model():
    """Create a mock judge model."""
    model = Mock()
    model.generate.return_value = """
Relevance: 8
Reasoning: The response directly addresses the task and provides relevant information.

Accuracy: 7
Reasoning: Most information is accurate but there are minor imprecisions.

Completeness: 6
Reasoning: Covers main points but some details are missing.

Overall Assessment: Good response with room for improvement.
"""
    return model


@pytest.fixture
def benchmark(mock_judge_model):
    """Create a ResponseQualityBenchmark instance."""
    return ResponseQualityBenchmark(judge_model=mock_judge_model)


def test_judging_criteria_initialization():
    """Test JudgingCriteria initialization."""
    criteria = JudgingCriteria(
        name="Test",
        description="Test description",
        scoring_guide="0=bad, 10=good",
        weight=1.5
    )
    
    assert criteria.name == "Test"
    assert criteria.description == "Test description"
    assert criteria.scoring_guide == "0=bad, 10=good"
    assert criteria.weight == 1.5
    assert criteria.min_score == 0.0
    assert criteria.max_score == 10.0


def test_benchmark_config_initialization():
    """Test JudgeBenchmarkConfig initialization."""
    criteria = [
        JudgingCriteria(
            name="Test",
            description="Test description",
            scoring_guide="0=bad, 10=good"
        )
    ]
    
    config = JudgeBenchmarkConfig(criteria=criteria)
    assert config.criteria == criteria
    assert config.num_judges == 1
    assert config.aggregation_method == "weighted_average"


def test_parse_judge_response(benchmark):
    """Test parsing of judge response."""
    response = """
Relevance: 8
Reasoning: Good relevance.

Accuracy: 7
Reasoning: Mostly accurate.

Overall Assessment: Good response.
"""
    
    result = benchmark._parse_judge_response(response)
    assert "Relevance" in result
    assert result["Relevance"]["score"] == 8
    assert "Accuracy" in result
    assert result["Accuracy"]["score"] == 7
    assert "overall_assessment" in result
    assert result["overall_assessment"] == "Good response."


def test_aggregate_scores(benchmark):
    """Test score aggregation."""
    scores = [
        {
            "Relevance": {"score": 8},
            "Accuracy": {"score": 7}
        },
        {
            "Relevance": {"score": 9},
            "Accuracy": {"score": 8}
        }
    ]
    
    aggregated = benchmark._aggregate_scores(scores)
    assert "Relevance" in aggregated
    assert "Accuracy" in aggregated
    assert 8 <= aggregated["Relevance"] <= 9
    assert 7 <= aggregated["Accuracy"] <= 8


def test_evaluate_response(benchmark, mock_judge_model):
    """Test response evaluation."""
    result = benchmark.evaluate_response(
        task="Test task",
        response="Test response"
    )
    
    assert "scores" in result
    assert "individual_judgments" in result
    assert "final_score" in result
    assert isinstance(result["final_score"], float)


def test_run_benchmark(benchmark, mock_judge_model):
    """Test running the full benchmark."""
    model_to_evaluate = Mock()
    model_to_evaluate.generate.return_value = "Test response"
    
    results = benchmark.run(model_to_evaluate)
    
    assert results.name == "response_quality"
    assert "mean_score" in results.metrics
    assert "min_score" in results.metrics
    assert "max_score" in results.metrics
    assert results.details is not None


def test_multiple_judges(mock_judge_model):
    """Test evaluation with multiple judges."""
    benchmark = ResponseQualityBenchmark(
        judge_model=mock_judge_model,
        config=JudgeBenchmarkConfig(
            criteria=[
                JudgingCriteria(
                    name="Test",
                    description="Test description",
                    scoring_guide="0=bad, 10=good"
                )
            ],
            num_judges=3
        )
    )
    
    result = benchmark.evaluate_response(
        task="Test task",
        response="Test response"
    )
    
    assert len(result["individual_judgments"]) == 3


def test_error_handling(benchmark):
    """Test error handling in evaluation."""
    mock_judge_model = Mock()
    mock_judge_model.generate.side_effect = Exception("Model error")
    
    benchmark = ResponseQualityBenchmark(judge_model=mock_judge_model)
    
    with pytest.raises(Exception) as exc_info:
        benchmark.evaluate_response(
            task="Test task",
            response="Test response"
        )
    assert str(exc_info.value) == "Model error" 
