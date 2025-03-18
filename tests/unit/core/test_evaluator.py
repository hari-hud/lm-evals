"""Unit tests for the Evaluator class."""
import pytest
from unittest.mock import Mock

from lm_evals.core.evaluator import Evaluator
from lm_evals.core.benchmark import Benchmark, BenchmarkResult


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return Mock()


@pytest.fixture
def mock_benchmark():
    """Create a mock benchmark."""
    benchmark = Mock(spec=Benchmark)
    benchmark.run.return_value = BenchmarkResult(
        name="test_benchmark",
        metrics={
            "score": 0.8,
            "accuracy": 0.75
        },
        details={
            "task_1": {"result": "success"},
            "task_2": {"result": "partial"}
        }
    )
    return benchmark


def test_evaluator_initialization(mock_model):
    """Test Evaluator initialization."""
    evaluator = Evaluator(model=mock_model)
    assert evaluator.model == mock_model


def test_evaluate_single_benchmark(mock_model, mock_benchmark):
    """Test evaluating a single benchmark."""
    evaluator = Evaluator(model=mock_model)
    result = evaluator.evaluate(mock_benchmark)
    
    assert isinstance(result, BenchmarkResult)
    assert result.name == "test_benchmark"
    assert result.metrics["score"] == 0.8
    assert result.metrics["accuracy"] == 0.75
    assert len(result.details) == 2
    
    mock_benchmark.run.assert_called_once_with(mock_model)


def test_evaluate_multiple_benchmarks(mock_model):
    """Test evaluating multiple benchmarks."""
    benchmark1 = Mock(spec=Benchmark)
    benchmark1.run.return_value = BenchmarkResult(
        name="benchmark1",
        metrics={"score": 0.8},
        details={}
    )
    
    benchmark2 = Mock(spec=Benchmark)
    benchmark2.run.return_value = BenchmarkResult(
        name="benchmark2",
        metrics={"score": 0.9},
        details={}
    )
    
    evaluator = Evaluator(model=mock_model)
    results = evaluator.evaluate_multiple([benchmark1, benchmark2])
    
    assert len(results) == 2
    assert results[0].name == "benchmark1"
    assert results[1].name == "benchmark2"
    assert results[0].metrics["score"] == 0.8
    assert results[1].metrics["score"] == 0.9


def test_evaluate_with_invalid_benchmark(mock_model):
    """Test evaluating with invalid benchmark."""
    invalid_benchmark = Mock()  # Not a Benchmark instance
    
    evaluator = Evaluator(model=mock_model)
    with pytest.raises(TypeError):
        evaluator.evaluate(invalid_benchmark)


def test_evaluate_with_failing_benchmark(mock_model):
    """Test evaluating with a failing benchmark."""
    failing_benchmark = Mock(spec=Benchmark)
    failing_benchmark.run.side_effect = Exception("Benchmark failed")
    
    evaluator = Evaluator(model=mock_model)
    with pytest.raises(Exception) as exc_info:
        evaluator.evaluate(failing_benchmark)
    assert str(exc_info.value) == "Benchmark failed"


@pytest.mark.asyncio
async def test_evaluate_async(mock_model, mock_benchmark):
    """Test asynchronous evaluation."""
    evaluator = Evaluator(model=mock_model)
    result = await evaluator.aevaluate(mock_benchmark)
    
    assert isinstance(result, BenchmarkResult)
    assert result.name == "test_benchmark"
    assert result.metrics["score"] == 0.8


def test_evaluator_str_representation(mock_model):
    """Test string representation of Evaluator."""
    evaluator = Evaluator(model=mock_model)
    str_repr = str(evaluator)
    
    assert "Evaluator" in str_repr
    assert str(mock_model) in str_repr 
