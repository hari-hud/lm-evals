"""Unit tests for similarity metrics benchmark."""
import pytest
from unittest.mock import Mock, patch
import json
import tempfile
from pathlib import Path

from lm_evals.benchmarks.similarity import (
    SimilarityMetricsConfig,
    SimilarityMetricsBenchmark
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "reference": "Paris is the capital of France."
        },
        {
            "input": "What is 2+2?",
            "output": "2+2 equals 4",
            "reference": "The answer is 4"
        }
    ]


@pytest.fixture
def temp_json_files(sample_data):
    """Create temporary JSON files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_path = Path(tmpdir) / "predictions.json"
        ref_path = Path(tmpdir) / "references.json"
        
        # Create predictions file
        with open(pred_path, 'w') as f:
            json.dump(sample_data, f)
        
        # Create references file (same structure but with 'reference' instead of 'output')
        references = [
            {
                "input": item["input"],
                "reference": item["reference"]
            }
            for item in sample_data
        ]
        with open(ref_path, 'w') as f:
            json.dump(references, f)
        
        yield str(pred_path), str(ref_path)


def test_benchmark_initialization():
    """Test benchmark initialization with default config."""
    benchmark = SimilarityMetricsBenchmark()
    assert isinstance(benchmark.config, SimilarityMetricsConfig)
    assert "rouge" in benchmark.config.metrics
    assert "bleu" in benchmark.config.metrics
    assert "bert_score" in benchmark.config.metrics
    assert "exact_match" in benchmark.config.metrics


def test_benchmark_with_custom_config():
    """Test benchmark initialization with custom config."""
    config = SimilarityMetricsConfig(
        metrics=["rouge", "bleu"],
        batch_size=16
    )
    benchmark = SimilarityMetricsBenchmark(config=config)
    assert benchmark.config.metrics == ["rouge", "bleu"]
    assert benchmark.config.batch_size == 16


def test_compute_metrics():
    """Test computation of similarity metrics."""
    benchmark = SimilarityMetricsBenchmark()
    predictions = ["The capital of France is Paris."]
    references = ["Paris is the capital of France."]
    
    metrics = benchmark.compute_metrics(predictions, references)
    
    assert "rouge1_f" in metrics
    assert "rouge2_f" in metrics
    assert "rougeL_f" in metrics
    assert "bleu" in metrics
    assert "bert_score_f1" in metrics
    assert "exact_match" in metrics
    
    # Check metric ranges
    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, f"Metric {metric_name} out of range [0,1]"


def test_benchmark_with_pregenerated_data(temp_json_files):
    """Test benchmark with pre-generated predictions."""
    pred_path, ref_path = temp_json_files
    
    config = SimilarityMetricsConfig(
        predictions_path=pred_path,
        reference_path=ref_path
    )
    benchmark = SimilarityMetricsBenchmark(config=config)
    
    # No model needed when using pre-generated predictions
    results = benchmark.run()
    
    assert results.name == "similarity_metrics"
    assert isinstance(results.metrics, dict)
    assert isinstance(results.details, dict)
    assert len(results.details) == 2  # Two test cases


def test_benchmark_with_model():
    """Test benchmark with model inference."""
    mock_model = Mock()
    mock_model.generate.return_value = "Test response"
    
    benchmark = SimilarityMetricsBenchmark()
    results = benchmark.run(model=mock_model)
    
    assert results.name == "similarity_metrics"
    assert isinstance(results.metrics, dict)
    assert isinstance(results.details, dict)
    assert mock_model.generate.called


def test_error_handling():
    """Test error handling in benchmark."""
    # Test missing model
    benchmark = SimilarityMetricsBenchmark()
    with pytest.raises(ValueError, match="Model must be provided"):
        benchmark.run()
    
    # Test invalid file format
    config = SimilarityMetricsConfig(dataset_path="invalid.txt")
    benchmark = SimilarityMetricsBenchmark(config=config)
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        benchmark.run(Mock())


@pytest.mark.parametrize("metric", ["rouge", "bleu", "bert_score", "exact_match"])
def test_individual_metrics(metric):
    """Test each metric individually."""
    config = SimilarityMetricsConfig(metrics=[metric])
    benchmark = SimilarityMetricsBenchmark(config=config)
    
    predictions = ["This is a test."]
    references = ["This is a test."]
    
    metrics = benchmark.compute_metrics(predictions, references)
    
    if metric == "rouge":
        assert "rouge1_f" in metrics
        assert "rouge2_f" in metrics
        assert "rougeL_f" in metrics
    elif metric == "bert_score":
        assert "bert_score_f1" in metrics
    else:
        assert metric in metrics 
