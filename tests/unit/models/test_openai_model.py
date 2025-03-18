"""Unit tests for OpenAI model implementation."""
import pytest
from unittest.mock import Mock, patch

from lm_evals.models.openai import OpenAIModel


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("openai.OpenAI") as mock_client:
        yield mock_client


@pytest.fixture
def model():
    """Create an OpenAI model instance."""
    return OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="test-key",
        temperature=0.7
    )


def test_model_initialization(model):
    """Test model initialization with correct parameters."""
    assert model.model == "gpt-3.5-turbo"
    assert model.temperature == 0.7
    assert model.api_key == "test-key"


def test_model_initialization_defaults():
    """Test model initialization with default parameters."""
    model = OpenAIModel(model="gpt-3.5-turbo")
    assert model.temperature == 0.0
    assert model.max_tokens is None


def test_generate(model, mock_openai_client):
    """Test text generation."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    response = model.generate("Test prompt")
    assert response == "Test response"
    
    mock_openai_client.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.7,
        max_tokens=None
    )


def test_generate_with_system_prompt(model, mock_openai_client):
    """Test text generation with system prompt."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    response = model.generate(
        "Test prompt",
        system_prompt="You are a helpful assistant."
    )
    assert response == "Test response"
    
    mock_openai_client.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test prompt"}
        ],
        temperature=0.7,
        max_tokens=None
    )


def test_generate_with_max_tokens(model, mock_openai_client):
    """Test text generation with max tokens limit."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    response = model.generate("Test prompt", max_tokens=100)
    assert response == "Test response"
    
    mock_openai_client.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.7,
        max_tokens=100
    )


@pytest.mark.asyncio
async def test_agenerate(model, mock_openai_client):
    """Test asynchronous text generation."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_openai_client.return_value.chat.completions.create = Mock(
        return_value=mock_response
    )

    response = await model.agenerate("Test prompt")
    assert response == "Test response"


def test_get_token_logprobs(model, mock_openai_client):
    """Test getting token logprobs."""
    mock_response = Mock()
    mock_response.choices = [
        Mock(
            logprobs=Mock(
                content=[
                    {"text": "Hello", "logprob": -0.1},
                    {"text": "world", "logprob": -0.2}
                ]
            )
        )
    ]
    mock_openai_client.return_value.chat.completions.create.return_value = mock_response

    logprobs = model.get_token_logprobs("Hello world")
    assert len(logprobs) == 2
    assert logprobs[0] == -0.1
    assert logprobs[1] == -0.2


def test_error_handling(model, mock_openai_client):
    """Test error handling during generation."""
    mock_openai_client.return_value.chat.completions.create.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        model.generate("Test prompt")
    assert str(exc_info.value) == "API Error" 
