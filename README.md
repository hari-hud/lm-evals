# LM-Evals: Unified Language Model Evaluation Framework

A comprehensive framework for evaluating language models, retrieval-augmented generation (RAG) systems, and AI agents.

## Features

- **Traditional LLM Benchmarks**
  - Support for common benchmarks (MMLU, ARC, HellaSwag, TruthfulQA, etc.)
  - Automated evaluation pipelines
  - Customizable scoring metrics

- **RAG System Evaluation**
  - Context relevance assessment
  - Answer accuracy and faithfulness
  - Retrieval quality metrics
  - End-to-end performance evaluation

- **Agent Evaluation**
  - Task completion assessment
  - Decision-making evaluation
  - Tool usage efficiency
  - Multi-step reasoning capabilities

## Supported Models

- **OpenAI Models**
  - GPT-4, GPT-3.5-turbo, etc.
  - Full API feature support
  - Streaming and non-streaming modes

- **NVIDIA NIM Models**
  - Support for any model deployed via NVIDIA Inference Microservices
  - Token logprobs and tokenization support
  - Configurable generation parameters
  - Easy integration with NIM endpoints

## Installation

```bash
# Using poetry (recommended)
poetry install

# Using pip
pip install -e .
```

## Quick Start

### Using OpenAI Models

```python
from lm_evals import Evaluator
from lm_evals.models import OpenAIModel
from lm_evals.benchmarks import MMLU

# Initialize model and evaluator
model = OpenAIModel(model="gpt-4")
evaluator = Evaluator(model=model)

# Run MMLU benchmark
results = evaluator.evaluate(MMLU())
print(results.summary())
```

### Using NVIDIA NIM Models

```python
from lm_evals import Evaluator
from lm_evals.models import NIMModel
from lm_evals.benchmarks import MMLU

# Initialize NVIDIA NIM model
model = NIMModel(
    model_name="llama2-70b",  # Name of your deployed model
    endpoint_url="https://your-nim-endpoint.com",  # Your NIM endpoint URL
    api_key="your-api-key"  # Optional API key
)
evaluator = Evaluator(model=model)

# Run MMLU benchmark
results = evaluator.evaluate(MMLU())
print(results.summary())
```

## Available Benchmarks

### Traditional Benchmarks
- MMLU (Massive Multitask Language Understanding)
- ARC (AI2 Reasoning Challenge)
- HellaSwag
- TruthfulQA
- GSM8K
- BBH (Big Bench Hard)

### RAG Benchmarks
- RAGAS Metrics Suite
- Context Relevance Score
- Answer Faithfulness
- Retrieval Precision/Recall

### Agent Benchmarks
- Task Completion Rate
- Decision Quality
- Tool Usage Efficiency
- Multi-step Reasoning

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Kubernetes Cluster

The Kubernetes configurations are split into separate files for better management:
- `infra/k8s/job.yaml`: Job configuration
- `infra/k8s/pvc.yaml`: PersistentVolumeClaim for storing results
- `infra/k8s/secret.yaml`: Secret for API keys

```bash
# Create secret with your API key (replace with actual key)
kubectl create secret generic api-keys \
    --from-literal=openai-api-key='your-key-here'

# Apply the configurations
kubectl apply -f infra/k8s/pvc.yaml
kubectl apply -f infra/k8s/job.yaml

# Monitor the job
kubectl get jobs
kubectl logs job/llm-evaluation
``` 
