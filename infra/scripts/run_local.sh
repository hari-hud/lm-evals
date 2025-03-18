#!/bin/bash

# Check if OPENAI_API_KEY is set
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Create output directory
mkdir -p "${PROJECT_ROOT}/results"

# Run evaluation from project root
cd "${PROJECT_ROOT}"
python examples/run_judge_eval.py \
    --output_dir results \
    --judge_model "gpt-4" \
    --eval_model "gpt-3.5-turbo" \
    --num_judges 3 
