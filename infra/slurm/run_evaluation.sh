#!/bin/bash
#SBATCH --job-name=llm-eval
#SBATCH --output=logs/llm-eval_%j.out
#SBATCH --error=logs/llm-eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --nodes=1

# Load any required modules
module load python/3.10

# Activate virtual environment if using one
# source /path/to/venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"
export OPENAI_API_KEY="your-api-key-here"

# Create results directory
mkdir -p results

# Run the evaluation from project root
cd "${SLURM_SUBMIT_DIR}"
srun python examples/run_judge_eval.py \
    --output_dir results \
    --job_id "${SLURM_JOB_ID}" 
