#!/bin/bash
#SBATCH --job-name=trace-collect
#SBATCH --output=logs/trace-collect-%j.log
#SBATCH --error=logs/trace-collect-%j.err
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=sergio.talavera@sjsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------
module load anaconda/3.9
source activate thesis
export PATH="/home/017557527/.conda/envs/thesis/bin:$PATH"

# MiniWoB task HTML files (served locally via file://)
export MINIWOB_URL="file:///home/017557527/cmpe299b/miniwob-plusplus/miniwob/html/miniwob/"

# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------
echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'"
echo "Python: $(which python)"
echo ""

# ------------------------------------------------------------------
# Ensure log directory exists
# ------------------------------------------------------------------
mkdir -p /home/017557527/agent_xes/logs

# ------------------------------------------------------------------
# Run trace collection
# ------------------------------------------------------------------
cd /home/017557527/agent_xes

python scripts/collect_traces.py \
    --model-name "${MODEL_NAME:-llama-3.2-3b}" \
    --model-path "${MODEL_PATH:-/home/017557527/cmpe299b/models/llama-3.2-3b}" \
    --output-dir data/raw_traces \
    --benchmark miniwob \
    --max-steps 30 \
    "$@"

EXIT_CODE=$?

echo ""
echo "=== Job finished at $(date) with exit code $EXIT_CODE ==="
exit $EXIT_CODE
