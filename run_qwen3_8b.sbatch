#!/bin/bash
#SBATCH --job-name=qwen3-8b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

cd /hpc/home/zsheng1/checkpoint

mkdir -p logs

nvidia-smi
free -h
nproc

sh train.sh qwen3-8b-h200-run-3/ --model=Qwen3-8B

echo "End time: $(date)"