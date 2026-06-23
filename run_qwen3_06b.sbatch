#!/bin/bash
#SBATCH --job-name=qwen3-0.6b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
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

sh train.sh qwen3-0.6b-h200-run-2/ --model=Qwen3-0.6B

echo "End time: $(date)"