#!/bin/bash
#SBATCH --job-name=pooling_method
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pooling_%A_%a.out
#SBATCH --array=0-2

source scripts/sweep_common.sh

POOLS=(last_token mean image_only)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}

run_attack InternVL pooling_method_$POOL
