#!/bin/bash
#SBATCH --job-name=epsilon
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/epsilon_%A_%a.out
#SBATCH --array=0-6

source scripts/sweep_common.sh

EPSILONS=(0.01 0.025 0.05 0.1 0.25 0.5 1)
EPSILON=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

run_attack LLaVA-1.5-7b epsilon_$EPSILON
