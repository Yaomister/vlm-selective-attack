#!/bin/bash
#SBATCH --job-name=steps
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/steps_%A_%a.out
#SBATCH --array=0-4

source scripts/sweep_common.sh

STEPS=(50 100 200 400 800)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}

run_attack LLaVA-NeXT STEP_$STEP
