#!/bin/bash
#SBATCH --job-name=mu
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mu_%A_%a.out
#SBATCH --array=0-4

source scripts/sweep_common.sh

MUS=(1 2 5 10 20)
MU=${MUS[$SLURM_ARRAY_TASK_ID]}

run_attack Qwen-VL mu_$MU
