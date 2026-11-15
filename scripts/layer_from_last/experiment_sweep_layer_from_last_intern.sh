#!/bin/bash
#SBATCH --job-name=layer_from_last
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/layer_%A_%a.out
#SBATCH --array=0-4

source scripts/sweep_common.sh

LAYERS=(-1 -2 -4 -8 -16)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

run_attack InternVL layer_from_last_$LAYER
