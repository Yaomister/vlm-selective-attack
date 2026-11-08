#!/bin/bash
#SBATCH --job-name=ablation_safety_prompt_with_description
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_safety_prompt_with_description_%A_%a.out
#SBATCH --array=0-7

mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

python -m experiments.ablation \
  --experiment safety_prompt_with_description \
  --dataset-dir ./attack_results \
  --output-dir ./ablation/shard_$SLURM_ARRAY_TASK_ID \
  --num-shards 8 \
  --shard-id $SLURM_ARRAY_TASK_ID
