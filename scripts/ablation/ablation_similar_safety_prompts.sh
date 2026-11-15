#!/bin/bash
#SBATCH --job-name=ablation_similar_safety_prompts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_similar_safety_prompts_%A_%a.out
#SBATCH --array=0-7

mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

python -m experiments.ablation \
  --experiment similar_safety_prompts \
  --dataset-dir ./attack_results \
  --output-dir ./ablation/shard_$SLURM_ARRAY_TASK_ID \
  --num-shards 8 \
  --shard-id $SLURM_ARRAY_TASK_ID
