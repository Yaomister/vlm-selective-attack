#!/bin/bash
#SBATCH --job-name=epsilon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mu_%A_%a.out

mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate


python experiments/experiment_v3.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/LLaVA-1.5-7b/epsilon_1 \
  --steps 200 \
  --epsilon 1 \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last -1 \
  --pooling_method last_token