#!/bin/bash
#SBATCH --job-name=epsilon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/epsilon_%A_%a.out
#SBATCH --array=0-7          

mkdir -p logs
source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

CHUNK_SIZE=100
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))

python experiments/experiment_v3.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/LLaVA-1.5-7b/epsilon_0.025 \
  --steps 200 \
  --epsilon 0.025 \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last -1 \
  --pooling_method last_token \
  --subset ${START}-${END}