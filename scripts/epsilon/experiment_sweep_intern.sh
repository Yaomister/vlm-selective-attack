#!/bin/bash
#SBATCH --job-name=epsilon
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/epsilon_%A_%a.out
#SBATCH --array=0-6

mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

EPSILONS=(0.01 0.025 0.05 0.1 0.25 0.5 1)
EPSILON=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

torchrun --standalone --nproc_per_node=4 experiments/experiment_v4.py \
  --model_name InternVL \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/InternVL/epsilon_$EPSILON \
  --steps 200 \
  --epsilon $EPSILON \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last -1 \
  --pooling_method last_token