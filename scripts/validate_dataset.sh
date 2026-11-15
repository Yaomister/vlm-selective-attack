#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/validate_dataset_%A_%a.out

mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

python dataset/validate_dataset.py --dataset-dir ./sorted