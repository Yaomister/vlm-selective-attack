#!/bin/bash
#SBATCH --job-name=extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/mu_0_%A_%a.out
#SBATCH --array=0

source scripts/sweep_common.sh

run_attack LLaVA-1.5-7b mu_0
