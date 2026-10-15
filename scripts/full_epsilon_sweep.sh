#!/bin/bash
mkdir -p logs
MODELS=(Qwen-VL InternVL LLaVA-NeXT LLaVA-1.5-7b)
EPSILONS=(0.01 0.025 0.05 0.1 0.25 0.5 1)

JOBS=()

for m in "${MODELS[@]}"; do
    for e in "${EPSILONS[@]}"; do JOBS+=("$m $e"); done
done

NUM_GPUS=8

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    (
        for ((i=gpu; i<${#JOBS[@]}; i+= NUM_GPUS)); do
            read MODEL EPSILON <<< "${JOBS[$i]}"
            CUDA_VISIBLE_DEVICES=$gpu uv run python experiments/experiment_v3.py \
                --model_name $MODEL --dataset_dir ./sorted \
                --output_dir ./attack_results/$MODEL/epsilon_$EPSILON \
                --steps 200 --epsilon $EPSILON --alpha 0.001 --mu 10 \
                --layer_from_last -1 --pooling_method last_token \
                > logs/${MODEL}_eps${EPSILON}.out 2>&1
            done
    ) & 
    done
    wait

