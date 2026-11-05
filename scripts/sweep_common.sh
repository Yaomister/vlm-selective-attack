mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

STEP=200
EPSILON=0.025
ALPHA=0.001
MU=10
LAYER=-1
POOL=last_token

run_attack() {
  python experiments/experiment_v4.py \
    --model_name $1 \
    --dataset_dir ./sorted \
    --output_dir ./attack_results/$1/$2 \
    --steps $STEP \
    --epsilon $EPSILON \
    --alpha $ALPHA \
    --mu $MU \
    --layer_from_last $LAYER \
    --pooling_method $POOL
}
