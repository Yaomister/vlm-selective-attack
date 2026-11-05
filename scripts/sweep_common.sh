# Shared setup for the experiment sweep jobs. Source this from the repo root,
# override whichever parameter the sweep varies, then call run_attack.

mkdir -p logs

source /home/yao.eric/vlm-selective-attack/.venv/bin/activate

# baseline attack parameters
STEP=200
EPSILON=0.025
ALPHA=0.001
MU=10
LAYER=-1
POOL=last_token

# usage: run_attack <model_name> <output_subdir>
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
