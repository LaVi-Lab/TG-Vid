export PYTHONPATH="./:$PYTHONPATH"

# usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/train/train.sh model_name
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash script/train/train.sh TG-Vid-197K

config=$1
mkdir -p output/$config

export WANDB_MODE="offline"
export WANDB_ENTITY="WANDB_ENTITY"
export WANDB_PROJECT="WANDB_PROJECT"
export WANDB_NAME=$config

deepspeed --master_port=20000 --include=localhost:0,1,2,3,4,5,6,7 \
    stllm/train/train_hf.py \
    --cfg-path config/$config.yaml \
    2>&1 | tee output/$config/log.txt