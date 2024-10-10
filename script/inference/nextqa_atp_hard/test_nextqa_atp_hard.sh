export PYTHONPATH="./:$PYTHONPATH"

model=$1
gpu=$2
output="test_output/nextqa_atp_hard/${model}/"
mkdir -p $output

# usage: 
# bash script/inference/nextqa_atp_hard/test_nextqa_atp_hard.sh model_name 0 &

CUDA_VISIBLE_DEVICES=$gpu python stllm/test/nextqa_atp_hard/nextqa_atp_hard_infer.py \
    --cfg-path config/$model.yaml \
    --ckpt-path output/${model}/pytorch_model.bin \
    --anno-path /path/to/Data-TG-Vid/Testing/NextQA-ATP-Hard \
    --output_dir $output \
    --output_name ${model} \
    --num-frames 0 \
    --ask_simple \
    2>&1 | tee $output/log.txt


