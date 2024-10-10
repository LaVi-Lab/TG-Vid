export PYTHONPATH="./:$PYTHONPATH"

model=$1
gpu=$2
output="test_output/tempcompass/${model}/"
mkdir -p $output

# usage: 
# bash script/inference/tempcompass/test_tempcompass.sh model_name 0 &

CUDA_VISIBLE_DEVICES=$gpu python stllm/test/tempcompass/tempcompass_infer.py \
    --cfg-path config/$model.yaml \
    --ckpt-path output/${model}/pytorch_model.bin \
    --anno-path /path/to/Data-TG-Vid/Testing/TempCompass \
    --output_dir $output \
    --output_name ${model} \
    --num-frames 0 \
    --ask_simple \
    2>&1 | tee $output/log.txt


