export PYTHONPATH="./:$PYTHONPATH"

model=$1
gpu=$2
output="test_output/mvbench/${model}/"
mkdir -p $output

# usage: 
# bash script/inference/mvbench/test_mvbench.sh model_name 0 &

CUDA_VISIBLE_DEVICES=$gpu python stllm/test/mvbench/mv_bench_infer.py \
    --cfg-path config/$model.yaml \
    --ckpt-path output/${model}/pytorch_model.bin \
    --anno-path /path/to/Data-TG-Vid/Testing/MVBench/json \
    --output_dir $output \
    --output_name ${model} \
    --num-frames 0 \
    --ask_simple \
    2>&1 | tee $output/log.txt


