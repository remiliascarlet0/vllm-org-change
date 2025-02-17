cd ../
bash compile.sh
cd examples

pip uninstall -y vllm-flash-attn

pkill -f spawn

# sleep 3000

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# pgrep -f 'api_server' | xargs kill -9

# preemption_mode=swap # 1: swap 2: recomputation
gpu_id=3
tensor_parallel_size=1
# gpu_memory_utilizations=(0.3)
# gpu_memory_utilizations=(0.2)
gpu_memory_utilizations=(0.4)
# gpu_memory_utilizations=(0.5)
preemption_mode=swap

# models=(facebook/opt-30b meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf)
# models=(facebook/opt-2.7b)
models=(meta-llama/Llama-2-7b-hf)
# models=(meta-llama/Llama-2-13b-hf)
# request_rates=(50 100 150 200 250 300)
request_rates=(300) 
num_prompts=(300)
max_num_seqs=512
dataset_path=/nfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json

wait_for_server() {
    local port=$1
    while true; do
        if netstat -tulnp | grep -q "${port}"; then
            echo "server is running on port ${port}"
            break
        else
            echo "server is not running on port ${port}"
            sleep 5
        fi
    done
}

for i in {1..1}; do
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        gpu_memory_utilization="${gpu_memory_utilizations[$i]}"
        model_name=$(echo "$model" | tr '/' '_')
        for request_rate in ${request_rates[@]}; do
            for num_prompt in ${num_prompts[@]}; do
                CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                    --model ${model} \
                    --port 8080 \
                    --tensor-parallel-size ${tensor_parallel_size} \
                    --swap-space 4 \
                    --gpu-memory-utilization ${gpu_memory_utilization} \
                    --max-num-seqs ${max_num_seqs} \
                    --enable-chunked-prefill \
                    --disable-log-requests > logs/${model_name}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_1.0_${tensor_parallel_size}gpu.log & 
                pid=$!    
                wait_for_server 8080
                sleep 1

                python3 ../benchmarks/benchmark_serving.py \
                    --model ${model} \
                    --port 8080 \
                    --dataset ${dataset_path} \
                    --request-rate ${request_rate} \
                    --num-prompts ${num_prompt} \
                    --save-result \
                    --result-dir logs \
                    --endpoint /v1/completions >> logs/${model_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_1.0_${tensor_parallel_size}gpu.log 
                kill -9 $pid 
                sleep 1
            done
        done
    done
done    
