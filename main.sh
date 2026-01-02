#!/bin/bash
#SBATCH --job-name=shinka-bias
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --output=/scratch/jacopo04/ShinkaEvolve/logs/main-%j.out
#SBATCH --error=/scratch/jacopo04/ShinkaEvolve/logs/main-%j.err
#SBATCH -D /scratch/jacopo04/ShinkaEvolve

module --force purge
module load StdEnv/2023 python/3.11.5
module load gcc opencv arrow

source .venv/bin/activate

export HF_HOME=".cache"
export CUDA_VISIBLE_DEVICES="0"
 
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 80000 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --port 8001 &

# Store the PID of the last background command
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Active Wait for Server
echo "Waiting for server to become available at http://localhost:8001 ..."
while ! curl -s http://localhost:8001/health > /dev/null; do
    echo "   ...server not ready yet. Retrying in 5 seconds."
    sleep 10
done
echo "Server is ready!"

python examples/maze_model_search/run_evo.py
