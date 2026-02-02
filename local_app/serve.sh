#!/bin/bash

# Ensure we are using the venv
# source venv/bin/activate 

echo "Starting vLLM Multi-LoRA Server..."

vllm serve unsloth/llama-3-8b-bnb-4bit \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype half \
    --enable-lora \
    --lora-modules sql-model=./sql_adapter shakespeare-model=./creative_adapter \
    --max-loras 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --port 8000