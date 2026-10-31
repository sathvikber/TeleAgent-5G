#!/bin/bash

sudo docker run --rm -it \
    --gpus all \
    -v ~/teleagent:/workspace \
    -v ~/slm-lora:/workspace/slm-lora \
    -v ~/slm-lora/hf_cache:/home/sapna/.cache/huggingface \
    -w /workspace \
    teleagent:0.2
