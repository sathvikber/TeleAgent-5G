#!/bin/bash

sudo docker run --rm -it \
    -p 8000:8000 \
    --env-file ~/teleagent/.env.niral \
    -v ~/teleagent:/workspace \
    -v ~/teleagent/slm-lora:/workspace/slm-lora \
    -v ~/teleagent/slm-lora/hf_cache:/root/.cache/huggingface \
    -w /workspace \
    teleagent:cpu
