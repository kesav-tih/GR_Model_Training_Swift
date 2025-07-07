#!/bin/bash

# Default USERNAME if not set
USERNAME=${USERNAME:-venkat_kesav}

# Activate the virtual environment
source /home/${USERNAME}/.venv/bin/activate

# Environmental Variables
export CUDA_VISIBLE_DEVICES=0

swift infer \
    --adapters /workspace/output/v29-20250421-153328/checkpoint-61 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048