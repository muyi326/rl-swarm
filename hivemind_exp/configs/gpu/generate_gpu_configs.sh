#!/bin/bash

ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

# TODO: Fix this to use the Gensyn uploaded prefix!!
python3 $ROOT_DIR/../generate_configs.py \
    --yaml_prefix "grpo-qwen-2.5" \
    --yaml_suffix "deepseek-r1" \
    --yaml_header "$ROOT_DIR/grpo-header.yaml" \
    --yaml_big_header "$ROOT_DIR/grpo-big-header.yaml" \
    --yaml_output_dir "$ROOT_DIR" \
    --model_prefix "Gensyn/Qwen2.5" \
    --param_counts 0.5 1.5 7 \
    --param_counts_4bit 32 72
