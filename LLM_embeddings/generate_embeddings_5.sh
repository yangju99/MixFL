#!/bin/bash

LLMS=(
    "codebert_base"
    "codegen_350m"
    "codet5_base"
    "graphcodebert_base"
    "incoder_1b"
    "unixcoder_base"
)

# 공통 설정
DATA_DIR="./defects4j_1.4.0"
# DATA_DIR="./defects4j_2.0.0"

SAVE_ROOT="./chunks_defects4j_1.4.0"
# SAVE_ROOT="./chunks_defects4j_2.0.0"

seed=5
for llm in "${LLMS[@]}"; do
    echo "=== Running generate_embeddings.py for $llm with seed $seed ==="
    python generate_embeddings.py \
        --llm "$llm" \
        --data_dir "$DATA_DIR" \
        --save_root "$SAVE_ROOT" \
        --seed "$seed"
done