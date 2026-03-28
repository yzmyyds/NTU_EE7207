#!/bin/bash

# 1. 环境配置
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src

# 2. 定义超参数 (Batch 和 Epochs 必须与 LoRA 保持一致，才能公平对比)
BATCH=16
EPOCHS=3

echo "-----------------------------------------------"
echo "🚀 Starting [MLP-Only] Training Mode (Ablation)"
echo "📊 Config: Batch=$BATCH, Epochs=$EPOCHS"
echo "-----------------------------------------------"

# 3. 执行训练 (不需要传 --r 参数)
python Assignments/A2/src/train.py \
    --mode mlp \
    --batch $BATCH \
    --epochs $EPOCHS \
    2>&1 | tee train_mlp.log

echo "✅ MLP-Only Training Completed!"