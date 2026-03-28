#!/bin/bash

# 1. 环境配置
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src

# 2. 定义超参数 (建议 r 取 8 或 16)
RANK=8
BATCH=16
EPOCHS=3

echo "-----------------------------------------------"
echo "🚀 Starting [LoRA] Training Mode"
echo "📊 Config: Rank=$RANK, Batch=$BATCH, Epochs=$EPOCHS"
echo "-----------------------------------------------"

# 3. 执行训练
python Assignments/A2/src/train.py \
    --mode lora \
    --r $RANK \
    --batch $BATCH \
    --epochs $EPOCHS \
    2>&1 | tee train_lora.log

echo "✅ LoRA Training Completed!"