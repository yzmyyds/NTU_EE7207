import os
import argparse
import numpy as np
import torch
import evaluate
from transformers import TrainingArguments, Trainer
from config import Config
from data_loader import LocalFinancialDataLoader
from model_lib import FinancialModelBuilder

def parse_args():
    parser = argparse.ArgumentParser(description="EE7207 A2 Training Script")
    # 核心：增加模式切换参数
    parser.add_argument('--mode', type=str, default='lora', choices=['lora', 'mlp'],
                        help="训练模式: 'lora' (PEFT 适配) 或 'mlp' (仅训练输出层)")
    parser.add_argument('--r', type=int, default=Config.LORA_R, help="LoRA 的秩 (Rank)")
    parser.add_argument('--batch', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    return parser.parse_args()

def run_experiment():
    args = parse_args()

    # 1. 环境与数据准备
    loader = LocalFinancialDataLoader(Config.BASE_MODEL_NAME, Config.MAX_LENGTH)
    train_ds, test_ds = loader.prepare_datasets(
        file_name="Sentences_50Agree.txt",
        seed=Config.SEED
    )

    # 2. 动态构建模型
    builder = FinancialModelBuilder(Config.BASE_MODEL_NAME)
    
    if args.mode == 'lora':
        model = builder.build_lora_model(r=args.r, lora_alpha=Config.LORA_ALPHA)
        experiment_id = f"lora_r{args.r}_b{args.batch}_e{args.epochs}"
    else:
        # 消融实验：只训练 MLP (Linear Probing)
        model = builder.build_mlp_only_model()
        experiment_id = f"mlp_only_b{args.batch}_e{args.epochs}"

    # 3. 设置评估指标
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 4. 配置训练参数
    # 自动生成不同的输出路径，方便 ipynb 对比
    output_path = os.path.join(Config.SAVE_DIR, experiment_id)
    
    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        # 针对 MacBook MPS 或普通显存优化的设置
        remove_unused_columns=False 
    )

    # 5. 启动 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\n🚀 [START] Mode: {args.mode.upper()} | Output: {output_path}")
    trainer.train()
    
    # 训练结束后的清理或保存操作
    print(f"✅ [FINISHED] Best model saved to {output_path}")

if __name__ == "__main__":
    run_experiment()