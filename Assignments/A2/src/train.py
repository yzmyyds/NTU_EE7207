import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
from config import Config
from data_loader import LocalFinancialDataLoader
from model_lib import FinancialModelBuilder

def run_experiment():
    # 1. 加载数据
    loader = LocalFinancialDataLoader(Config.BASE_MODEL_NAME, Config.MAX_LENGTH)
    train_ds, test_ds = loader.prepare_datasets(
        file_name="Sentences_50Agree.txt",
        seed=Config.SEED
    )

    # 2. 构建 LoRA 模型
    builder = FinancialModelBuilder(Config.BASE_MODEL_NAME)
    model = builder.build_lora_model(r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA)

    # 3. 定义评估指标 (机理：计算预测值与真实标签的 Accuracy)
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 4. 设置训练指令
    training_args = TrainingArguments(
        output_dir=Config.SAVE_DIR,
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none" # 防止弹出额外的登录提示
    )

    # 5. 实例化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    # 6. 执行！
    print(f"🚀 启动训练...")
    trainer.train()

if __name__ == "__main__":
    run_experiment()