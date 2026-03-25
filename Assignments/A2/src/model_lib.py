import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

class FinancialModelBuilder:
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels

    def build_lora_model(self, r: int = 8, lora_alpha: int = 16, dropout: float = 0.1):
        """
        机理：在预训练 BERT 模型中注入 LoRA 层
        """
        # 1. 加载基础序列分类模型
        # 注意：此时模型的分类头是随机初始化的
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )

        # 2. 定义 LoRA 配置
        # target_modules: BERT 结构中通常针对 query 和 value 进行优化效果最好
        # modules_to_save: 必须包含 classifier，确保随机初始化的分类头参与训练
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=r, 
            lora_alpha=lora_alpha, 
            target_modules=["query", "value"],
            lora_dropout=dropout,
            modules_to_save=["classifier"] 
        )

        # 3. 转化为 PEFT 模型
        lora_model = get_peft_model(model, peft_config)
        
        # 打印可训练参数占比（这是 A2 实验报告的核心数据点）
        print("--- Trainable Parameters Status ---")
        lora_model.print_trainable_parameters()
        
        return lora_model

# 测试逻辑
if __name__ == "__main__":
    builder = FinancialModelBuilder()
    model = builder.build_lora_model()