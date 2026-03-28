import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

class FinancialModelBuilder:
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels

    def build_lora_model(self, r: int = 8, lora_alpha: int = 16, dropout: float = 0.1):
        """
        机理：LoRA 模式 - 在 Transformer 内部注入低秩矩阵 (BA) 并训练分类头
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=r, 
            lora_alpha=lora_alpha, 
            target_modules=["query", "value"],
            lora_dropout=dropout,
            modules_to_save=["classifier"] 
        )

        lora_model = get_peft_model(model, peft_config)
        
        print("--- [MODE: LoRA] Trainable Parameters Status ---")
        lora_model.print_trainable_parameters()
        
        return lora_model

    def build_mlp_only_model(self):
        """
        机理：Linear Probing 模式 - 彻底冻结 BERT Backbone，仅开放顶层分类头 (MLP)
        """
        # 1. 加载原始序列分类模型
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )

        # 2. 遍历所有参数，手动执行冻结逻辑
        for name, param in model.named_parameters():
            # 只有名字里包含 'classifier' 的参数（即最顶层的 MLP）才允许计算梯度
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 3. 打印参数状态用于 A2 报告对比
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print("--- [MODE: MLP Only] Trainable Parameters Status ---")
        print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")
        
        return model

# --- 测试逻辑 ---
if __name__ == "__main__":
    builder = FinancialModelBuilder()
    
    print("\n[Testing LoRA Construction...]")
    m_lora = builder.build_lora_model(r=8)
    
    print("\n[Testing MLP-Only Construction...]")
    m_mlp = builder.build_mlp_only_model()