import torch
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class FinancialInferenceEngine:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # 1. 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        
        # 2. 核心机理：自动识别加载模式
        adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
        
        if os.path.exists(adapter_cfg_path):
            print(f"🔎 检测到 LoRA 适配器，正在执行 [PEFT] 挂载...")
            # 这种方式最稳妥：先读配置，再加载 base，最后挂载
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                config.BASE_MODEL_NAME, 
                num_labels=3
            )
            # 加载并自动处理 modules_to_save (即 classifier)
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            print(f"🔎 检测到全量/MLP 权重，正在执行 [Direct] 加载...")
            # 对于 MLP-Only 模式，权重直接就在主模型里，不需要 PeftModel
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path, 
                num_labels=3
            )
        
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 推理引擎已就绪：{checkpoint_path}")

    def predict_batch(self, test_dataset):
        all_preds = []
        print(f"🚀 正在执行 {len(test_dataset)} 条数据的原生推断...")
        
        with torch.no_grad():
            for batch in test_dataset:
                # 保持你原版成功的推理逻辑
                inputs = {
                    'input_ids': batch['input_ids'].unsqueeze(0).to(self.device),
                    'attention_mask': batch['attention_mask'].unsqueeze(0).to(self.device),
                    'token_type_ids': batch['token_type_ids'].unsqueeze(0).to(self.device)
                }
                outputs = self.model(**inputs)
                pred_idx = torch.argmax(outputs.logits, dim=-1).item()
                all_preds.append(pred_idx)
                
        return all_preds