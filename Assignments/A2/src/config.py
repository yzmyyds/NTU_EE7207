import torch
import os


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # 路径配置
    DATA_PATH = "Assignments/A2/data/Sentences_50Agree.txt"
    SAVE_DIR = "Assignments/A2/saved_models"
    
    # 模型配置
    BASE_MODEL_NAME = "bert-base-uncased"
    
    # LoRA 参数 (方案 C 核心)
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-4
    MAX_LENGTH = 128
    SEED = 42

    # 自动创建保存目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)