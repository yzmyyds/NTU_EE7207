import torch
import os
from transformers import AutoTokenizer
from datasets import Dataset

class LocalFinancialDataLoader:
    def __init__(self, model_name: str, max_length: int = 128):
        """
        :param model_name
        :param max_length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        self.label_map = {"neutral": 0, "positive": 1, "negative": 2}

    def _load_and_parse(self, file_path: str):
        data = {"sentence": [], "labels": []}
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到文件: {file_path}")

        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if "@" in line:
                    parts = line.split("@")
                    sentence = "@".join(parts[:-1]) 
                    sentiment = parts[-1].lower()
                    
                    if sentiment in self.label_map:
                        data["sentence"].append(sentence)
                        data["labels"].append(self.label_map[sentiment])
        
        return Dataset.from_dict(data)

    def _tokenize_fn(self, examples):
        return self.tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

    def prepare_datasets(self, file_name: str, data_dir: str = "Assignments/A2/data", test_size=0.2, seed=42):
        """
        核心流程：读取 -> 转换 -> Tokenize -> 切分
        """
        file_path = os.path.join(data_dir, file_name)
        
        # 1. 加载并解析本地文本
        raw_dataset = self._load_and_parse(file_path)

        # 2. 特征编码 (Feature Encoding)
        tokenized_ds = raw_dataset.map(
            self._tokenize_fn,
            batched=True,
            remove_columns=["sentence"]
        )

        tokenized_ds.set_format("torch")
        split_ds = tokenized_ds.train_test_split(test_size=test_size, seed=seed)
        
        return split_ds["train"], split_ds["test"]

