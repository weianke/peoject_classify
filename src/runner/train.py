import torch
from torch.optim.adam import Adam
# 导入预训练分词器（后续文本分词使用）
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from configuration.config import *
from process.dataset import get_dataloader
from dataclasses import dataclass

# 训练配置类
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-5
    output_dir: str = MODEL_DIR

# 训练器类
class Trainer:
    # 初始化（修正拼写 __init__）
    def __init__(self, model, train_loader, optinizer, device, train_config=None):
        pass
    # 核心训练方法
    def train(self):
        pass

if __name__ == '__main__':
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    # 3. 读取分类标签文件
    with open(MODEL_DIR / LABELS_FILE, encoding='utf-8') as f:
        all_labels = f.read().splitlines()

    # 构建双向映射字典
    id2label = {index: label for index, label in enumerate(all_labels)}
    label2id = {label: index for index, label in enumerate(all_labels)}

    # 4. 加载预训练分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id
    )

    # 把模型迁移到GPU/CPU设备
    model = model.to(device)

    print(model.config.label2id)
    model.save_pretrained(MODEL_DIR)

    # 5. 优化器
    optinizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 数据加载
    train_loader = get_dataloader(tokenizer)

    # 7. 定义训练配置
    train_config = TrainConfig()

    # 8. 定义训练器
    trainer = Trainer(model, train_loader, optinizer, device, train_config)

    # 8. 训练
    trainer.train()