import time

import torch
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# 导入预训练分词器（后续文本分词使用）
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from configuration.config import *
from process.dataset import get_dataset
from dataclasses import dataclass
# 导入PyTorch批次加载器
from torch.utils.data import DataLoader


# 训练配置类
@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    save_steps: int = 100
    output_dir: str = MODEL_DIR
    log_dir: str = LOG_DIR

# 训练器类
class Trainer:
    # 初始化（修正拼写 __init__）
    def __init__(self, model, train_dataset, collate_fn , device, train_config=None):
        # 训练参数配置
        self.train_config = train_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device
        # 数据集和数据整理函数
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=self.train_config.learning_rate)
        # 全局迭代次数（运行的step数）
        self.step = 1
        # tensorboard写入器
        self.writer = SummaryWriter(
            log_dir=str(Path(self.train_config.log_dir) / time.strftime("%Y-%m-%d %H-%M-%S"))
        )
        # 全局最小的loss值
        self.min_loss = float('inf')

    # 定义内部方法：获取数据加载
    def _get_dataloader(self):
        # 将数据集格式直接转为PyTorch张量格式
        # self.train_dataset.set_format(type='torch')

        # 构建DataLoader迭代器
        dataloader = DataLoader(
            self.train_dataset,  # 传入加载好的数据集对象
            batch_size=self.train_config.batch_size,  # 设置单次迭代样本数量
            shuffle=True,  # 打乱训练集样本顺序
            collate_fn=self.collate_fn,  # 使用自定义填充函数组批次
        )
        return dataloader


    # 核心训练方法
    def train(self):
        self.model.train()
        # 获取训练集加载器
        dataloader = self._get_dataloader()

        # 双层for循环，外出遍历所有的epoch
        for epoch in range(self.train_config.epochs):
            # 内层循环遍一个epochs的所有批次
            for inputs in tqdm(dataloader, desc=f'[Epoch: {epoch+1}]'):
                # 调用一步（step)的训练过程，得到损失
                this_loss = self._train_one_step(inputs)

                # 判断如果达到save_steps就记录损失，判断是否保存模型
                if self.step % self.train_config.save_steps == 0:
                    # 记录损失
                    tqdm.write(f'[Epoch {epoch+1} | Step: {self.step}] Loss:{this_loss}')
                    self.writer.add_scalar('loss', this_loss, self.step)
                    # 判断是否保存模型
                    if this_loss < self.min_loss:
                        self.min_loss = this_loss
                        tqdm.write('保存最优模型...')
                        self.model.save_pretrained(self.train_config.output_dir)

                self.step += 1  #迭代次数+1

    # 一步训练
    def _train_one_step(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 前向传播
        outputs = self.model(**inputs)
        loss = outputs.loss
        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

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

    # print(model.config.label2id)
    model.save_pretrained(MODEL_DIR)

    # 5. 数据集和整理函数
    train_dataset = get_dataset()
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,  # 使用当前分词器匹配填充规则
        padding=True,  # 开启批次内动态填充
        return_tensors='pt',  # 输出格式为PyTorch tensor
    )

    # 6. 定义训练配置
    train_config = TrainConfig()

    # 7. 定义训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        collate_fn=collate_fn,
        device=device,
        train_config=train_config
    )

    # 8. 训练
    trainer.train()