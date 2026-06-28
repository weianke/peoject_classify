# 从本地磁盘读取预处理好的数据集
from datasets import load_from_disk
# 加载分词器与动态填充工具
from transformers import AutoTokenizer, DataCollatorWithPadding
# 导入PyTorch批次加载器
from torch.utils.data import DataLoader
# 导入项目全局路径与参数配置
from configuration.config import *


# 构造训练/验证集的数据迭代器
def get_dataloader(tokenizer, ds_type='train'):
    # 拼接数据集文件夹路径
    path = str(PROCESSED_DATA_DIR / ds_type)
    # 读取磁盘上保存的arrow格式数据集
    dataset = load_from_disk(path)

    # 将数据集格式直接转为PyTorch张量格式
    dataset.set_format(type='torch')

    # 定义批次内动态填充器，自动补齐本批次句子长度
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer,        # 使用当前分词器匹配填充规则
        padding=True,                # 开启批次内动态填充
        return_tensors='pt',         # 输出格式为PyTorch tensor
    )
    # 构建DataLoader迭代器
    dataloader = DataLoader(
        dataset,                     # 传入加载好的数据集对象
        batch_size=BATCH_SIZE,       # 设置单次迭代样本数量
        shuffle=True,                # 打乱训练集样本顺序
        collate_fn=collate_fn        # 使用自定义填充函数组批次
    )
    return dataloader


# 脚本入口：测试数据加载是否正常
if __name__ == '__main__':
    # 加载预训练BERT分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    # 生成数据迭代器
    dataloader = get_dataloader(tokenizer)

    # 只读取第一个批次，打印张量维度
    for batch in dataloader:
        for k, v in batch.items():
            # 打印字段名 + 张量形状
            print(k, v)
        break