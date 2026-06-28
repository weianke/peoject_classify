# 导入全局路径配置
from configuration.config import *
# 导入数据集加载工具与类别编码工具
from datasets import load_dataset, ClassLabel
# 导入预训练分词器（后续文本分词使用）
from transformers import AutoTokenizer


def preprocess():
    # ========== 步骤1：加载TSV格式数据集 ==========
    dataset_dict = load_dataset(
        'csv',  # 用csv解析器读取制表符分隔的txt文件
        data_files={
            'train': str(RAW_DATA_DIR / RAW_TRAIN_DATA),   # 训练集文件路径
            'valid': str(RAW_DATA_DIR / RAW_VALID_DATA),   # 验证集文件路径
            'test': str(RAW_DATA_DIR / RAW_TEST_DATA)      # 测试集文件路径
        },
        delimiter='\t'  # 指定分隔符为制表符\t，对应两列：label、text_a
    )

    # 从训练集中提取全部不重复的类别标签，并排序
    all_labels = sorted(set(dataset_dict['train']['label']))
    # 构建类别转换器：实现文字标签 ↔ 数字ID的互相转换
    label_cls = ClassLabel(names=all_labels)

    # ========== 步骤2：新增数字标签列，保留原始文字label ==========
    def add_label_id(example):
        # 将文字类别转为数字ID，存入新字段label_id
        example["label_id"] = label_cls.str2int(example["label"])
        return example

    # 遍历全量数据，批量执行字段新增操作
    dataset_dict = dataset_dict.map(add_label_id)


    # 保存类别id -> label映射关系
    with open( MODEL_DIR / LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_labels))

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def batch_encode(example):
        # 对整批文本进行分词
        inputs = tokenizer(example['text_a'], truncation=True)
        # 模型要求labels为数字，使用label_id，不要用文字label
        inputs['labels'] = example['label_id']
        return inputs

    # 处理标题文本，得到模型输入
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['label', 'text_a', 'label_id'])

    # 打印前3条数据，验证字段结果
    print(dataset_dict['train'][0:3])

    # 保存数据集
    dataset_dict.save_to_disk( PROCESSED_DATA_DIR )

if __name__ == '__main__':
    # 执行数据预处理函数
    preprocess()