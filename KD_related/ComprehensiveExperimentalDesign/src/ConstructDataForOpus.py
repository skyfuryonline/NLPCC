# 用于将数据集构造成instruction，input和output的形式

# 构造opus_books翻译数据集
from dataset import train_opus_dataset,val_opus_dataset
from datasets import Dataset

# 处理整个 train 数据集
alpaca_dataset = []
for item in train_opus_dataset:
    alpaca_format = {
        "instruction": "Translate the following text from English to French.",
        "input": item['translation']['fr'],
        "output": item['translation']['en']
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
train_opus_dataset = Dataset.from_list(alpaca_dataset)

# 处理整个 val 数据集
alpaca_dataset = []
for item in val_opus_dataset:
    alpaca_format = {
        "instruction": "Translate the following text from English to French.",
        "input": item['translation']['fr'],
        "output": item['translation']['en']
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
val_opus_dataset = Dataset.from_list(alpaca_dataset)