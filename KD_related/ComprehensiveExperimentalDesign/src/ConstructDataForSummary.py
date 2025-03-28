# 用于将数据集构造成instruction，input和output的形式

# 构造summary数据集
from dataset import train_summary_dataset,val_summary_dataset
from datasets import Dataset

# 处理整个 train 数据集
alpaca_dataset = []
for item in train_summary_dataset:
    alpaca_format = {
        "instruction": "Summarize the following news article into 2-3 concise sentences, ensuring to include key details",
        "input": item['article'],
        "output": item['highlights']
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
train_dataset = Dataset.from_list(alpaca_dataset)

# 处理整个 val 数据集
alpaca_dataset = []
for item in val_summary_dataset:
    alpaca_format = {
        "instruction": "Summarize the following news article into 2-3 concise sentences, ensuring to include key details",
        "input": item['article'],
        "output": item['highlights']
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
val_dataset = Dataset.from_list(alpaca_dataset)
