from dataset import train_imdb_dataset,val_imdb_dataset
from datasets import Dataset
# 处理整个 train 数据集
alpaca_dataset = []
for item in train_imdb_dataset:
    alpaca_format = {
        "instruction": "Please classify the sentiment of the following movie review as either 1 (Positive) or 0 (Negative)",
        "input":item['text'],
        "output":item['label']
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
train_dataset = Dataset.from_list(alpaca_dataset)

# 处理整个 val 数据集
alpaca_dataset = []
for item in val_imdb_dataset:
    alpaca_format = {
        "instruction": "Please classify the sentiment of the following movie review as either 1 (Positive) or 0 (Negative)",
        "input": item['text'],
        "output": item['label']
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
val_dataset = Dataset.from_list(alpaca_dataset)