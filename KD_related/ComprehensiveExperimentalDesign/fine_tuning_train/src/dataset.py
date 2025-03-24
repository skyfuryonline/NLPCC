import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# 加载并预处理Alpaca数据集
from datasets import load_dataset
train_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir="../data",split='train[:3000]')
val_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir="../data", split = "train[2000:2100]")

# 加载STS-B数据集
# train_stsb_dataset = load_dataset('glue', 'stsb',cache_dir="../data",split = "train[:2000]")
# val_stsb_dataset =load_dataset('glue', 'stsb',cache_dir="../data",split = "train[2000:2100]")
