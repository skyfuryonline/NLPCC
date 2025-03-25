# # 下载数据集
# dataset = load_dataset("yahma/alpaca-cleaned", cache_dir="../data")
# print(dataset)
# '''
# DatasetDict({
#     train: Dataset({
#         features: ['output', 'input', 'instruction'],
#         num_rows: 51760
#     })
# })
# '''
# print(dataset['train'][-1])

import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from datasets import load_dataset


# 加载并预处理Alpaca数据集
# train_alpaca_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir="../data", split = "train[:2000]")
# val_alpaca_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir="../data", split = "train[2000:2100]")

# print(train_dataset)
# print(val_dataset)
'''
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 2000
})
Dataset({
    features: ['output', 'input', 'instruction'],
    num_rows: 1000
})
'''

# 加载STS-B数据集
# train_stsb_dataset = load_dataset('glue', 'stsb',cache_dir="../data",split = "train[:2000]")
# val_stsb_dataset =load_dataset('glue', 'stsb',cache_dir="../data",split = "train[2000:2100]")
# print(train_stsb_dataset)
# print(val_stsb_dataset)
'''
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 2000
})
Dataset({
    features: ['sentence1', 'sentence2', 'label', 'idx'],
    num_rows: 100
})
'''


train_wmt_dataset = load_dataset('wmt/wmt14','fr-en',cache_dir="../data",split = "train[:2000]")
val_wmt_dataset = load_dataset('wmt/wmt14','fr-en',cache_dir="../data",split = "train[-200:]")