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


# 加载并预处理Alpaca数据集
from datasets import load_dataset
train_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir="../data", split = "train[:2000]")
val_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir="../data", split = "train[2000:3000]")

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