# 用于将数据集构造成instruction，input和output的形式

# 构造wmt的翻译数据集，即fr-en的翻译对
from dataset import train_wmt_dataset,val_wmt_dataset

# l = train_wmt_dataset[0]['translation']['cs']
# print(l)
# #{'translation': {'cs': 'Následný postup na základě usnesení Parlamentu: viz zápis', 'en': "Action taken on Parliament's resolutions: see Minutes"}}

# 处理整个train数据集
alpaca_dataset = []
for item in train_wmt_dataset:
    alpaca_format = {
        "instruction": "Translate the following text from English to French.",
        "input": item['translation']['fr'],
        "output": item['translation']['en']
    }
    alpaca_dataset.append(alpaca_format)
train_wmt_dataset = alpaca_dataset

# 处理整个val数据集
alpaca_dataset = []
for item in val_wmt_dataset:
    alpaca_format = {
        "instruction": "Translate the following text from English to French.",
        "input": item['translation']['fr'],
        "output": item['translation']['en']
    }
    alpaca_dataset.append(alpaca_format)
val_wmt_dataset = alpaca_dataset

print(train_wmt_dataset[0])
print(val_wmt_dataset[0])