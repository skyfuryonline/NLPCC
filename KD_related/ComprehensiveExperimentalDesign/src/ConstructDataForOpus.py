# 用于将数据集构造成instruction，input和output的形式

# 构造opus_books翻译数据集
from dataset import train_opus_dataset,val_opus_dataset
# 处理整个train数据集
alpaca_dataset = []
for item in train_opus_dataset:
    alpaca_format = {
        "instruction": "Translate the following text from English to French.",
        "input": item['translation']['fr'],
        "output": item['translation']['en']
    }
    alpaca_dataset.append(alpaca_format)
train_opus_dataset = alpaca_dataset
# 处理整个val数据集
alpaca_dataset = []
for item in val_opus_dataset:
    alpaca_format = {
        "instruction": "Translate the following text from English to French.",
        "input": item['translation']['fr'],
        "output": item['translation']['en']
    }
    alpaca_dataset.append(alpaca_format)
val_opus_dataset = alpaca_dataset

print(train_opus_dataset[0])
print(val_opus_dataset[0])
'''
{'instruction': 'Translate the following text from English to French.', 'input': 'Le grand Meaulnes', 'output': 'The Wanderer'}
{'instruction': 'Translate the following text from English to French.', 'input': "Il remerciait presque sa femme d'être allée chez un amant lorsqu'il croyait qu'elle se rendait chez un commissaire de police.", 'output': 'He almost thanked his wife for having gone to a sweetheart, when he thought her on her way to a commissary of police.'}
'''