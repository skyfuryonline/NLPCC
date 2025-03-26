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

#加载opusbook
# train_opus_dataset = load_dataset("Helsinki-NLP/opus_books","en-fr",cache_dir="../data",split = "train[:2000]")
# val_opus_dataset = load_dataset("Helsinki-NLP/opus_books","en-fr",cache_dir="../data",split = "train[-200:]")


# 加载总结的数据集
train_summary_dataset = load_dataset("abisee/cnn_dailymail",'3.0.0',cache_dir="../data",split='train[:2000]')
val_summary_dataset = load_dataset("abisee/cnn_dailymail",'3.0.0',cache_dir='../data',split='train[-200:]')


# 加载QA数据集
train_qa_dataset = load_dataset("rajpurkar/squad",cache_dir="../data",split='train[:2000]')
val_qa_dataset = load_dataset("rajpurkar/squad",cache_dir="../data",split='train[-200:]')
# {'id': '5733be284776f41900661182', 'title': 'University_of_Notre_Dame', 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 'answers': {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}}