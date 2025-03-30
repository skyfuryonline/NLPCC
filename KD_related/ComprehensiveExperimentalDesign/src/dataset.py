import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from datasets import load_dataset

data_path = "/home/lihao/lh/ComprehensiveExperimentalDesign/data"

# 加载并预处理Alpaca数据集
# train_alpaca_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir=data_path, split = "train[:2000]")
# val_alpaca_dataset = load_dataset("yahma/alpaca-cleaned",cache_dir=data_path, split = "train[-200:]")

# # 加载STS-B数据集
# train_stsb_dataset = load_dataset('glue', 'stsb',cache_dir=data_path,split = "train[:2000]")
# val_stsb_dataset =load_dataset('glue', 'stsb',cache_dir=data_path,split = "train[2000:2100]")

#加载opusbook
# train_opus_dataset = load_dataset("Helsinki-NLP/opus_books","en-fr",cache_dir=data_path,split = "train[:2000]")
# val_opus_dataset = load_dataset("Helsinki-NLP/opus_books","en-fr",cache_dir=data_path,split = "train[-200:]")

# 加载总结的数据集
# train_summary_dataset = load_dataset("abisee/cnn_dailymail",'3.0.0',cache_dir="../data",split='train[:2000]')
# val_summary_dataset = load_dataset("abisee/cnn_dailymail",'3.0.0',cache_dir='../data',split='train[-200:]')

# 加载QA数据集
# train_qa_dataset = load_dataset("rajpurkar/squad",cache_dir="../data",split='train[:2000]')
# val_qa_dataset = load_dataset("rajpurkar/squad",cache_dir="../data",split='train[-200:]')


# 加载IMDB数据集
train_imdb_dataset = load_dataset("stanfordnlp/imdb",cache_dir="../data",split='train[:2000]')
val_imdb_dataset = load_dataset("stanfordnlp/imdb",cache_dir="../data",split='train[-200:]')
