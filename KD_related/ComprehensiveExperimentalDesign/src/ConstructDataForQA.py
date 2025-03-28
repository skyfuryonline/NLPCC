# 构造qa数据集
from dataset import train_qa_dataset,val_qa_dataset
from datasets import Dataset

# 处理整个 train 数据集
alpaca_dataset = []
for item in train_qa_dataset:
    alpaca_format = {
        "instruction": "Please read the following article and answer the question based on its content. Provide a direct answer to the question.",
        "input": "Article: "+item['context']+"\n\nQuestion: "+item['question'],
        "output":item['answers']['text'][0]
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
train_dataset = Dataset.from_list(alpaca_dataset)

# 处理整个 val 数据集
alpaca_dataset = []
for item in val_qa_dataset:
    alpaca_format = {
        "instruction": "Please read the following article and answer the question based on its content. Provide a direct answer to the question.",
        "input":"Article: "+item['context']+"\n\nQuestion: "+item['question'] ,
        "output":item['answers']['text'][0]
    }
    alpaca_dataset.append(alpaca_format)

# 转换为 Dataset 类型
val_dataset = Dataset.from_list(alpaca_dataset)
# {'instruction': 'Please read the following article and answer the question based on its content.', 'input': 'Article: Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nQuestion: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?', 'output': 'Saint Bernadette Soubirous'}

