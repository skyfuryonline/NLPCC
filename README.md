# NLPCC  
[BERT-EMD](https://github.com/lxk00/BERT-EMD)  
| EMD计算方法 | KL变化 | Bleu变化 |
|------------|---------|---------|
| 概率分布的差异计算逐点L1或L2距离（类似KL散度的方式） | 🔻-83.45% | 🔺+11.64% |
| 基于CDF和vocab_indice | 🔻-53.05% | 🔺+26.05% |
| 词嵌入相似度作为代价矩阵+top-k | 🔺+33.37% | 🔺+16.42% |
| 基于输入的嵌入构建代价矩阵，成本基于嵌入的欧几里得距离反映学生和教师模型在嵌入空间中的语义差异+top-k+分块 | 🔺+60.46% | 🔺+48.39% |
| 混合OT和KL(各占一半) | 🔻-0.80% | 🔺+34.02% |
| 混合OT和KL(3：7) | 🔻-2.45% | 🔺+35.48% |

# shared task 8: PESC  
[esconv](https://huggingface.co/datasets/thu-coai/esconv)  
[augesc](https://huggingface.co/datasets/thu-coai/augesc)  
[empathetic_dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues)  
