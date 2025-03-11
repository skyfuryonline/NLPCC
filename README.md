# NLPCC  
[BERT-EMD](https://github.com/lxk00/BERT-EMD)  
| 方法                           | KL变化   | Bleu变化   | 与baseline的差异（KL） | 与baseline的差异（Bleu） |
|--------------------------------|----------|------------|------------------------|--------------------------|
| EMD计算                        | -83.45%  | 11.64%     | +3.54% ↑               | -4.92% ↓                 |
| 概率分布的差异计算逐点L1或L2距离 | -53.05%  | 26.05%     | +33.94% ↑              | +9.49% ↑                 |
| 基于CDF和vocab_indice            | 33.37%   | 16.42%     | +120.36% ↑             | -0.14% ↓                 |
| 词嵌入相似度作为代价矩阵+top-k   | 60.46%   | 48.39%     | +147.45% ↑             | +31.83% ↑                |
| 基于输入的嵌入构建代价矩阵，成本基于嵌入的欧几里得距离反映学生和教师模型在嵌入空间中的语义差异+top-k+分块 | -0.80%   | 34.02%     | +86.19% ↑              | +17.46% ↑                |
| 混合OT和KL(各占一半)            | -2.45%   | 35.48%     | +84.54% ↑              | +18.92% ↑                |
| 混合OT和KL(3：7)                | -86.99%  | 16.56%     | 0%                     | 0%                       |
| 原始KL蒸馏（0.5*CELoss+0.5*KL）  | -86.99%  | 16.56%     | —                      | —                        |


补充说明：
- 直接用概率分布的差异（student_probs - teacher_probs），计算逐点L1或L2距离，简单的概率分布L1/L2损失，与Wasserstein距离的几何特性偏离
- 基于累积分布函数（CDF），计算CDF差异并用vocab_indices加权，更接近Wasserstein距离的定义（衡量分布间的“搬运成本”）引入 vocab 索引的距离信息，从而提供了一种自然的 metric 结构，使得 Wasserstein 计算在 NLP 任务（如词汇分布匹配）中更具物理意义。即vocab indice充当了代价矩阵
- 提取教师模型和学生模型top-k的词嵌入并用投影矩阵对齐维度。用1减余弦相似度矩阵充当代价矩阵。用Sinkhorn 算法构造运输矩阵T。最终的 OT 损失通过计算 𝑇 和代价矩阵的元素乘积的和获得
- 成本基于嵌入的欧几里得距离，反映学生和教师模型在嵌入空间中的语义差异。不显式构造完整的 (topk, topk)矩阵，而是基于样本点（嵌入）的距离动态计算 Sinkhorn 距离，节省内存并加速计算。
- 使用可训练的投影矩阵进行维度对齐。代价矩阵是计算学生和教师的 top-k 词向量的欧几里得距离，再用 Sinkhorn 计算最优传输矩阵。计算整个词汇表上的概率分布之间的 KL 散度

# shared task 8: PESC  
[esconv](https://huggingface.co/datasets/thu-coai/esconv)  
[augesc](https://huggingface.co/datasets/thu-coai/augesc)  
[empathetic_dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues)  
