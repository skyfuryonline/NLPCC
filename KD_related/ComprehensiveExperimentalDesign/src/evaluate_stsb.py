import torch
from unsloth import FastLanguageModel
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
import re

# # 配置参数（保持与原代码一致）
# max_seq_length = 2048
# dtype = None
# load_in_4bit = True
from config import max_seq_length,dtype,load_in_4bit
# 配置参数（保持不变）
max_seq_length = max_seq_length
dtype = dtype
load_in_4bit = load_in_4bit


# # 蒸馏参数
# temperature = 2.0
# reduction = "sum"
# topk = None


# 查找检查点路径（与原代码一致）
def find_checkpoint():
    results_dir = "../models/results"
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            return item_path
    return None


# checkpoint_path = find_checkpoint()
# origin_student_path = "../models/unsloth/Qwen2.5-1.5B"
# distill_student_path = checkpoint_path
# teacher_path = "../models/unsloth/Qwen2.5-7B"

from config import origin_student_path,teacher_path
checkpoint_path = find_checkpoint()
origin_student_path = origin_student_path
distill_student_path = checkpoint_path
teacher_path = teacher_path

# # 这次记录的名字
# run_name = "STS_B_Eval"
from config import run_name
run_name = run_name

# 加载模型（与原代码一致）
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name=teacher_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
teacher.eval()

original_student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
original_student.eval()

distilled_student, _ = FastLanguageModel.from_pretrained(
    model_name=distill_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
distilled_student.eval()

# STS-B 数据集格式化函数
sts_prompt = """### Sentence 1:
{}

### Sentence 2:
{}

### Similarity Score:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    sentence1 = examples["sentence1"]
    sentence2 = examples["sentence2"]
    scores = examples["score"]
    texts = []
    for s1, s2, score in zip(sentence1, sentence2, scores):
        text = sts_prompt.format(s1, s2, score) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}


# 提取句子对和真实分数
def extract_sentences_and_score(text):
    s1_match = re.search(r"### Sentence 1:\n(.*?)(?=\n|$)", text, re.DOTALL)
    s2_match = re.search(r"### Sentence 2:\n(.*?)(?=\n|$)", text, re.DOTALL)
    score_match = re.search(r"### Similarity Score:\n(.*?)(?=\n|$)", text, re.DOTALL)
    s1 = s1_match.group(1).strip() if s1_match else ""
    s2 = s2_match.group(1).strip() if s2_match else ""
    score = float(score_match.group(1).strip()) if score_match else 0.0
    return s1, s2, score


# 加载 STS-B 验证集（假设从 dataset 模块加载）
from dataset import val_stsb_dataset  # 假设有一个 sts_b_dataset

val_stsb_dataset = val_stsb_dataset.map(formatting_prompts_func, batched=True)


# 计算句子对嵌入的函数
def get_sentence_embeddings(model, tokenizer, sentence1, sentence2, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []

    for i in range(0, len(sentence1), batch_size):
        batch_s1 = sentence1[i:i + batch_size]
        batch_s2 = sentence2[i:i + batch_size]
        inputs = tokenizer(batch_s1 + batch_s2, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_seq_length).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # 使用 CLS token 的表示（假设是第一个 token）
            batch_embeddings = outputs.logits[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    embeddings = np.concatenate(embeddings, axis=0)
    s1_emb = embeddings[:len(sentence1)]
    s2_emb = embeddings[len(sentence1):]
    return s1_emb, s2_emb


# 计算余弦相似度
def cosine_similarity(emb1, emb2):
    dot_product = np.sum(emb1 * emb2, axis=1)
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)
    return dot_product / (norm1 * norm2 + 1e-10)


# 评估函数
def evaluate_sts_b(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    sentence1_list, sentence2_list, true_scores = [], [], []
    for text in dataset["text"]:
        s1, s2, score = extract_sentences_and_score(text)
        sentence1_list.append(s1)
        sentence2_list.append(s2)
        true_scores.append(score)

    # 获取嵌入
    teacher_s1_emb, teacher_s2_emb = get_sentence_embeddings(teacher, tokenizer, sentence1_list, sentence2_list)
    orig_s1_emb, orig_s2_emb = get_sentence_embeddings(original_student, tokenizer, sentence1_list, sentence2_list)
    dist_s1_emb, dist_s2_emb = get_sentence_embeddings(distilled_student, tokenizer, sentence1_list, sentence2_list)

    # 计算余弦相似度
    teacher_sim = cosine_similarity(teacher_s1_emb, teacher_s2_emb)
    orig_sim = cosine_similarity(orig_s1_emb, orig_s2_emb)
    dist_sim = cosine_similarity(dist_s1_emb, dist_s2_emb)

    # 计算 Pearson 和 Spearman 相关系数
    orig_pearson, _ = pearsonr(true_scores, orig_sim)
    dist_pearson, _ = pearsonr(true_scores, dist_sim)
    orig_spearman, _ = spearmanr(true_scores, orig_sim)
    dist_spearman, _ = spearmanr(true_scores, dist_sim)

    # 计算与教师模型的 KL 散度（可选，仅用于参考）
    kl_orig = np.mean((teacher_sim - orig_sim) ** 2)  # 简化为 MSE 代替 KL
    kl_dist = np.mean((teacher_sim - dist_sim) ** 2)

    return {
        "Original Pearson": orig_pearson,
        "Distilled Pearson": dist_pearson,
        "Original Spearman": orig_spearman,
        "Distilled Spearman": dist_spearman,
        "Original KL (wrt Teacher)": kl_orig,
        "Distilled KL (wrt Teacher)": kl_dist
    }


# 执行评估
results = evaluate_sts_b(teacher, original_student, distilled_student, val_stsb_dataset, tokenizer)
print("STS-B 评估结果：")
print(f"原始模型 Pearson 相关系数: {results['Original Pearson']:.4f}")
print(f"蒸馏模型 Pearson 相关系数: {results['Distilled Pearson']:.4f}")
print(f"原始模型 Spearman 相关系数: {results['Original Spearman']:.4f}")
print(f"蒸馏模型 Spearman 相关系数: {results['Distilled Spearman']:.4f}")
print(f"原始模型与教师模型 KL: {results['Original KL (wrt Teacher)']:.4f}")
print(f"蒸馏模型与教师模型 KL: {results['Distilled KL (wrt Teacher)']:.4f}")

# 同步到文件
log_dir = "../results"
os.makedirs(log_dir, exist_ok=True)
result_file = os.path.join(log_dir, f"{run_name}.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write("STS-B 评估结果：\n")
    f.write(f"原始模型 Pearson 相关系数: {results['Original Pearson']:.4f}\n")
    f.write(f"蒸馏模型 Pearson 相关系数: {results['Distilled Pearson']:.4f}\n")
    f.write(f"原始模型 Spearman 相关系数: {results['Original Spearman']:.4f}\n")
    f.write(f"蒸馏模型 Spearman 相关系数: {results['Distilled Spearman']:.4f}\n")
    f.write(f"原始模型与教师模型 KL: {results['Original KL (wrt Teacher)']:.4f}\n")
    f.write(f"蒸馏模型与教师模型 KL: {results['Distilled KL (wrt Teacher)']:.4f}\n")
print(f"评估结果已保存到 {result_file}")