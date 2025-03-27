'''
使用了 Exact Match (EM) 和 F1 Score 作为评估指标，与 SQuAD 数据集的评估需求一致。
EM 判断预测答案是否与参考答案完全一致。
F1 基于词级别计算精确率和召回率，适用于短答案的评估。
'''

import torch
from unsloth import FastLanguageModel
import numpy as np
import re
import os
import string

from config import max_seq_length, dtype, load_in_4bit
# 配置参数
max_seq_length = max_seq_length
dtype = dtype
load_in_4bit = load_in_4bit

from config import temperature, reduction
temperature = temperature
reduction = reduction

def find_checkpoint():
    results_dir = "/home/lihao/lh/ComprehensiveExperimentalDesign/models/results"
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            return item_path
    return None  # 理论上不会执行到这里，因为保证有 checkpoint 文件夹

from config import origin_student_path, teacher_path
checkpoint_path = find_checkpoint()
origin_student_path = origin_student_path
distill_student_path = checkpoint_path
teacher_path = teacher_path

from config import run_name
run_name = run_name

# 加载模型
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

# 定义 Alpaca 格式的 prompt 模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# 规范化处理函数
def normalize_answer(s):
    """将文本转换为小写，去除标点、冠词和多余空白字符"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def extract_response(text):
    match = re.search(r"### Response:\n(.*?)(?=\n|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# 加载 SQuAD 验证集
from ConstructDataForQA import val_qa_dataset
val_dataset = val_qa_dataset

# 生成函数（返回 response）
def generate_response_batch(model, tokenizer, instructions, inputs, max_new_tokens=512):
    prompts = [alpaca_prompt.format(instruction, inp, "") for instruction, inp in zip(instructions, inputs)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(
        model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        responses = [extract_response(text) for text in generated_texts]

    return responses

# 评估函数（计算 EM 和 F1）
def evaluateQA(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    em_teacher, em_original, em_distilled = [], [], []
    f1_teacher, f1_original, f1_distilled = [], [], []

    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        instructions = batch["instruction"]
        inputs = batch["input"]
        true_responses = [extract_response(text) for text in batch["text"]]

        # 批量生成响应
        teacher_responses = generate_response_batch(teacher, tokenizer, instructions, inputs)
        original_responses = generate_response_batch(original_student, tokenizer, instructions, inputs)
        distilled_responses = generate_response_batch(distilled_student, tokenizer, instructions, inputs)

        # 计算 EM 和 F1
        for j in range(len(inputs)):
            ref = true_responses[j]
            teacher_pred = teacher_responses[j]
            orig_pred = original_responses[j]
            dist_pred = distilled_responses[j]

            # EM 计算
            em_teacher.append(1 if ref == teacher_pred else 0)
            em_original.append(1 if ref == orig_pred else 0)
            em_distilled.append(1 if ref == dist_pred else 0)

            # F1 计算
            ref_tokens = set(ref.split())
            teacher_pred_tokens = set(teacher_pred.split())
            orig_pred_tokens = set(orig_pred.split())
            dist_pred_tokens = set(dist_pred.split())

            common_teacher = ref_tokens.intersection(teacher_pred_tokens)
            common_orig = ref_tokens.intersection(orig_pred_tokens)
            common_dist = ref_tokens.intersection(dist_pred_tokens)

            if len(ref_tokens) == 0 or len(teacher_pred_tokens) == 0:
                f1_teacher.append(0)
            else:
                precision = len(common_teacher) / len(teacher_pred_tokens)
                recall = len(common_teacher) / len(ref_tokens)
                f1_teacher.append(2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)

            if len(ref_tokens) == 0 or len(orig_pred_tokens) == 0:
                f1_original.append(0)
            else:
                precision = len(common_orig) / len(orig_pred_tokens)
                recall = len(common_orig) / len(ref_tokens)
                f1_original.append(2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)

            if len(ref_tokens) == 0 or len(dist_pred_tokens) == 0:
                f1_distilled.append(0)
            else:
                precision = len(common_dist) / len(dist_pred_tokens)
                recall = len(common_dist) / len(ref_tokens)
                f1_distilled.append(2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0)

    # 计算各指标平均值
    avg_em_teacher = np.mean(em_teacher)
    avg_em_original = np.mean(em_original)
    avg_em_distilled = np.mean(em_distilled)
    avg_f1_teacher = np.mean(f1_teacher)
    avg_f1_original = np.mean(f1_original)
    avg_f1_distilled = np.mean(f1_distilled)

    return {
        "Teacher EM": avg_em_teacher,
        "Original EM": avg_em_original,
        "Distilled EM": avg_em_distilled,
        "Teacher F1": avg_f1_teacher,
        "Original F1": avg_f1_original,
        "Distilled F1": avg_f1_distilled
    }

# 执行评估
results = evaluateQA(teacher, original_student, distilled_student, val_dataset, tokenizer)
print("评估结果（针对 SQuAD 数据集）：")
print(f"教师模型 EM: {results['Teacher EM']:.4f}")
print(f"原始模型 EM: {results['Original EM']:.4f}")
print(f"蒸馏模型 EM: {results['Distilled EM']:.4f}")
print(f"教师模型 F1: {results['Teacher F1']:.4f}")
print(f"原始模型 F1: {results['Original F1']:.4f}")
print(f"蒸馏模型 F1: {results['Distilled F1']:.4f}")

# 同步到指定文件
log_dir = "/home/lihao/lh/ComprehensiveExperimentalDesign/results"
os.makedirs(log_dir, exist_ok=True)
result_file = os.path.join(log_dir, f"{run_name}_squad.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write("评估结果（针对 SQuAD 数据集）：\n")
    f.write(f"教师模型 EM: {results['Teacher EM']:.4f}\n")
    f.write(f"原始模型 EM: {results['Original EM']:.4f}\n")
    f.write(f"蒸馏模型 EM: {results['Distilled EM']:.4f}\n")
    f.write(f"教师模型 F1: {results['Teacher F1']:.4f}\n")
    f.write(f"原始模型 F1: {results['Original F1']:.4f}\n")
    f.write(f"蒸馏模型 F1: {results['Distilled F1']:.4f}\n")
print(f"评估结果已保存到 {result_file}")