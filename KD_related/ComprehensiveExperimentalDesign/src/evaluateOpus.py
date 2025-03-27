import torch
from unsloth import FastLanguageModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import re
import os
from rouge_score import rouge_scorer  # 用于计算 Rouge-L 指标

from config import max_seq_length, dtype, load_in_4bit
# 配置参数
max_seq_length = max_seq_length
dtype = dtype
load_in_4bit = load_in_4bit

from config import temperature, reduction
temperature = temperature
reduction = reduction

def find_checkpoint():
    results_dir = "../models/results"
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

# 定义Alpaca格式的prompt模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

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

# 加载 Opus_books 验证集（假设从 dataset 模块导入）
from ConstructDataForOpus import val_opus_dataset
val_dataset = val_opus_dataset.map(formatting_prompts_func, batched=True)

# 生成函数（返回 response）
def generate_response_batch(model, tokenizer,instrcts, input_texts, max_new_tokens=512):
    prompts = [alpaca_prompt.format(instrct,inp, "") for inp,instrct in zip(input_texts,instrcts)]
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

    # # 保存模型的响应
    # from config import response_save_path
    # with open(response_save_path, "a", encoding="utf-8") as f:
    #     for response in responses:
    #         f.write(response + "\n")

    return responses

# 评估函数（仅计算 BLEU 和 Rouge-L）
def evaluate_opus_books(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    bleu_teacher, bleu_original, bleu_distilled = [], [], []
    rouge_teacher, rouge_original, rouge_distilled = [], [],[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    # 初始化 Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        inputs = batch["input"]
        instruct = batch["instruction"]
        true_responses = [extract_response(text) for text in batch["text"]]

        # 批量生成响应
        teacher_responses = generate_response_batch(teacher, tokenizer, instruct,inputs)
        original_responses = generate_response_batch(original_student, tokenizer, instruct,inputs)
        distilled_responses = generate_response_batch(distilled_student, tokenizer, instruct,inputs)

        # 计算 BLEU 和 Rouge-L
        for j in range(len(inputs)):
            ref_tokens = true_responses[j].split()
            teacher_pred_tokens = teacher_responses[j].split()
            orig_pred_tokens = original_responses[j].split()
            dist_pred_tokens = distilled_responses[j].split()

            smoothie = SmoothingFunction().method1
            bleu_teacher.append(sentence_bleu([ref_tokens], teacher_pred_tokens, smoothing_function=smoothie))
            bleu_original.append(sentence_bleu([ref_tokens], orig_pred_tokens, smoothing_function=smoothie))
            bleu_distilled.append(sentence_bleu([ref_tokens], dist_pred_tokens, smoothing_function=smoothie))

            rouge_teacher_score = scorer.score(true_responses[j], teacher_responses[j])['rougeL'].fmeasure
            rouge_orig_score = scorer.score(true_responses[j], original_responses[j])['rougeL'].fmeasure
            rouge_dist_score = scorer.score(true_responses[j], distilled_responses[j])['rougeL'].fmeasure
            rouge_teacher.append(rouge_teacher_score)
            rouge_original.append(rouge_orig_score)
            rouge_distilled.append(rouge_dist_score)

    # 计算各指标平均值
    avg_bleu_teacher = np.mean(bleu_teacher)
    avg_bleu_original = np.mean(bleu_original)
    avg_bleu_distilled = np.mean(bleu_distilled)
    avg_rouge_teacher = np.mean(rouge_teacher)
    avg_rouge_original = np.mean(rouge_original)
    avg_rouge_distilled = np.mean(rouge_distilled)

    return {
        "Teacher BLEU": avg_bleu_teacher,
        "Original BLEU": avg_bleu_original,
        "Distilled BLEU": avg_bleu_distilled,
        "Teacher Rouge-L": avg_rouge_teacher,
        "Original Rouge-L": avg_rouge_original,
        "Distilled Rouge-L": avg_rouge_distilled
    }

# 执行评估
results = evaluate_opus_books(teacher, original_student, distilled_student, val_dataset, tokenizer)
print("评估结果（针对 Opus_books 数据集）：")
print(f"教师模型 BLEU 分数: {results['Teacher BLEU']:.4f}")
print(f"原始模型 BLEU 分数: {results['Original BLEU']:.4f}")
print(f"蒸馏模型 BLEU 分数: {results['Distilled BLEU']:.4f}")
print(f"教师模型 Rouge-L 分数: {results['Teacher Rouge-L']:.4f}")
print(f"原始模型 Rouge-L 分数: {results['Original Rouge-L']:.4f}")
print(f"蒸馏模型 Rouge-L 分数: {results['Distilled Rouge-L']:.4f}")

# 同步到指定文件
log_dir = "/home/lihao/lh/ComprehensiveExperimentalDesign/results"
os.makedirs(log_dir, exist_ok=True)
result_file = os.path.join(log_dir, f"{run_name}.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write("评估结果（针对 Opus_books 数据集）：\n")
    f.write(f"教师模型 BLEU 分数: {results['Teacher BLEU']:.4f}\n")
    f.write(f"原始模型 BLEU 分数: {results['Original BLEU']:.4f}\n")
    f.write(f"蒸馏模型 BLEU 分数: {results['Distilled BLEU']:.4f}\n")
    f.write(f"教师模型 Rouge-L 分数: {results['Teacher Rouge-L']:.4f}\n")
    f.write(f"原始模型 Rouge-L 分数: {results['Original Rouge-L']:.4f}\n")
    f.write(f"蒸馏模型 Rouge-L 分数: {results['Distilled Rouge-L']:.4f}\n")
print(f"评估结果已保存到 {result_file}")