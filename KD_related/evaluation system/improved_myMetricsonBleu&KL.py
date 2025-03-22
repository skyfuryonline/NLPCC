'''
注意对比和原始的metrics对比，这个的优势是什么，为什么显著降低了KL；
'''

import torch
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import re

origin_student_path = "/root/shared-nvme/models/Qwen2.5-1.5B-bnb-4bit"
distill_student_path = "./results/checkpoint-620"
teacher_path = "/root/shared-nvme/models/Qwen2.5-7B"

# 配置参数（保持不变）
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 计算 KL 散度的函数（略有调整）
def compute_fkl(logits, teacher_logits, target, padding_id=-100, reduction="sum", temp=2.0):
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    # 调整序列长度，取较短的那个
    # min_seq_length = min(logits.shape[1], teacher_logits.shape[1], target.shape[1])
    # logits = logits[:, :min_seq_length, :]
    # teacher_logits = teacher_logits[:, :min_seq_length, :]
    # target = target[:, :min_seq_length]

    # 处理词汇表维度不匹配
    if logits.shape[-1] != teacher_logits.shape[-1]:
        teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    
    kl = (teacher_probs * (teacher_log_probs - log_probs)).sum(-1)  # 形状: [batch_size, seq_len]
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)  # 形状: [batch_size, seq_len]
        kl = kl.masked_fill(pad_mask, 0.0)  # 使用 masked_fill 而非 masked_fill_，避免修改原张量
        kl = kl.sum()
    return kl

# 加载模型（保持不变）
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

# 数据集格式化函数（保持不变）
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

# 加载验证集（保持不变）
val_dataset = load_dataset("yahma/alpaca-cleaned", split="train[2000:2100]")
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# 优化后的生成函数（返回 logits、response 和 input_ids）
def generate_response_batch_with_logits(model, tokenizer, instructions, input_texts, max_new_tokens=512):
    prompts = [alpaca_prompt.format(instr, inp, "") for instr, inp in zip(instructions, input_texts)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(model.device)
    
    with torch.no_grad():
        # 生成 logits
        outputs = model(**inputs)
        logits = outputs.logits  # 形状: [batch_size, seq_len, vocab_size]


        # 使用 models.generate 生成高质量响应
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            # num_beams=5,  # 使用 beam search
            # early_stopping=True
        )
        generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        responses = [extract_response(text) for text in generated_texts]
        # # 从 logits 生成 token IDs（贪婪解码）
        # generated_ids = torch.argmax(logits, dim=-1)
        # generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        # responses = [extract_response(text) for text in generated_texts]
    
    return logits, responses, inputs["input_ids"]  # 返回原始输入的 input_ids 作为 target

# 优化后的评估函数
def evaluate_response_only(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    kl_original, kl_distilled = [], []
    bleu_original, bleu_distilled = [],[]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher.to(device)
    original_student.to(device)
    distilled_student.to(device)
    
    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        instructions = batch["instruction"]
        inputs = batch["input"]
        true_responses = [extract_response(text) for text in batch["text"]]

        # 批量生成 logits、response 和 input_ids
        teacher_logits, teacher_responses, teacher_input_ids = generate_response_batch_with_logits(teacher, tokenizer, instructions, inputs)
        original_logits, original_responses, original_input_ids = generate_response_batch_with_logits(original_student, tokenizer, instructions, inputs)
        distilled_logits, distilled_responses, distilled_input_ids = generate_response_batch_with_logits(distilled_student, tokenizer, instructions, inputs)

        # 批量计算 KL 散度，使用原始 input_ids 作为 target
        for j in range(len(instructions)):
            kl_orig = compute_fkl(
                original_logits[j:j+1],
                teacher_logits[j:j+1],
                original_input_ids[j:j+1],  # 使用原始输入的 input_ids
                padding_id=tokenizer.pad_token_id,  # 确保 padding_id 正确
                temp=2.0
            )
            kl_original.append(kl_orig.item())

            kl_dist = compute_fkl(
                distilled_logits[j:j+1],
                teacher_logits[j:j+1],
                distilled_input_ids[j:j+1],  # 使用原始输入的 input_ids
                padding_id=tokenizer.pad_token_id,
                temp=2.0
            )
            kl_distilled.append(kl_dist.item())

        # 批量计算 BLEU 分数
        for j in range(len(instructions)):
            ref_tokens = true_responses[j].split()
            orig_pred_tokens = original_responses[j].split()
            dist_pred_tokens = distilled_responses[j].split()

            smoothie = SmoothingFunction().method1
            bleu_original.append(sentence_bleu([ref_tokens], orig_pred_tokens, smoothing_function=smoothie))
            bleu_distilled.append(sentence_bleu([ref_tokens], dist_pred_tokens, smoothing_function=smoothie))

    # 计算平均值
    avg_kl_original = np.mean(kl_original)
    avg_kl_distilled = np.mean(kl_distilled)
    avg_bleu_original = np.mean(bleu_original)
    avg_bleu_distilled = np.mean(bleu_distilled)

    return {
        "Original KL": avg_kl_original,
        "Distilled KL": avg_kl_distilled,
        "Original BLEU": avg_bleu_original,
        "Distilled BLEU": avg_bleu_distilled
    }

# 执行评估
results = evaluate_response_only(teacher, original_student, distilled_student, val_dataset, tokenizer)
print("评估结果（仅针对响应response部分）：")
print(f"原始模型 KL 散度: {results['Original KL']:.4f}")
print(f"蒸馏模型 KL 散度: {results['Distilled KL']:.4f}")
print(f"原始模型 BLEU 分数: {results['Original BLEU']:.4f}")
print(f"蒸馏模型 BLEU 分数: {results['Distilled BLEU']:.4f}")