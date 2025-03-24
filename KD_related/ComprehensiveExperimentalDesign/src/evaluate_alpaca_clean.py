import torch
from unsloth import FastLanguageModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import re
import os
from rouge_score import rouge_scorer  # 新增：用于计算 Rouge-L 指标



from config import max_seq_length,dtype,load_in_4bit
# 配置参数（保持不变）
max_seq_length = max_seq_length
dtype = dtype
load_in_4bit = load_in_4bit

# # 蒸馏参数
# temperature = 2.0
# reduction = "sum"
# # topk = None

from config import temperature,reduction
temperature = temperature
reduction = reduction
# topk = None


def find_checkpoint():
    results_dir = "../models/results"
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            return item_path
    return None  # 理论上不会执行到这里，因为保证有 checkpoint 文件夹


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
# run_name = "OT_KD"
from config import run_name
run_name = run_name


# 优化后的 compute_fkl，支持批量计算（同时支持 "none"、"sum"、"mean" 三种 reduction 模式）
def compute_fkl(logits, teacher_logits, target, padding_id=-100, reduction="sum", temp=2.0):
    # 归一化温度
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    # 处理词汇表维度不匹配
    if logits.shape[-1] != teacher_logits.shape[-1]:
        teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

    # 计算 softmax 与 log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    # 批量计算 KL 散度，形状：[batch_size, seq_len]
    kl = (teacher_probs * (teacher_log_probs - log_probs)).sum(dim=-1)

    # 根据 reduction 参数进行返回
    if reduction == "none":
        return kl  # 每个 token 的 KL 值
    elif reduction == "sum":
        # 创建 padding mask
        pad_mask = target.eq(padding_id)  # [batch_size, seq_len]
        kl = kl.masked_fill(pad_mask, 0.0)
        # 对每个样本的 token 求和，返回形状：[batch_size]
        return kl.sum(dim=1)
    elif reduction == "mean":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill(pad_mask, 0.0)
        # 计算非 padding token 数量，避免除零
        token_counts = (~pad_mask).sum(dim=1).clamp(min=1)
        return kl.sum(dim=1) / token_counts
    else:
        raise ValueError("Unsupported reduction mode.")


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
from dataset import val_alpaca_dataset
val_dataset = val_alpaca_dataset.map(formatting_prompts_func, batched=True)


# 优化后的生成函数（返回 logits、response 和 input_ids）
def generate_response_batch_with_logits(model, tokenizer, instructions, input_texts, max_new_tokens=512):
    prompts = [alpaca_prompt.format(instr, inp, "") for instr, inp in zip(instructions, input_texts)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(
        model.device)

    with torch.no_grad():
        # 生成 logits
        outputs = model(**inputs)
        logits = outputs.logits  # 形状: [batch_size, seq_len, vocab_size]

        # 使用 model.generate 生成高质量响应
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
        responses = [extract_response(text) for text in generated_texts]

    return logits, responses, inputs["input_ids"]  # 返回原始输入的 input_ids 作为 target


# 在评估函数中，我们将 KL 计算从逐样本循环改为对整个 batch 一次性计算：
def evaluate_response_only(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    kl_original, kl_distilled = [], []
    bleu_original, bleu_distilled = [], []
    rouge_original, rouge_distilled = [], []  # 新增：保存 Rouge-L 分数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    # 初始化 Rouge scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        instructions = batch["instruction"]
        inputs = batch["input"]
        true_responses = [extract_response(text) for text in batch["text"]]

        # 批量生成 logits、响应和 input_ids
        teacher_logits, teacher_responses, teacher_input_ids = generate_response_batch_with_logits(teacher, tokenizer,
                                                                                                   instructions, inputs)
        original_logits, original_responses, original_input_ids = generate_response_batch_with_logits(original_student,
                                                                                                      tokenizer,
                                                                                                      instructions,
                                                                                                      inputs)
        distilled_logits, distilled_responses, distilled_input_ids = generate_response_batch_with_logits(
            distilled_student, tokenizer, instructions, inputs)

        # 优化：直接对整个 batch 计算 KL 散度（返回 [batch_size] 的向量）
        kl_orig_batch = compute_fkl(original_logits, teacher_logits, original_input_ids,
                                    padding_id=tokenizer.pad_token_id, reduction=reduction, temp=temperature)
        kl_dist_batch = compute_fkl(distilled_logits, teacher_logits, distilled_input_ids,
                                    padding_id=tokenizer.pad_token_id, reduction=reduction, temp=temperature)
        kl_original.extend(kl_orig_batch.cpu().tolist())
        kl_distilled.extend(kl_dist_batch.cpu().tolist())

        # 对 BLEU 和 Rouge-L 仍需逐样本计算（字符串处理部分）
        for j in range(len(instructions)):
            ref_tokens = true_responses[j].split()
            orig_pred_tokens = original_responses[j].split()
            dist_pred_tokens = distilled_responses[j].split()

            smoothie = SmoothingFunction().method1
            bleu_original.append(sentence_bleu([ref_tokens], orig_pred_tokens, smoothing_function=smoothie))
            bleu_distilled.append(sentence_bleu([ref_tokens], dist_pred_tokens, smoothing_function=smoothie))

            rouge_orig = scorer.score(true_responses[j], original_responses[j])['rougeL'].fmeasure
            rouge_dist = scorer.score(true_responses[j], distilled_responses[j])['rougeL'].fmeasure
            rouge_original.append(rouge_orig)
            rouge_distilled.append(rouge_dist)

    # 计算各指标平均值
    avg_kl_original = np.mean(kl_original)
    avg_kl_distilled = np.mean(kl_distilled)
    avg_bleu_original = np.mean(bleu_original)
    avg_bleu_distilled = np.mean(bleu_distilled)
    avg_rouge_original = np.mean(rouge_original)
    avg_rouge_distilled = np.mean(rouge_distilled)

    return {
        "Original KL": avg_kl_original,
        "Distilled KL": avg_kl_distilled,
        "Original BLEU": avg_bleu_original,
        "Distilled BLEU": avg_bleu_distilled,
        "Original Rouge-L": avg_rouge_original,
        "Distilled Rouge-L": avg_rouge_distilled
    }


# 执行评估
results = evaluate_response_only(teacher, original_student, distilled_student, val_dataset, tokenizer)
print("评估结果（仅针对响应）：")
print(f"原始模型 KL 散度: {results['Original KL']:.4f}")
print(f"蒸馏模型 KL 散度: {results['Distilled KL']:.4f}")
print(f"原始模型 BLEU 分数: {results['Original BLEU']:.4f}")
print(f"蒸馏模型 BLEU 分数: {results['Distilled BLEU']:.4f}")
print(f"原始模型 Rouge-L 分数: {results['Original Rouge-L']:.4f}")
print(f"蒸馏模型 Rouge-L 分数: {results['Distilled Rouge-L']:.4f}")


# 同步到制定文件内
log_dir = "../results"
os.makedirs(log_dir, exist_ok=True)
result_file = os.path.join(log_dir, f"{run_name}.txt")
# 写入评估结果到文件
with open(result_file, "w", encoding="utf-8") as f:
    f.write("评估结果（仅针对响应）：\n")
    f.write(f"原始模型 KL 散度: {results['Original KL']:.4f}\n")
    f.write(f"蒸馏模型 KL 散度: {results['Distilled KL']:.4f}\n")
    f.write(f"原始模型 BLEU 分数: {results['Original BLEU']:.4f}\n")
    f.write(f"蒸馏模型 BLEU 分数: {results['Distilled BLEU']:.4f}\n")
    f.write(f"原始模型 Rouge-L 分数: {results['Original Rouge-L']:.4f}\n")
    f.write(f"蒸馏模型 Rouge-L 分数: {results['Distilled Rouge-L']:.4f}\n")
print(f"评估结果已保存到 {result_file}")