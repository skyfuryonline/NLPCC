'''
应该实现一个基于LLM的评估方式
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


# 加载特定验证集
from ConstructDataForQA import val_qa_dataset
val_dataset = val_qa_dataset.map(formatting_prompts_func, batched=True)


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


from config import client,generate
#通过config传入指定的llm
llm  = client

def extract_score(eval_text):
    """
    从 LLM 返回的文本中提取评分，支持换行，并限制范围在 0-10 之间。
    如果模型直接返回了 score，则直接提取。
    """
    try:
        # 如果 eval_text 本身就是一个数字字符串，直接转换
        if isinstance(eval_text, (int, float)):
            return max(0, min(float(eval_text), 10))

        eval_text = str(eval_text).strip()
        if eval_text.replace('.', '', 1).isdigit():  # 处理字符串形式的纯数字
            return max(0, min(float(eval_text), 10))

        # 允许 "Score:" 之后换行再写分数
        score_pattern = r"(?:###\s*Score:|Score:|Final Score:)\s*\n?(\d+\.?\d*)"
        match = re.search(score_pattern, eval_text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(0, min(score, 10))  # 限制评分在 0-10 之间
        else:
            print(f"Warning: No score found in: {eval_text}. Defaulting to 5.0")
            return 5.0
    except (ValueError, TypeError):
        print(f"Warning: Invalid score format in: {eval_text}. Defaulting to 5.0")
        return 5.0



# Alpaca Prompt 模板（优化版，便于提取 Score）
ALPACA_PROMPT = """\
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}

### Score:
[Your score]
"""

# 评估函数（简化版，仅返回评分）
def evaluate_based_on_LLM(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    scores_teacher, scores_original, scores_distilled = [], [], []

    # 设置模型为推理模式
    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        instructions = batch["instruction"]
        inputs = batch["output"]

        # 批量生成响应
        teacher_responses = generate_response_batch(teacher, tokenizer, instructions, inputs)
        original_responses = generate_response_batch(original_student, tokenizer, instructions, inputs)
        distilled_responses = generate_response_batch(distilled_student, tokenizer, instructions, inputs)

        # 对每个响应进行评分
        for j in range(len(inputs)):
            instruction = instructions[j]
            input_text = inputs[j]

            # 构造 Alpaca Prompt 并使用外部 LLM 评分
            teacher_prompt = ALPACA_PROMPT.format(
                instruction=instruction,
                input=input_text,
                response=teacher_responses[j],
            )
            original_prompt = ALPACA_PROMPT.format(
                instruction=instruction,
                input=input_text,
                response=original_responses[j],
            )
            distilled_prompt = ALPACA_PROMPT.format(
                instruction=instruction,
                input=input_text,
                response=distilled_responses[j],
            )

            # 调用外部 LLM 进行评分
            teacher_eval = generate(llm,prompt=teacher_prompt, max_length=50)  # 缩短 max_length，因为只需要分数
            original_eval =generate(llm,prompt=original_prompt, max_length=50)
            distilled_eval = generate(llm,prompt=distilled_prompt, max_length=50)

            # 提取评分
            teacher_score = extract_score(teacher_eval)
            original_score = extract_score(original_eval)
            distilled_score = extract_score(distilled_eval)

            scores_teacher.append(teacher_score)
            scores_original.append(original_score)
            scores_distilled.append(distilled_score)

    # 计算平均分
    avg_score_teacher = np.mean(scores_teacher)
    avg_score_original = np.mean(scores_original)
    avg_score_distilled = np.mean(scores_distilled)

    return {
        "Teacher Score": avg_score_teacher,
        "Original Score": avg_score_original,
        "Distilled Score": avg_score_distilled
    }

# 执行评估
results = evaluate_based_on_LLM(teacher, original_student, distilled_student, val_dataset, tokenizer)
print("评估结果（基于 LLM 的评分）：")
print(f"教师模型得分: {results['Teacher Score']:.2f}/10")
print(f"原始模型得分: {results['Original Score']:.2f}/10")
print(f"蒸馏模型得分: {results['Distilled Score']:.2f}/10")

# 同步到指定文件
log_dir = "/home/lihao/lh/ComprehensiveExperimentalDesign/results"
os.makedirs(log_dir, exist_ok=True)
result_file = os.path.join(log_dir, f"{run_name}.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write("评估结果（基于 LLM 的评分）：\n")
    f.write(f"教师模型得分: {results['Teacher Score']:.2f}/10\n")
    f.write(f"原始模型得分: {results['Original Score']:.2f}/10\n")
    f.write(f"蒸馏模型得分: {results['Distilled Score']:.2f}/10\n")
print(f"评估结果已保存到 {result_file}")

