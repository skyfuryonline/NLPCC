import torch
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import re

# 配置参数
max_seq_length = 2048 # 模型能够处理的最大序列长度。
dtype = None # 数据类型，如果未指定，则自动选择合适的类型（例如，使用 bfloat16 如果支持的话）
load_in_4bit = True

# 已定义的 compute_fkl 函数（复用）
def compute_fkl(logits, teacher_logits, target, padding_id=-100, reduction="sum", temp=2.0):
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    # 调整序列长度，取较短的那个
    min_seq_length = min(logits.shape[1], teacher_logits.shape[1])
    logits = logits[:, :min_seq_length, :]
    teacher_logits = teacher_logits[:, :min_seq_length, :]
    target = target[:, :min_seq_length]  # 同时调整 target 的长度

    # 处理词汇表维度不匹配，仅截断教师模型的 logits
    if isinstance(logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
            

    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    kl = (teacher_probs * (teacher_log_probs - log_probs))
    kl = kl.sum(-1)
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()
    return kl
    
# 加载教师模型
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/shared-nvme/model/Qwen2.5-7B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
teacher.eval()

# 加载原始小模型
original_student, _ = FastLanguageModel.from_pretrained(
    model_name="/root/shared-nvme/model/Qwen2.5-1.5B-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
original_student.eval()

# 加载蒸馏后的小模型（替换为实际检查点路径）
distilled_student, _ = FastLanguageModel.from_pretrained(
    model_name="./results/checkpoint-620",  # 替换为实际路径
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
distilled_student.eval()

# 数据集格式化函数（复用）
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 获取结束符

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# 提取 response 的函数
def extract_response(text):
    '''
    :param text:输入的是模型的输出
    :return: 返回的是输出中的response部分（利用正则）
    注意最后返回的，如果text字段中没有response，则正则会返回None，据此判断，而验证集中的字段是拼在prompt里面的，保证一定有response部分
    '''
    match = re.search(r"### Response:\n(.*?)(?=\n|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# 加载验证集
val_dataset = load_dataset("yahma/alpaca-cleaned", split="train[2000:2100]")
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)


# 生成 response 的函数（支持批量）
def generate_response_batch(model, tokenizer, instructions, input_texts, max_new_tokens=512):
    '''

    :param model: 预训练的语言模型qwen
    :param tokenizer:分词器
    :param instructions:
    :param input_texts:
    :param max_new_tokens:512
    :return:获取llm对prompt生成的内容，并提取其中的response用于后续评估，返回类型是response list
    （感觉没有真正的批量，应该使用bath_decode才是）
    '''
    prompts = [alpaca_prompt.format(instr, inp, "") for instr, inp in zip(instructions, input_texts)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return [extract_response(text) for text in generated_texts]

# 评估函数（批量生成版本）
def evaluate_response_only(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
    # 初始化存储 KL 散度和 BLEU 分数的列表
    kl_original, kl_distilled = [], []
    bleu_original, bleu_distilled = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher.to(device)
    original_student.to(device)
    distilled_student.to(device)

    # 为推理设置模型，注意使用unsloth进行推理前必须设置
    FastLanguageModel.for_inference(teacher)
    FastLanguageModel.for_inference(original_student)
    FastLanguageModel.for_inference(distilled_student)

    # 遍历数据集，按批次处理
    for i in range(0, len(dataset), batch_size):
        # 获取当前批次的数据
        batch = dataset[i:i + batch_size]
        instructions = batch["instruction"]
        inputs = batch["input"]
        true_responses = [extract_response(text) for text in batch["text"]]

        # 批量生成 response
        teacher_responses = generate_response_batch(teacher, tokenizer, instructions, inputs)
        original_responses = generate_response_batch(original_student, tokenizer, instructions, inputs)
        distilled_responses = generate_response_batch(distilled_student, tokenizer, instructions, inputs)

        # 批量计算 logits
        teacher_inputs = tokenizer(teacher_responses, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(device)
        original_inputs = tokenizer(original_responses, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(device)
        distilled_inputs = tokenizer(distilled_responses, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(device)

        with torch.no_grad():
            teacher_logits = teacher(**teacher_inputs).logits  # (batch_size, seq_len_teacher, vocab_size)
            original_logits = original_student(**original_inputs).logits  # (batch_size, seq_len_original, vocab_size)
            distilled_logits = distilled_student(**distilled_inputs).logits  # (batch_size, seq_len_distilled, vocab_size)

            # 批量计算 KL 散度
            for j in range(len(instructions)):
                # 计算原始学生模型与教师模型之间的 KL 散度
                kl_orig = compute_fkl(
                    original_logits[j:j+1],  # 取单条的 logits，保持批次维度
                    teacher_logits[j:j+1],
                    original_inputs["input_ids"][j:j+1],
                    padding_id=-100,
                    temp=2.0
                )
                kl_original.append(kl_orig.item())

                # 计算蒸馏学生模型与教师模型之间的 KL 散度
                kl_dist = compute_fkl(
                    distilled_logits[j:j+1],
                    teacher_logits[j:j+1],
                    distilled_inputs["input_ids"][j:j+1],
                    padding_id=-100,
                    temp=2.0
                )
                kl_distilled.append(kl_dist.item())

        # 批量计算 BLEU 分数
        for j in range(len(instructions)):
            # 将真实响应拆分为单词列表
            ref_tokens = true_responses[j].split()
            # 将原始学生模型的响应拆分为单词列表
            orig_pred_tokens = original_responses[j].split()
            # 将蒸馏学生模型的响应拆分为单词列表
            dist_pred_tokens = distilled_responses[j].split()

            # 使用平滑函数计算 BLEU 分数
            smoothie = SmoothingFunction().method1
            # 计算原始学生模型的 BLEU 分数
            bleu_original.append(sentence_bleu([ref_tokens], orig_pred_tokens, smoothing_function=smoothie))
            # 计算蒸馏学生模型的 BLEU 分数
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

# # 生成 response 的函数
# def generate_response(model, tokenizer, instruction, input_text, max_new_tokens=512):
#     prompt = alpaca_prompt.format(instruction, input_text, "")
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length).to(model.device)
    
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return extract_response(generated_text)

# # 评估函数（只针对 response）
# def evaluate_response_only(teacher, original_student, distilled_student, dataset, tokenizer, batch_size=4):
#     kl_original, kl_distilled = [], []
#     bleu_original, bleu_distilled = [], []  # 修复：初始化两个空列表
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     teacher.to(device)
#     original_student.to(device)
#     distilled_student.to(device)

#     # 提前优化模型为推理模式
    # FastLanguageModel.for_inference(teacher)
    # FastLanguageModel.for_inference(original_student)
    # FastLanguageModel.for_inference(distilled_student)
    
#     for i in range(0, len(dataset), batch_size):
#         batch = dataset[i:i + batch_size]
#         instructions = batch["instruction"]
#         inputs = batch["input"]
#         true_responses = [extract_response(text) for text in batch["text"]]

#         for j in range(len(instructions)):
#             # 真实 response
#             true_response = true_responses[j]

#             # 教师模型生成 response
#             teacher_response = generate_response(teacher, tokenizer, instructions[j], inputs[j])
#             teacher_inputs = tokenizer(teacher_response, return_tensors="pt").to(device)
#             with torch.no_grad():
#                 teacher_logits = teacher(**teacher_inputs).logits

#             # 原始小模型生成 response
#             original_response = generate_response(original_student, tokenizer, instructions[j], inputs[j])
#             original_inputs = tokenizer(original_response, return_tensors="pt").to(device)
#             with torch.no_grad():
#                 original_logits = original_student(**original_inputs).logits
#                 # 处理维度不匹配
#                 # kl_orig = 0
#                 # if isinstance(original_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
#                 #     if original_logits.shape[-1] != teacher_logits.shape[-1]:
#                 #         teacher_logits_orig = teacher_logits[:, :, :original_logits.shape[-1]]
#                 #         labels = original_inputs["input_ids"]
#                 #         kl_orig = compute_fkl(original_logits, teacher_logits_orig, labels, padding_id=-100, temp=2.0)
#                 #     else:
#                 #         labels = original_inputs["input_ids"]
#                 #         kl_orig = compute_fkl(original_logits, teacher_logits, labels, padding_id=-100, temp=2.0)
#                 # kl_original.append(kl_orig.item())
            
#                 # 直接调用 compute_fkl，不重复处理维度
#                 kl_orig = compute_fkl(original_logits, teacher_logits, original_inputs["input_ids"], padding_id=-100, temp=2.0)
#                 kl_original.append(kl_orig.item())

#             # 蒸馏小模型生成 response
#             distilled_response = generate_response(distilled_student, tokenizer, instructions[j], inputs[j])
#             distilled_inputs = tokenizer(distilled_response, return_tensors="pt").to(device)
#             with torch.no_grad():
#                 distilled_logits = distilled_student(**distilled_inputs).logits
#                 # # 处理维度不匹配
#                 # kl_dist = 0
#                 # if isinstance(distilled_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
#                 #     if distilled_logits.shape[-1] != teacher_logits.shape[-1]:
#                 #         teacher_logits_dist = teacher_logits[:, :, :distilled_logits.shape[-1]]
#                 #         labels = distilled_inputs["input_ids"]
#                 #         kl_dist = compute_fkl(distilled_logits, teacher_logits_dist, labels, padding_id=-100, temp=2.0)
#                 #     else:
#                 #         labels = distilled_inputs["input_ids"]
#                 #         kl_dist = compute_fkl(distilled_logits, teacher_logits, labels, padding_id=-100, temp=2.0)
#                 # kl_distilled.append(kl_dist.item())
            
#                 # 直接调用 compute_fkl，不重复处理维度
#                 kl_dist = compute_fkl(distilled_logits, teacher_logits, distilled_inputs["input_ids"], padding_id=-100, temp=2.0)
#                 kl_distilled.append(kl_dist.item())

#             # BLEU 分数计算
#             ref_tokens = true_response.split()
#             orig_pred_tokens = original_response.split()
#             dist_pred_tokens = distilled_response.split()

#             smoothie = SmoothingFunction().method1
#             bleu_original.append(sentence_bleu([ref_tokens], orig_pred_tokens, smoothing_function=smoothie))
#             bleu_distilled.append(sentence_bleu([ref_tokens], dist_pred_tokens, smoothing_function=smoothie))

#     # 计算平均值
#     avg_kl_original = np.mean(kl_original)
#     avg_kl_distilled = np.mean(kl_distilled)
#     avg_bleu_original = np.mean(bleu_original)
#     avg_bleu_distilled = np.mean(bleu_distilled)

#     return {
#         "Original KL": avg_kl_original,
#         "Distilled KL": avg_kl_distilled,
#         "Original BLEU": avg_bleu_original,
#         "Distilled BLEU": avg_bleu_distilled
#     }

# 执行评估
results = evaluate_response_only(teacher, original_student, distilled_student, val_dataset, tokenizer)
print("Evaluation Results (Response Only):")
print(f"Original Model KL Divergence: {results['Original KL']:.4f}")
print(f"Distilled Model KL Divergence: {results['Distilled KL']:.4f}")
print(f"Original Model BLEU Score: {results['Original BLEU']:.4f}")
print(f"Distilled Model BLEU Score: {results['Distilled BLEU']:.4f}")

'''  
维度：  
教师模型：unsloth/Qwen2.5-7B，词汇表大小约为 152064（根据之前的错误信息推测）。  
学生模型：unsloth/Qwen2.5-1.5B 和蒸馏模型，词汇表大小约为 151936。  
max_seq_length = 2048，max_new_tokens = 512，batch_size = 4。  
任务：对 val_dataset（1000 条数据）生成 response，计算 KL 散度和 BLEU 分数。  

teacher_inputs：  
来源：teacher_inputs = tokenizer(teacher_response, return_tensors="pt").to(device)  
维度：  
teacher_inputs["input_ids"]：(1, seq_len_teacher)  
teacher_inputs["attention_mask"]：(1, seq_len_teacher)  
含义：  
"input_ids"：一个批次大小为 1 的张量，序列长度 seq_len_teacher 取决于 teacher_response 的 token 数量（通常小于 max_seq_length=2048）。  
"attention_mask"：与 "input_ids" 形状相同，表示哪些 token 是有效输入（1）或填充  

teacher_logits：  
来源：teacher_logits = teacher(**teacher_inputs).logits  
维度：(1, seq_len_teacher, vocab_size_teacher)  
含义：  
第 0 维：批次大小，固定为 1（单条生成）。  
第 1 维：序列长度 seq_len_teacher，与 teacher_inputs["input_ids"] 的长度一致。  
第 2 维：词汇表大小 vocab_size_teacher，约为 152064（Qwen2.5-7B 的词汇表大小）。  


KL 散度计算相关变量  
以下变量在 compute_fkl 中生成，基于传入的 logits, teacher_logits, 和 target：  

logits：  
来源：original_logits 或 distilled_logits 传入。  
维度：  
original_logits：(1, seq_len_original, vocab_size_student)  
distilled_logits：(1, seq_len_distilled, vocab_size_student)  
含义：学生模型的 logits，未经调整。  
teacher_logits（调整前）：  
来源：teacher(**teacher_inputs).logits 传入。  
维度：(1, seq_len_teacher, vocab_size_teacher)  
含义：教师模型的 logits，未经调整。  
teacher_logits（调整后）：  
来源：在 compute_fkl 中若 logits.shape[-1] != teacher_logits.shape[-1]，则截断。  
维度：  
若调整：(1, seq_len_teacher, vocab_size_student)，vocab_size_student = 151936。  
若未调整：(1, seq_len_teacher, vocab_size_teacher)，vocab_size_teacher = 152064。  
含义：截断后与学生模型的词汇表大小对齐。  
target：  
来源：original_inputs["input_ids"] 或 distilled_inputs["input_ids"] 传入。  
维度：  
original_inputs["input_ids"]：(1, seq_len_original)  
distilled_inputs["input_ids"]：(1, seq_len_distilled)  
含义：学生模型生成的 response 的 token ID，作为 KL 散度的掩码依据。  
log_probs：  
来源：log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)  
维度：  
original：(1, seq_len_original, vocab_size_student)  
distilled：(1, seq_len_distilled, vocab_size_student)  
含义：学生模型 logits 的对数 softmax 概率分布。  
teacher_probs：  
来源：teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)  
维度：  
若调整：(1, seq_len_teacher, vocab_size_student)  
若未调整：(1, seq_len_teacher, vocab_size_teacher)  
含义：教师模型 logits 的 softmax 概率分布。  
teacher_log_probs：  
来源：teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)  
维度：同 teacher_probs。  
含义：教师模型 logits 的对数 softmax 概率分布。  
kl（中间结果）：  
来源：kl = (teacher_probs * (teacher_log_probs - log_probs))  
维度：  
original：(1, seq_len_original, vocab_size_student) → 求和后 (1, seq_len_original)distilled：(1, seq_len_distilled, vocab_size_student) → 求和后 (1, seq_len_distilled)含义：逐 token 的 KL 散度，沿词汇表维度求和后得到每个位置的散度值。  
pad_mask：  
来源：pad_mask = target.eq(padding_id)  
维度：  
original：(1, seq_len_original)  
distilled：(1, seq_len_distilled)  
含义：布尔张量，标记 target 中哪些位置是填充 token（-100）。  
kl（最终结果）：  
来源：kl = kl.masked_fill_(pad_mask, 0.0).sum()  
维度：标量（()）  
含义：对整个序列的 KL 散度总和，填充位置被置为 0。  
'''