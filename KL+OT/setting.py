import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
import wandb
from torch.cuda.amp import GradScaler
from trl import SFTTrainer
from bitsandbytes.optim import AdamW8bit

# 初始化学生模型
origin_student_path = "your_student_model_path"  # 替换为实际路径
teacher_path = "your_teacher_model_path"  # 替换为实际路径
max_seq_length = 2048  # 根据需要调整
dtype = torch.bfloat16
load_in_4bit = True

student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

student = FastLanguageModel.get_peft_model(
    student,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 初始化教师模型
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name=trainer_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
teacher.eval()

# Alpaca prompt 模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

# 数据集格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# 加载数据集
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

val_dataset = load_dataset("yahma/alpaca-cleaned", split="train[2000:3000]")
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# 训练参数
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    do_train=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=["wandb"],
    bf16=True,
    learning_rate=0.0005,
    lr_scheduler_type='constant',
    optim="adamw_8bit",
)