import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()

# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel

# 导入PyTorch相关模块
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# 导入Hugging Face的Trainer和训练参数配置
from transformers import  TrainingArguments
# 导入trl库的SFTTrainer（监督式微调训练器）
from trl import SFTTrainer, SFTConfig

# 导入Hugging Face的dataset模块
from datasets import load_dataset
from OT_based_loss import *


# 配置参数
max_seq_length = 2048
dtype = None
load_in_4bit = True

origin_student_path = "/root/shared-nvme/model/Qwen2.5-1.5B-bnb-4bit"
teacher_path = "/root/shared-nvme/model/Qwen2.5-7B"


def compute_wasserstein_loss(
        logits,
        teacher_logits,
        target,
        teacher_emb_reduced,
        student_emb_reduced,
        padding_id=-100,
        reduction="sum",
        temp=2.0,
        wasserstein_version=1
):
    """
    计算 Wasserstein 损失。
    """
    # 创建掩码
    mask = (target != padding_id).float() if target is not None else None

    if wasserstein_version == 1:
        # 使用 EMD_diff_probability (Wasserstein-1) 损失
        return compute_emd_loss(
            teacher_logits=teacher_logits,
            student_logits=logits,
            teacher_emb=teacher_emb_reduced,
            student_emb=student_emb_reduced,
            temperature=temp,
            reduction=reduction,
            mask=mask
        )
    else:
        raise ValueError(f"Unsupported wasserstein_version: {wasserstein_version}")

# 自定义知识蒸馏训练器
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=None, wasserstein_version=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.wasserstein_version = wasserstein_version
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

        # 预处理嵌入
        teacher_emb = self.teacher_model.get_input_embeddings().weight
        student_emb = self.model.get_input_embeddings().weight
        self.teacher_emb_reduced, self.student_emb_reduced = align_embeddings(teacher_emb,student_emb,method="pca"),student_emb

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        loss = outputs_student.loss
        logits = outputs_student.logits
        with torch.no_grad():
            teacher_logits = teacher_outputs.logits

        wasserstein_loss = 0
        if isinstance(logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            labels = inputs.get('labels')
            wasserstein_loss = compute_wasserstein_loss(
                logits=logits,
                teacher_logits=teacher_logits,
                target=labels,
                teacher_emb_reduced=self.teacher_emb_reduced,
                student_emb_reduced=self.student_emb_reduced,
                padding_id=-100,
                reduction="sum",
                temp=2.0,
                wasserstein_version=self.wasserstein_version
            )

        if self.if_use_entropy:
            loss_total = 0.5 * wasserstein_loss + 0.5 * loss
        else:
            loss_total = wasserstein_loss

        wandb.log({"train_loss": loss_total.item()})
        return (loss_total, outputs_student) if return_outputs else loss_total


# 初始化学生模型
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
student.print_trainable_parameters()

# 初始化教师模型
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name=teacher_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
teacher.eval()

# 定义Alpaca格式的prompt模板
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


# 加载并预处理数据集
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

val_dataset = load_dataset("yahma/alpaca-cleaned", split="train[2000:3000]")
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# 配置训练参数
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=["wandb"],
    bf16=True,
    learning_rate=0.0005,
    lr_scheduler_type='constant',
    optim="adamw_torch_fused",
)

# 初始化知识蒸馏训练器
trainer = KDTrainer(
    model=student,
    teacher_model=teacher,
    if_use_entropy=True,
    wasserstein_version=1,

    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

# 开始训练
trainer.train(resume_from_checkpoint=False)