import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
from datasets import load_dataset
from emd_loss import EMDLossWithProjection  # 导入新的 EMD_diff_probability 损失类

origin_student_path = "/root/shared-nvme/model/Qwen2.5-1.5B-bnb-4bit"
teacher_path = "/root/shared-nvme/model/Qwen2.5-7B"
# 配置参数
max_seq_length = 2048  # 最大序列长度，支持RoPE扩展
dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = True  # 使用4bit量化减少内存占用



# 自定义 KDTrainer 类
class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=None, temperature=1.0, reduction="mean", block_size=64,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self.temperature = temperature
        self.reduction = reduction
        self.block_size = block_size
        self.padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # 初始化 EMD_diff_probability 损失函数，传入学生和教师的词表大小
        student_vocab_size = self.model.config.vocab_size
        teacher_vocab_size = self.teacher_model.config.vocab_size
        self.emd_loss_fn = EMDLossWithProjection(student_vocab_size, teacher_vocab_size).to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 学生模型前向传播
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits  # [batch_size, seq_length, vocab_size]

        ce_loss = outputs_student.loss  # 默认的语言建模损失

        # 获取目标序列
        target_ids = inputs["labels"]  # [batch_size, seq_length]

        # 教师模型推理（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits  # [batch_size, seq_length, vocab_size]

        # 计算 EMD_diff_probability 损失（批次维度逐个处理）
        batch_size = student_logits.shape[0]
        emd_loss_total = 0.0
        for b in range(batch_size):
            student_logits_b = student_logits[b]  # [seq_length, vocab_size]
            teacher_logits_b = teacher_logits[b]  # [seq_length, vocab_size]
            emd_loss_b = self.emd_loss_fn(
                student_logits_b,
                teacher_logits_b,
                temperature=self.temperature,
                reduction=self.reduction,
                padding_id=self.padding_id,
                block_size=self.block_size
            )
            emd_loss_total += emd_loss_b

        # EMD_diff_probability 损失平均或求和
        if self.reduction == "mean":
            emd_loss = emd_loss_total / batch_size
        else:  # "sum"
            emd_loss = emd_loss_total

        # 组合损失
        loss_total = emd_loss
        if self.if_use_entropy:
            loss_total = loss_total + ce_loss

        return (loss_total, outputs_student) if return_outputs else loss_total


# 加载学生模型
student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 应用 LoRA 适配器
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

# 定义 Alpaca 格式的 prompt 模板
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


# 加载并预处理 Alpaca 数据集
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 配置训练参数
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    logging_steps=50,
    save_strategy='epoch',
    save_total_limit=2,
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
    temperature=2.0,
    reduction="sum",
    block_size=64,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

# 开始训练
trainer.train(resume_from_checkpoint=False)