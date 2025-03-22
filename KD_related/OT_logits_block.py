# import os
#
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import wandb
#
# os.environ["WANDB_PROJECT"] = "KD"
# os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
# wandb.init()

origin_student_path = "/root/shared-nvme/models/Qwen2.5-1.5B-bnb-4bit"
teacher_path = "/root/shared-nvme/models/Qwen2.5-7B"

# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel

# 导入trl库的SFTTrainer（监督式微调训练器）
from trl import SFTTrainer, SFTConfig

# 导入PyTorch相关模块
import torch
import torch.nn.functional as F

# 导入Hugging Face的Trainer和训练参数配置
from transformers import TrainingArguments
# 导入Hugging Face的dataset模块
from datasets import load_dataset

# 配置参数
max_seq_length = 1024  # 最大序列长度，支持RoPE扩展
dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = True  # 使用4bit量化减少内存占用

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义 KDTrainer 类
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=None, temperature=1.0, reduction="mean", block_size=64,
                 lambda_reg=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model  # 教师模型
        self.if_use_entropy = if_use_entropy  # 是否使用交叉熵
        self.temperature = temperature  # softmax 温度参数
        self.reduction = reduction  # OT 损失聚合方式
        self.block_size = block_size  # 分块大小
        self.lambda_reg = lambda_reg  # 熵正则化超参数
        self.padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0  # 默认填充 ID

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 学生模型前向传播
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits  # [batch_size, seq_length, vocab_size]

        ce_loss = outputs_student.loss  # 默认的语言建模损失

        # 获取目标序列（通常是移位后的 input_ids）
        target_ids = inputs["labels"]  # [batch_size, seq_length]

        # 教师模型推理（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits  # [batch_size, seq_length, vocab_size]

        # 计算 OT 损失（批次维度逐个处理）
        batch_size = student_logits.shape[0]
        ot_loss_total = 0.0
        for b in range(batch_size):
            student_logits_b = student_logits[b]  # [seq_length, vocab_size]
            teacher_logits_b = teacher_logits[b]  # [seq_length, vocab_size]
            target_ids_b = target_ids[b]  # [seq_length]
            ot_loss_b = self.compute_ot_loss(student_logits_b, teacher_logits_b, target_ids_b)
            ot_loss_total += ot_loss_b

        # OT 损失平均或求和
        if self.reduction == "mean":
            ot_loss = ot_loss_total / batch_size
        else:  # "sum"
            ot_loss = ot_loss_total

        # 组合损失
        loss_total = ot_loss
        if self.if_use_entropy:
            loss_total = loss_total + ce_loss

        return (loss_total, outputs_student) if return_outputs else loss_total

    def compute_ot_loss(self, student_logits, teacher_logits, target, temperature=None, reduction=None,
                        padding_id=None):
        """计算分块 OT 损失"""
        temperature = temperature if temperature is not None else self.temperature
        reduction = reduction if reduction is not None else self.reduction
        padding_id = padding_id if padding_id is not None else self.padding_id

        N, V = student_logits.shape  # [seq_length, vocab_size]
        num_blocks = N // self.block_size
        ot_loss = 0.0
        valid_blocks = 0

        # 创建掩码以屏蔽 padding
        mask = (target != padding_id).float()  # [seq_length]

        # 分块计算
        for i in range(num_blocks):
            for j in range(num_blocks):
                start_i, end_i = i * self.block_size, (i + 1) * self.block_size
                start_j, end_j = j * self.block_size, (j + 1) * self.block_size
                student_block = student_logits[start_i:end_i]  # [block_size, vocab_size]
                teacher_block = teacher_logits[start_j:end_j]  # [block_size, vocab_size]
                block_mask = mask[start_i:end_i] * mask[start_j:end_j]

                if block_mask.sum() == 0:  # 全为 padding 跳过
                    continue

                # 计算相似性矩阵
                S = torch.matmul(student_block, teacher_block.T) / np.sqrt(V)  # [block_size, block_size]
                S = S / temperature
                S_norm = F.softmax(S, dim=-1)
                C = 1 - S_norm  # 成本矩阵

                # 使用 OTT 求解 OT
                geom = pointcloud.PointCloud(student_block, teacher_block, cost_fn=None, scale_cost="mean")
                prob = linear_problem.LinearProblem(geom)
                solver = sinkhorn.Sinkhorn()
                ot_result = solver(prob, epsilon=self.lambda_reg)
                T_star = ot_result.matrix  # 传输计划

                # 计算 OT 损失贡献
                block_loss = torch.sum(T_star * C) * block_mask.mean()
                ot_loss += block_loss
                valid_blocks += 1

        # 应用 reduction
        if valid_blocks == 0:
            return torch.tensor(0.0, device=device)
        if reduction == "mean":
            ot_loss = ot_loss / valid_blocks
        return ot_loss


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
    temperature=1.0,
    reduction="mean",
    block_size=64,
    lambda_reg=0.1,

    processing_class=tokenizer,
    train_dataset=dataset,

    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

# 开始训练
trainer.train(resume_from_checkpoint=False)