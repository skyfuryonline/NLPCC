import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb

os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()

origin_student_path = "/root/shared-nvme/model/Qwen2.5-1.5B"
teacher_path = "/root/shared-nvme/model/Qwen2.5-7B"

# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel

# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入Hugging Face的Trainer和训练参数配置
from transformers import Trainer, TrainingArguments
# 导入trl库的SFTTrainer（监督式微调训练器）
from trl import SFTTrainer, SFTConfig

# 导入自定义的损失函数模块
from loss import  compute_ot_loss_improved

# 导入Hugging Face的dataset模块
from datasets import load_dataset

# 配置参数
max_seq_length = 2048  # 最大序列长度，支持RoPE扩展
dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = True  # 使用4bit量化减少内存占用


# 自定义知识蒸馏训练器（继承自SFTTrainer）
class KDTrainer(SFTTrainer):

    def __init__(self, *args, teacher_model=None, if_use_entropy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model  # 教师模型
        self.if_use_entropy = if_use_entropy  # 是否使用交叉熵损失

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """ 计算知识蒸馏的损失 """
        # 前向传播获取 student 输出
        outputs_student = model(**inputs)

        # 获取 teacher 输出（无梯度计算）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        # 获取 logits
        student_logits = outputs_student.logits  # (batch_size, seq_len, vocab_size_s)
        teacher_logits = teacher_outputs.logits  # (batch_size, seq_len, vocab_size_t)

        # 获取交叉熵损失
        loss_ce = outputs_student.loss  # 假设 model 返回了 loss

        # 获取词嵌入
        student_embeddings = model.get_input_embeddings().weight  # (vocab_size_s, student_embed_dim), e.g., (151665, 1536)
        teacher_embeddings = self.teacher_model.get_input_embeddings().weight  # (vocab_size_t, teacher_embed_dim), e.g., (151665, 3584)

        # 计算 OT 蒸馏损失
        ot_loss = 0.0  # 默认值
        if isinstance(student_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            labels = inputs.get('labels', None)  # 获取标签，可能为 None
            ot_loss = compute_ot_loss_improved(
                student_logits,
                teacher_logits,
                student_embeddings,  # 直接传入
                teacher_embeddings,  # 直接传入
                target=labels,
                padding_id=-100,
                reduction="sum",
                temp=2.0
            )
        # 组合损失（示例）
        loss_total = 0.5*loss_ce + 0.5*ot_loss  # 根据需求调整权重

        # 记录损失
        if hasattr(wandb, "log"):
            wandb.log({"train_loss": loss_total.item()})

        return (loss_total, outputs_student) if return_outputs else loss_total


# 初始化学生模型（使用unsloth的优化实现）
student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,  # 1.5B参数的千问模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # 4bit量化加载
)

# 应用LoRA适配器（参数高效微调）
student = FastLanguageModel.get_peft_model(
    student,
    r=16,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  # 目标注意力模块
                    "gate_proj", "up_proj", "down_proj", ],  # FFN模块
    lora_alpha=16,  # LoRA alpha参数
    lora_dropout=0,  # 无dropout
    bias="none",  # 不训练偏置参数
    use_gradient_checkpointing="unsloth",  # 使用优化版梯度检查点
    random_state=3407,  # 随机种子
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

student.print_trainable_parameters()  # 打印可训练参数量

# 初始化教师模型（更大的7B模型）
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name=teacher_path,  # 7B参数的教师模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
teacher.eval()  # 固定教师模型参数

# 定义Alpaca格式的prompt模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 获取结束符


# 数据集格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts, }


pass

# 加载并预处理Alpaca数据集
from datasets import load_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched=True, )

val_dataset = load_dataset("yahma/alpaca-cleaned", split="train[2000:3000]")
val_dataset = val_dataset.map(formatting_prompts_func, batched=True, )

# 配置训练参数
args = TrainingArguments(
    output_dir='./results',  # 输出目录
    num_train_epochs=10,  # 训练轮次

    do_train=True,  # 启用训练模式

    # 为什么小train batch会导致更好的效果？
    per_device_train_batch_size=4,  # 单设备批次大小
    gradient_accumulation_steps=16,  # 梯度累积步数

    logging_steps=50,  # 日志记录间隔

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
    model=student,  # 学生模型
    teacher_model=teacher,  # 教师模型

    if_use_entropy=True,  # 启用混合损失
    processing_class=tokenizer,  # 使用教师模型的tokenizer

    train_dataset=dataset,  # 训练数据集

    dataset_text_field="text",  # 文本字段名
    max_seq_length=max_seq_length,  # 最大序列长度
    dataset_num_proc=2,  # 数据集处理进程数
    packing=False,  # 禁用序列打包（短序列时可加速）
    args=args,  # 训练参数配置
)

# 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
# 需要当前的epoch比上次的大
trainer.train(resume_from_checkpoint=False)