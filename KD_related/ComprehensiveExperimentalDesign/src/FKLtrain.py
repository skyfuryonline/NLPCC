import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()


# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel
# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入Hugging Face的Trainer和训练参数配置
from transformers import TrainingArguments
# 导入trl库的SFTTrainer（监督式微调训练器）
from trl import SFTTrainer, SFTConfig

# 导入自定义的损失函数模块
from OTloss import OT_loss

# # 配置模型路径
# # 请改为完整路径
# origin_student_path = "../models/unsloth/Qwen2.5-1.5B"
# teacher_path = "../models/unsloth/Qwen2.5-7B"
# save_path = "../models/results"
# resume_from_checkpoint = False
from config import origin_student_path,teacher_path,save_path,resume_from_checkpoint
origin_student_path = origin_student_path
teacher_path = teacher_path
save_path = save_path
resume_from_checkpoint = resume_from_checkpoint


# 配置参数
# max_seq_length = 2048  # 最大序列长度，支持RoPE扩展
# dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
# load_in_4bit = True  # 使用4bit量化减少内存占用
from config import max_seq_length,dtype,load_in_4bit
# 配置参数（保持不变）
max_seq_length = max_seq_length
dtype = dtype
load_in_4bit = load_in_4bit

from config import epoch,lr,temperature,reduction,topk,alpha,chunk_size
epoch = epoch
lr = lr
temperature = temperature
reduction = reduction
topk = topk
alpha = alpha
chunk_size = chunk_size


# # 加载并预处理Alpaca数据集
# from dataset import train_dataset
# train_dataset = train_dataset.map(formatting_prompts_func, batched=True, )

# # 加载OpusBooks数据集
# from ConstructDataForOpus import train_opus_dataset

# 加载Summary数据集
from ConstructDataForSummary import train_summary_dataset

class KDTrainer(SFTTrainer):

    def __init__(self, *args, teacher_model=None, if_use_entropy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        loss = outputs_student.loss
        logits = outputs_student.logits

        with torch.no_grad():
            teacher_logits = teacher_outputs.logits

        # 如果教师模型和学生模型输出形状不匹配，对学生模型进行padding或对教师模型进行截断
        # print(logits.shape, teacher_logits.shape)
        # print(type(logits), type(teacher_logits))
        # if logits is None or teacher_logits is None:

        kl = 0
        if isinstance(logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            if logits.shape[-1] != teacher_logits.shape[-1]:
                # gap = teacher_logits.shape[-1] - logits.shape[-1]
                # if gap > 0:
                #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
                #     logits = torch.cat([logits, pad_logits], dim=-1)

                teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

                labels = inputs['labels']
                kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)

        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl

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

# train_dataset = train_opus_dataset.map(formatting_prompts_func,batched=True,)

train_dataset = train_summary_dataset.map(formatting_prompts_func,batched=True,)

# 配置训练参数
args = TrainingArguments(
    output_dir=save_path,  # 输出目录

    num_train_epochs=epoch,  # 训练轮次

    do_train=True,  # 启用训练模式

    # 为什么小train batch会导致更好的效果？
    per_device_train_batch_size=4,  # 单设备批次大小
    gradient_accumulation_steps=16,  # 梯度累积步数

    logging_steps=50,  # 日志记录间隔

    save_strategy="epoch",
    save_total_limit=1,
    report_to=["wandb"],
    bf16=True,
    learning_rate=lr,
    lr_scheduler_type='constant',
    optim="adamw_torch_fused",
)

# 初始化知识蒸馏训练器
trainer = KDTrainer(
    model=student,  # 学生模型
    teacher_model=teacher,  # 教师模型

    if_use_entropy=True,  # 启用混合损失
    processing_class=tokenizer,  # 使用教师模型的tokenizer

    train_dataset=train_dataset,  # 训练数据集

    dataset_text_field="text",  # 文本字段名
    max_seq_length=max_seq_length,  # 最大序列长度
    dataset_num_proc=4,  # 数据集处理进程数
    packing=False,  # 禁用序列打包（短序列时可加速）
    args=args,  # 训练参数配置
)

# 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
# 需要当前的epoch比上次的大
trainer.train(resume_from_checkpoint=resume_from_checkpoint)