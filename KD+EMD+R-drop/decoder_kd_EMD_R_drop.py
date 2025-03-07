# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# import wandb
# os.environ["WANDB_PROJECT"] = "KD"
# os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
# wandb.init()

# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel
from EMD_with_pytorch import compute_wasserstein_loss
# 导入PyTorch相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入Hugging Face的Trainer和训练参数配置
from transformers import Trainer, TrainingArguments
# 导入trl库的SFTTrainer（监督式微调训练器）
from trl import SFTTrainer, SFTConfig

# 导入自定义的损失函数模块
from losses import compute_fkl
# 导入Hugging Face的dataset模块
from datasets import load_dataset

# 配置参数
max_seq_length = 2048  # 最大序列长度，支持RoPE扩展
dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = True  # 使用4bit量化减少内存占用

origin_student_path =""
teacher_path = ""


# 自定义知识蒸馏训练器（继承自SFTTrainer）
class KDWithR_DropTrainer(SFTTrainer):
    # rdrop_alpha是关键超参数，建议尝试范围（如0.5到5.0），以平衡R - Drop正则化与KD损失的效果
    def __init__(self, *args, teacher_model=None, if_use_entropy=None,wasserstein_version=1, rdrop_alpha=1.0,**kwargs):
        '''
         **kwargs：表示“关键字参数”（keyword arguments）:
        其作用是收集所有未在函数参数列表中显式列出的以 key=value 形式传入的参数，并将它们以字典的形式存储。

        *args：用于收集额外的位置参数（以元组形式传入）

        当你初始化 Trainer 时，你会传入 model=student_model，这个模型会赋值给 Trainer 实例的 self.model。

        新增参数：
        - rdrop_alpha: R-Drop一致性损失的权重系数
        '''
        super().__init__(*args, **kwargs)
        self.wasserstein_version = wasserstein_version
        self.teacher_model = teacher_model  # 教师模型
        self.if_use_entropy = if_use_entropy  # 是否使用交叉熵的标记
        self.rdrop_alpha = rdrop_alpha  # R-Drop损失的权重

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        #在训练或评估时，Trainer 内部会调用 self.model 来进行前向传播（例如 compute_loss 中传入的 model 参数就是 self.model）
        '''
        修改后的compute_loss函数，加入R-Drop损失：
        1. 学生模型进行两次Dropout前向传播，生成两个输出logits。
        2. 计算R-Drop的KL散度损失。
        3. 组合Wasserstein损失、交叉熵损失（可选）和R-Drop损失。
        '''
        # 确保模型处于训练模式，以便Dropout生效
        model.train()

        # 第一次学生模型前向传播
        outputs_student_1 = model(**inputs)
        logits_1 = outputs_student_1.logits

        # 第二次学生模型前向传播（由于Dropout随机性，输出会有差异）
        outputs_student_2 = model(**inputs)
        logits_2 = outputs_student_2.logits

        # 教师模型推理（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # 获取学生模型的基础损失（交叉熵）
        loss_ce = outputs_student_1.loss  # 假设loss是基于labels计算的交叉熵

        # 计算Wasserstein损失
        wasserstein_loss = 0
        if isinstance(logits_1, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            labels = inputs['labels']
            wasserstein_loss = compute_wasserstein_loss(
                logits=logits_1,  # 使用第一次前向传播的logits
                teacher_logits=teacher_logits,
                target=labels,
                padding_id=-100,
                reduction="sum",
                temp=2.0,
                wasserstein_version=self.wasserstein_version
            )

        # 计算R-Drop损失（两次学生模型输出的KL散度）
        rdrop_loss = 0
        if isinstance(logits_1, torch.Tensor) and isinstance(logits_2, torch.Tensor):
            # 将logits转为概率分布
            probs_1 = F.softmax(logits_1, dim=-1)
            probs_2 = F.softmax(logits_2, dim=-1)
            # 计算双向KL散度（对称形式）
            kl_loss_1 = F.kl_div(
                F.log_softmax(logits_1, dim=-1),
                probs_2,
                reduction='batchmean'
            )
            kl_loss_2 = F.kl_div(
                F.log_softmax(logits_2, dim=-1),
                probs_1,
                reduction='batchmean'
            )
            rdrop_loss = 0.5 * (kl_loss_1 + kl_loss_2)

        # 组合最终损失
        if self.if_use_entropy:
            # 混合损失：交叉熵 + Wasserstein + R-Drop
            loss_total = 0.5 * loss_ce + 0.5 * wasserstein_loss + self.rdrop_alpha * rdrop_loss
        else:
            # 不使用交叉熵，仅用Wasserstein + R-Drop
            loss_total = wasserstein_loss + self.rdrop_alpha * rdrop_loss

        return (loss_total, outputs_student_1) if return_outputs else loss_total


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
    lora_dropout=0,  # 无dropout（注意：R-Drop依赖模型内部Dropout，这里仅关闭LoRA的额外dropout）
    bias="none",  # 不训练偏置参数
    use_gradient_checkpointing="unsloth",  # 使用优化版梯度检查点
    random_state=3407,  # 随机种子
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
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
'''
Dropout 层：在训练模式（model.train()）下，Dropout 会随机丢弃神经元以增强泛化性；
但在评估模式下，Dropout 完全失效，所有神经元都会参与计算。

BatchNorm 层：在训练时，BatchNorm 会计算当前 batch 的均值和方差；
但在评估模式下，它会使用训练阶段累积的全局均值和方差，而不是当前 batch 的统计量。

教师模型的输出需要是确定且稳定的（基于预训练知识），
如果保留 Dropout 或动态 BatchNorm，会引入随机性，导致知识蒸馏过程不稳定

双重保险：eval() + torch.no_grad() 确保：
- 不更新教师模型的参数。
- 减少内存消耗（不存储中间梯度）。
'''

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
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


# 加载并预处理Alpaca数据集
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched = True,)

val_dataset = load_dataset("yahma/alpaca-cleaned", split = "train[2000:3000]")
val_dataset = val_dataset.map(formatting_prompts_func, batched = True,)

# 配置训练参数
args = TrainingArguments(
    output_dir='./results',  # 输出目录
    num_train_epochs=20,  # 训练轮次

    do_train=True,  # 启用训练模式
    do_eval=True,  # 启用评估模式

    per_device_train_batch_size=4,  # 单设备批次大小
    per_device_eval_batch_size=8,  # 单设备评估批次大小
    gradient_accumulation_steps=16,  # 梯度累积步数

    logging_steps=500,  # 日志记录间隔

    # save_strategy='steps',  # 按step保存模型
    eval_strategy="epoch",  # 或 "steps"，确保评估被触发
    report_to="wandb",

    save_strategy="epoch",  # 每个epoch保存一次模型（可选）
    load_best_model_at_end=True,  # 训练结束时加载最佳模型（需要结合metric_for_best_model）
    metric_for_best_model="eval_loss",  # 用于选择最佳模型的指标
    greater_is_better=False,  # eval_loss越小越好
    save_total_limit=1,  # 最大保存检查点数

    bf16=True,  # 使用bfloat16精度
    learning_rate=0.0005,  # 学习率
    lr_scheduler_type='constant',  # 恒定学习率
    optim="adamw_torch_fused",  # 使用融合AdamW优化器
)

# 初始化知识蒸馏训练器
trainer = KDWithR_DropTrainer(
    model=student,  # 学生模型
    # 手动设置这个参数，注意看trainer中是怎么定义的
    wasserstein_version=1,

    teacher_model=teacher,  # 教师模型

    if_use_entropy=True,  # 启用混合损失
    rdrop_alpha=1.0,  # R-Drop损失权重（可调）
    processing_class=tokenizer,  # 使用教师模型的tokenizer

    train_dataset=dataset,  # 训练数据集
    eval_dataset=val_dataset,  # 验证数据集

    dataset_text_field="text",  # 文本字段名
    max_seq_length=max_seq_length,  # 最大序列长度
    dataset_num_proc=2,  # 数据集处理进程数
    packing=False,  # 禁用序列打包（短序列时可加速）
    args=args,  # 训练参数配置
)

# 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
# 需要当前的epoch比上次的大
trainer.train(resume_from_checkpoint=False)
