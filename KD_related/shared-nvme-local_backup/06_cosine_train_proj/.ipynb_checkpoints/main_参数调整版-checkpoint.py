import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()

origin_student_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-1.5B"
teacher_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-7B"


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

from loss import compute_ot_loss_improved, ProjectionMatrix

# 配置参数
max_seq_length = 2048
dtype = None
load_in_4bit = True

epoch = 10
temperature = 2.0
reduction = "sum"
topk = 100
alpha = 0.5
chunk_size = 4

# 自定义知识蒸馏训练器
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

        # **可训练投影矩阵**
        teacher_dim = teacher_model.get_input_embeddings().weight.shape[-1]
        student_dim = self.model.get_input_embeddings().weight.shape[-1]
        self.proj_layer = ProjectionMatrix(teacher_dim, student_dim).to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """ 计算知识蒸馏的损失 """
        outputs_student = model(**inputs)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_logits = outputs_student.logits
        teacher_logits = teacher_outputs.logits
        loss_ce = outputs_student.loss

        student_embeddings = self.model.get_input_embeddings().weight
        teacher_embeddings = self.teacher_model.get_input_embeddings().weight

        ot_loss = 0
        if isinstance(student_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            labels = inputs.get('labels', None)

            ot_loss = compute_ot_loss_improved(
                student_logits, teacher_logits,
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_embeddings,
                proj_layer=self.proj_layer,  # 传入可训练投影层
                target=labels, padding_id=-100,
                reduction=reduction, temp=temperature,
                topk=topk, chunk_size=chunk_size,
            )

        loss_total = alpha * ot_loss + (1-alpha) * loss_ce if self.if_use_entropy else ot_loss

        if hasattr(wandb, "log"):
            wandb.log({"train_loss": loss_total.item()})

        return (loss_total, outputs_student) if return_outputs else loss_total


# **初始化学生模型**
student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

student = FastLanguageModel.get_peft_model(
    student, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, use_rslora=False, loftq_config=None,
)

student.print_trainable_parameters()

# **初始化教师模型**
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



# 训练配置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()

origin_student_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-1.5B"
teacher_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-7B"


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

from loss import compute_ot_loss_improved, ProjectionMatrix

# 配置参数
max_seq_length = 2048
dtype = None
load_in_4bit = True

epoch = 10
temperature = 2.0
reduction = "sum"
topk = 150
alpha = 0.5
chunk_size = 4

# 自定义知识蒸馏训练器
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

        # **可训练投影矩阵**
        teacher_dim = teacher_model.get_input_embeddings().weight.shape[-1]
        student_dim = self.model.get_input_embeddings().weight.shape[-1]
        self.proj_layer = ProjectionMatrix(teacher_dim, student_dim).to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """ 计算知识蒸馏的损失 """
        outputs_student = model(**inputs)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_logits = outputs_student.logits
        teacher_logits = teacher_outputs.logits
        loss_ce = outputs_student.loss

        student_embeddings = self.model.get_input_embeddings().weight
        teacher_embeddings = self.teacher_model.get_input_embeddings().weight

        ot_loss = 0
        if isinstance(student_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            labels = inputs.get('labels', None)

            ot_loss = compute_ot_loss_improved(
                student_logits, teacher_logits,
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_embeddings,
                proj_layer=self.proj_layer,  # 传入可训练投影层
                target=labels, padding_id=-100,
                reduction=reduction, temp=temperature,
                topk=topk, chunk_size=chunk_size,
            )

        loss_total = alpha * ot_loss + (1-alpha) * loss_ce if self.if_use_entropy else ot_loss

        if hasattr(wandb, "log"):
            wandb.log({"train_loss": loss_total.item()})

        return (loss_total, outputs_student) if return_outputs else loss_total


# **初始化学生模型**
student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

student = FastLanguageModel.get_peft_model(
    student, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, use_rslora=False, loftq_config=None,
)

student.print_trainable_parameters()

# **初始化教师模型**
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



import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()

origin_student_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-1.5B"
teacher_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-7B"


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

from loss import compute_ot_loss_improved, ProjectionMatrix

# 配置参数
max_seq_length = 2048
dtype = None
load_in_4bit = True

epoch = 10
temperature = 2.0
reduction = "sum"
topk = 150
alpha = 0.5
chunk_size = 4

# 自定义知识蒸馏训练器
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy

        # **可训练投影矩阵**
        teacher_dim = teacher_model.get_input_embeddings().weight.shape[-1]
        student_dim = self.model.get_input_embeddings().weight.shape[-1]
        self.proj_layer = ProjectionMatrix(teacher_dim, student_dim).to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """ 计算知识蒸馏的损失 """
        outputs_student = model(**inputs)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_logits = outputs_student.logits
        teacher_logits = teacher_outputs.logits
        loss_ce = outputs_student.loss

        student_embeddings = self.model.get_input_embeddings().weight
        teacher_embeddings = self.teacher_model.get_input_embeddings().weight

        ot_loss = 0
        if isinstance(student_logits, torch.Tensor) and isinstance(teacher_logits, torch.Tensor):
            labels = inputs.get('labels', None)

            ot_loss = compute_ot_loss_improved(
                student_logits, teacher_logits,
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_embeddings,
                proj_layer=self.proj_layer,  # 传入可训练投影层
                target=labels, padding_id=-100,
                reduction=reduction, temp=temperature,
                topk=topk, chunk_size=chunk_size,
            )

        loss_total = alpha * ot_loss + (1-alpha) * loss_ce if self.if_use_entropy else ot_loss

        if hasattr(wandb, "log"):
            wandb.log({"train_loss": loss_total.item()})

        return (loss_total, outputs_student) if return_outputs else loss_total


# **初始化学生模型**
student, _ = FastLanguageModel.from_pretrained(
    model_name=origin_student_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

student = FastLanguageModel.get_peft_model(
    student, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407, use_rslora=False, loftq_config=None,
)

student.print_trainable_parameters()

# **初始化教师模型**
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



# 原配置：固定学习率（可能导致后期震荡）
# 修改后：余弦退火 + 权重衰减 + 更小梯度累积步数
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epoch,
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,  
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=1,
    report_to=["wandb"],
    bf16=True,
    learning_rate=0.0005,
    lr_scheduler_type='cosine',  # 从constant改为余弦退火，缓解后期震荡
    weight_decay=0.01,  # 新增L2正则化，抑制过拟合
    optim="adamw_torch_fused",
    warmup_ratio=0.1,  # 添加10%训练步数的预热，稳定初始训练
)

trainer = KDTrainer(
    model=student, teacher_model=teacher,
    
    if_use_entropy=True, processing_class=tokenizer,
    
    train_dataset=dataset, dataset_text_field="text",
    max_seq_length=max_seq_length, dataset_num_proc=2,
    packing=False, args=args,
)

trainer.train(resume_from_checkpoint=False)


'''
投影矩阵变为可训练参数：

用 torch.nn.Parameter 代替原始固定投影矩阵。
在 KDTrainer 中定义 ProjectionMatrix，并传入 compute_ot_loss_improved。
代价矩阵更新：

默认 使用余弦相似度（1 - cosine_similarity）。
Wasserstein-1 距离（torch.cdist）保留为注释。
这样，训练过程中 proj_matrix 会自动优化，同时 OT 计算基于余弦相似度，实现更加高效的对齐！
'''