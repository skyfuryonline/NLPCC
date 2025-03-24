'''
主要用于在各种数据集上微调教师模型，保存chpt文件--这个chpt放在teacher模型处;
并在外面用FKL/OT进行蒸馏；
最后验证效果(微调后的教师模型(origin teacher)和2种方法蒸馏的学生模型(FKL_student,OT_student)；
'''

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "FT_alpaca"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()


# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel
# 导入Hugging Face的Trainer和训练参数配置
from transformers import TrainingArguments
# 导入trl库的SFTTrainer（监督式微调训练器）
from trl import SFTTrainer


# # 配置模型路径
from config import origin_student_path,teacher_path,save_path,resume_from_checkpoint
origin_student_path = origin_student_path
teacher_path = teacher_path
save_path = save_path
resume_from_checkpoint = resume_from_checkpoint


# 配置参数
from config import max_seq_length,dtype,load_in_4bit,weight_decay
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



# 初始化教师模型（更大的7B模型）
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name=teacher_path,  # 7B参数的教师模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
teacher = FastLanguageModel.get_peft_model(
    teacher,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
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

# 加载并预处理Alpaca数据集
from dataset import train_dataset
train_dataset = train_dataset.map(formatting_prompts_func, batched=True, )

# 配置训练参数
args = TrainingArguments(
    output_dir=save_path,  # 输出目录

    num_train_epochs=epoch,  # 训练轮次

    do_train=True,  # 启用训练模式

    # 为什么小train batch会导致更好的效果？
    per_device_train_batch_size=4,  # 单设备批次大小
    gradient_accumulation_steps=16,  # 梯度累积步数
    weight_decay = weight_decay,

    logging_steps=50,  # 日志记录间隔

    save_strategy="epoch",
    save_total_limit=1,
    report_to=["wandb"],
    bf16=True,
    learning_rate=lr,
    lr_scheduler_type='constant',
    optim="adamw_torch_fused",
)

trainer = SFTTrainer(
    model = teacher,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False,
    args = args,
)

# 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
# 需要当前的epoch比上次的大
trainer.train(resume_from_checkpoint=resume_from_checkpoint)