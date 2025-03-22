import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import wandb
os.environ["WANDB_PROJECT"] = "KD"
os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
wandb.init()

origin_student_path = "/root/shared-nvme/model/unsloth/Qwen2.5-1.5B"
teacher_path = "/root/shared-nvme/model/unsloth/Qwen2.5-7B"


from unsloth import FastLanguageModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from losses import compute_fkl
from datasets import load_dataset

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

temperature = 2.0
epoch = 20
lr = 0.0005
reduction = "sum"

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
                kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=temperature,reduction=reduction)

        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl

        return (loss_total, outputs_student) if return_outputs else loss_total

# 学生模型
student, _ = FastLanguageModel.from_pretrained(
    model_name =origin_student_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

student = FastLanguageModel.get_peft_model(
    student,
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

student.print_trainable_parameters()

teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name = teacher_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

teacher.eval()


# 读取数据集
# process dataset
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
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


from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched = True,)


# 训练参数
args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epoch,
    
    do_train=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    
    logging_steps=50,

    save_strategy='epoch',
    save_total_limit=1,
    
    bf16=True,
    learning_rate=lr,
    lr_scheduler_type='constant',
    optim="adamw_torch_fused",
)


trainer = KDTrainer(
    model=student,
    teacher_model=teacher,

    if_use_entropy=True,
    processing_class=tokenizer,
    train_dataset=dataset,
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=args,
)


# 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
trainer.train(resume_from_checkpoint=False)