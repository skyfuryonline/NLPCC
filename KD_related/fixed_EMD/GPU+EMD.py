import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
import numpy as np
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import wandb
import os

# 配置参数
max_seq_length = 1024
dtype = torch.float32
load_in_4bit = True
chunk_size = 500  # 减小分块大小
student_chunk_size = 500  # 减小 student 分块大小

origin_student_path = "/root/shared-nvme/model/Qwen2.5-1.5B-bnb-4bit"
teacher_path = "/root/shared-nvme/model/Qwen2.5-7B"
cost_matrix_dir = "/root/shared-nvme/fixed_EMD/cost_matrix_chunks"
# 用于存储**成本矩阵（cost matrix）**分块的目录。

def align_embeddings(teacher_emb, student_emb, method="linear", device="cuda"):
    '''
    作用:
    词嵌入对齐（Embedding Alignment）：将教师模型的词嵌入投影到学生模型的词嵌入空间，以减少特征维度差异带来的影响。
    方法
    linear：使用线性变换（全连接层）调整维度。
    pca：使用**PCA（主成分分析）**降维，使教师嵌入与学生嵌入维度匹配。
    '''
    V_teacher, D_teacher = teacher_emb.shape
    V_student, D_student = student_emb.shape

    if D_teacher == D_student:
        return teacher_emb

    if method == "linear":
        projection = torch.nn.Linear(D_teacher, D_student, bias=False).to(device)
        return projection(teacher_emb)
    elif method == "pca":
        teacher_emb_fp32 = teacher_emb.to(dtype=torch.float32)
        teacher_emb_centered = teacher_emb_fp32 - teacher_emb_fp32.mean(dim=0, keepdim=True)
        try:
            u, s, v = torch.pca_lowrank(teacher_emb_centered, q=D_student)
            aligned_emb = teacher_emb_fp32 @ v
            return aligned_emb
        except RuntimeError as e:
            raise ValueError(f"PCA computation failed: {str(e)}")
    else:
        raise ValueError(f"Unsupported alignment method: {method}")


def precompute_cost_matrix_chunks(teacher_emb, student_emb, save_dir, chunk_size, student_chunk_size, method="linear"):
    '''
    作用:
    计算 Wasserstein（EMD）损失所需的成本矩阵（cost matrix）
    分块存储，减少计算和内存占用：
    余弦相似度计算：1 - F.cosine_similarity(...)，转换为“距离”。
    结果存入磁盘，用于后续训练。
    '''
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda")
    with torch.no_grad():
        teacher_emb = teacher_emb.to(device=device, dtype=torch.float32)
        student_emb = student_emb.to(device=device, dtype=torch.float32)
        teacher_emb_aligned = align_embeddings(teacher_emb, student_emb, method=method, device=device)

        V_teacher, D_teacher = teacher_emb_aligned.shape
        V_student, D_student = student_emb.shape

        for i in range(0, V_teacher, chunk_size):
            chunk_end = min(i + chunk_size, V_teacher)
            chunk_teacher = teacher_emb_aligned[i:chunk_end]
            cost_chunk_full = torch.zeros(chunk_end - i, V_student, device=device, dtype=torch.float32)

            for j in range(0, V_student, student_chunk_size):
                student_end = min(j + student_chunk_size, V_student)
                chunk_student = student_emb[j:student_end]
                cost_chunk = 1 - F.cosine_similarity(
                    chunk_teacher.unsqueeze(1),
                    chunk_student.unsqueeze(0),
                    dim=-1
                )
                cost_chunk_full[:, j:student_end] = cost_chunk

            chunk_path = os.path.join(save_dir, f"cost_matrix_chunk_{i}.pt")
            torch.save(cost_chunk_full.cpu(), chunk_path)
            print(f"Saved chunk {i} to {chunk_end - 1}, shape: {cost_chunk_full.shape}, path: {chunk_path}")
    return V_teacher, V_student


def load_cost_matrix_chunk(chunk_idx, chunk_size, V_teacher, save_dir, device="cuda"):
    start_idx = chunk_idx * chunk_size
    if start_idx >= V_teacher:
        raise ValueError(f"Chunk index {chunk_idx} out of range for V_teacher {V_teacher}")
    chunk_path = os.path.join(save_dir, f"cost_matrix_chunk_{start_idx}.pt")
    cost_chunk = torch.load(chunk_path, map_location=device).to(dtype=torch.float32)
    return cost_chunk


def compute_emd_loss(
        teacher_logits,
        student_logits,
        cost_matrix_dir,
        chunk_size,
        V_teacher,
        V_student,
        temperature=2.0,
        reduction='sum',
        mask=None
):
    '''
    作用：
    计算基于 Sinkhorn-Knopp 逼近的 EMD 损失：
    SamplesLoss(loss="sinkhorn")：用于计算 Wasserstein-1（EMD）。
    temperature=2.0：平滑 softmax 温度。
    chunk_size：减少计算开销。


    '''
    batch_size, seq_len, V_teacher_input = teacher_logits.shape
    assert V_teacher_input == V_teacher and student_logits.shape[2] == V_student, "Logits vocab size mismatch"

    device = torch.device("cuda")
    compute_dtype = torch.float32

    teacher_logits = teacher_logits.to(device=device, dtype=compute_dtype)
    student_logits = student_logits.to(device=device, dtype=compute_dtype)
    if mask is not None:
        mask = mask.to(device=device)

    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    teacher_probs = teacher_probs.view(batch_size * seq_len, V_teacher)
    student_probs = student_probs.view(batch_size * seq_len, V_student)

    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.05, scaling=0.8, backend="tensorized")
    emd_loss = torch.zeros(batch_size * seq_len, device=device, dtype=compute_dtype)
    num_chunks = (V_teacher + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, V_teacher)
        cost_chunk = load_cost_matrix_chunk(chunk_idx, chunk_size, V_teacher, cost_matrix_dir, device)

        t_probs_chunk = teacher_probs[:, start_idx:end_idx]
        s_probs = student_probs

        if t_probs_chunk.shape[1] < chunk_size:
            t_probs_chunk = F.pad(t_probs_chunk, (0, chunk_size - t_probs_chunk.shape[1]), value=0)

        chunk_loss = loss_fn(t_probs_chunk, s_probs, cost_chunk)
        emd_loss += chunk_loss / num_chunks

    emd_loss = emd_loss.view(batch_size, seq_len)
    if mask is not None:
        emd_loss = emd_loss * mask

    if reduction == 'mean':
        valid_count = (batch_size * seq_len) if mask is None else mask.sum()
        return emd_loss.sum() / max(valid_count, 1)
    elif reduction == 'sum':
        return emd_loss.sum()
    elif reduction == 'none':
        return emd_loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")


def compute_wasserstein_loss(logits, teacher_logits, target, cost_matrix_dir, chunk_size, V_teacher, V_student,
                             padding_id=-100, reduction="sum", temp=2.0, wasserstein_version=1):
    mask = (target != padding_id).float() if target is not None else None
    if wasserstein_version == 1:
        return compute_emd_loss(teacher_logits, logits, cost_matrix_dir, chunk_size, V_teacher, V_student,
                                temperature=temp, reduction=reduction, mask=mask)
    else:
        raise ValueError(f"Unsupported wasserstein_version: {wasserstein_version}")


class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, if_use_entropy=None, wasserstein_version=1, cost_matrix_dir=None,
                 chunk_size=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.wasserstein_version = wasserstein_version
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self.cost_matrix_dir = cost_matrix_dir
        self.chunk_size = chunk_size

        with torch.no_grad():
            teacher_emb = self.teacher_model.get_input_embeddings().weight
            student_emb = self.model.get_input_embeddings().weight
            self.teacher_emb_reduced = align_embeddings(teacher_emb, student_emb, method="pca", device="cuda")
            self.student_emb_reduced = student_emb
            self.V_teacher, self.V_student = teacher_emb.shape[0], student_emb.shape[0]

            if not os.path.exists(os.path.join(cost_matrix_dir, f"cost_matrix_chunk_0.pt")):
                precompute_cost_matrix_chunks(self.teacher_emb_reduced, self.student_emb_reduced, cost_matrix_dir,
                                              self.chunk_size, student_chunk_size=1000)
                print(f"Computed and saved cost matrix chunks to {cost_matrix_dir}")
            else:
                print(f"Using precomputed cost matrix chunks from {cost_matrix_dir}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        loss = outputs_student.loss
        logits = outputs_student.logits
        teacher_logits = teacher_outputs.logits

        wasserstein_loss = compute_wasserstein_loss(
            logits, teacher_logits, inputs.get('labels'), self.cost_matrix_dir, self.chunk_size, self.V_teacher,
            self.V_student
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

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


dataset = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")
dataset = dataset.map(formatting_prompts_func, batched=True)

val_dataset = load_dataset("yahma/alpaca-cleaned", split="train[2000:3000]")
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=20,
    do_train=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=["wandb"],
    learning_rate=0.0005,
    lr_scheduler_type='constant',
    optim="adamw_torch_fused",
)

trainer = KDTrainer(
    model=student,
    teacher_model=teacher,
    if_use_entropy=True,
    wasserstein_version=1,
    cost_matrix_dir=cost_matrix_dir,
    chunk_size=chunk_size,
    processing_class=tokenizer,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

trainer.train(resume_from_checkpoint=False)