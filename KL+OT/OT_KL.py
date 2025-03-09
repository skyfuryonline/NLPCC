from ott.geometry import pointcloud
from ott.tools import sinkhorn_divergence

# pip install ott-jax

# OT+KL 损失函数
def compute_ot_kl_loss(student_logits, teacher_logits, student_embeddings, teacher_embeddings,
                      projection, target=None, padding_id=None, temp=1.0, reduction='sum', chunk_size=2, alpha=0.2):
    device = student_logits.device
    batch_size, seq_len, vocab_size_student = student_logits.shape
    _, _, vocab_size_teacher = teacher_logits.shape

    total_loss = 0.0
    min_vocab_size = min(vocab_size_student, vocab_size_teacher)

    for batch_start in range(0, batch_size, chunk_size):
        batch_end = min(batch_start + chunk_size, batch_size)
        chunk_student_logits = student_logits[batch_start:batch_end]
        chunk_teacher_logits = teacher_logits[batch_start:batch_end]

        with torch.cuda.amp.autocast():
            student_probs = F.softmax(chunk_student_logits / temp, dim=-1)
            teacher_probs = F.softmax(chunk_teacher_logits / temp, dim=-1)

            student_emb_proj = projection(student_embeddings[:min_vocab_size])
            teacher_emb = teacher_embeddings[:min_vocab_size]
            ot_loss = sinkhorn_divergence.sinkhorn_divergence(
                pointcloud.PointCloud,
                student_emb_proj, teacher_emb,
                a=student_probs.view(-1, min_vocab_size),
                b=teacher_probs.view(-1, min_vocab_size),
                sinkhorn_iterations=20,
                epsilon=0.1
            ).divergence

            kl_loss = F.kl_div(F.log_softmax(chunk_student_logits / temp, dim=-1),
                              F.softmax(chunk_teacher_logits / temp, dim=-1),
                              reduction='none').sum(dim=-1)

            if target is not None and padding_id is not None:
                padding_mask = (target[batch_start:batch_end] != padding_id).float()
                ot_loss = ot_loss.view(batch_end - batch_start, seq_len) * padding_mask
                kl_loss = kl_loss * padding_mask

            total_loss += alpha * ot_loss.sum() + (1 - alpha) * kl_loss.sum()

    return total_loss if reduction == 'sum' else total_loss / (batch_size * seq_len)

# 自定义 KDTrainer（优化 OT+KL）
class OTKLTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, student_embed_dim=1536, teacher_embed_dim=3584, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.projection = nn.Sequential(
            nn.Linear(student_embed_dim, teacher_embed_dim, bias=False, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(teacher_embed_dim, teacher_embed_dim, bias=False, dtype=torch.bfloat16)
        ).to(self.args.device)
        self.scaler = GradScaler(enabled=self.args.bf16 or self.args.fp16)
        self.do_grad_scaling = self.args.bf16 or self.args.fp16

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_logits = outputs_student.logits
        teacher_logits = teacher_outputs.logits
        loss_ce = outputs_student.loss

        student_embeddings = model.get_input_embeddings().weight
        teacher_embeddings = self.teacher_model.get_input_embeddings().weight
        labels = inputs.get('labels', None)

        ot_kl_loss = compute_ot_kl_loss(
            student_logits, teacher_logits,
            student_embeddings=student_embeddings,
            teacher_embeddings=teacher_embeddings,
            projection=self.projection,
            target=labels,
            padding_id=-100,
            temp=1.0,
            reduction="sum",
            chunk_size=2,
            alpha=0.2
        )

        loss_total = 0.5 * ot_kl_loss + 0.5 * loss_ce
        wandb.log({"train_loss": loss_total.item(), "ot_kl_loss": ot_kl_loss.item()})

        return (loss_total, outputs_student) if return_outputs else loss_total

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.lr_scheduler.step()
        model.zero_grad()
        self.projection.zero_grad()
        return loss.detach()

# 初始化并训练
otkl_trainer = OTKLTrainer(
    model=student,
    teacher_model=teacher,
    student_embed_dim=1536,
    teacher_embed_dim=3584,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

optimizer = AdamW8bit(
    list(otkl_trainer.model.parameters()) + list(otkl_trainer.projection.parameters()),
    lr=args.learning_rate
)
otkl_trainer.optimizer = optimizer
otkl_trainer.train(resume_from_checkpoint=False)

# 保存模型
otkl_trainer.model.save_pretrained("./results/otkl_optimized")