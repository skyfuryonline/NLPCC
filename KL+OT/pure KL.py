# 纯 KL 损失函数
def compute_kl_loss(student_logits, teacher_logits, temp=1.0, reduction='sum', target=None, padding_id=None):
    kl_loss = F.kl_div(F.log_softmax(student_logits / temp, dim=-1),
                       F.softmax(teacher_logits / temp, dim=-1),
                       reduction='none').sum(dim=-1)
    if target is not None and padding_id is not None:
        padding_mask = (target != padding_id).float()
        kl_loss = kl_loss * padding_mask
    return kl_loss.sum() if reduction == 'sum' else kl_loss.mean()

# 自定义 KDTrainer（纯 KL）
class KLTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.scaler = GradScaler(enabled=self.args.bf16 or self.args.fp16)
        self.do_grad_scaling = self.args.bf16 or self.args.fp16

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs_student = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        student_logits = outputs_student.logits
        teacher_logits = teacher_outputs.logits
        loss_ce = outputs_student.loss
        labels = inputs.get('labels', None)

        kl_loss = compute_kl_loss(
            student_logits, teacher_logits,
            temp=1.0,
            reduction="sum",
            target=labels,
            padding_id=-100
        )

        loss_total = 0.5 * kl_loss + 0.5 * loss_ce  # 结合 CE 损失，与原始代码一致
        wandb.log({"train_loss": loss_total.item(), "kl_loss": kl_loss.item()})

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
        return loss.detach()

# 初始化并训练
kl_trainer = KLTrainer(
    model=student,
    teacher_model=teacher,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=args,
)

optimizer = AdamW8bit(kl_trainer.model.parameters(), lr=args.learning_rate)
kl_trainer.optimizer = optimizer
kl_trainer.train(resume_from_checkpoint=False)

# 保存模型
kl_trainer.model.save_pretrained("./results/kl_baseline")