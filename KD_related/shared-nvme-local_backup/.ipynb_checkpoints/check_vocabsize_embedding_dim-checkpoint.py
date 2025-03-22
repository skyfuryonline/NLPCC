from unsloth import FastLanguageModel

# 配置参数
max_seq_length = 1024
dtype = None
load_in_4bit = True

origin_student_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-1.5B"
teacher_path = "/root/shared-nvme-local_backup/model/unsloth/Qwen2.5-7B"

student,s_tokenizer = FastLanguageModel.from_pretrained(
    origin_student_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

teacher,t_tokenizer = FastLanguageModel.from_pretrained(
    teacher_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("词表大小是：---------------------")
print(s_tokenizer.vocab_size)
# 151643
print(t_tokenizer.vocab_size)
# 151643

print("完整词表大小是：---------------------")
print(len(s_tokenizer.get_vocab()))
# 151665
print(len(t_tokenizer.get_vocab()))
# 151665

print("嵌入层维度大小是：---------------------")
print(student.get_input_embeddings().embedding_dim)
# 1536
print(teacher.get_input_embeddings().embedding_dim)
# 3584

print("隐藏层维度大小是：---------------------")
print(student.config.hidden_size)
# 1536
print(teacher.config.hidden_size)
# 3584


# 词表大小是：---------------------
# 151643
# 151643
# 完整词表大小是：---------------------
# 151665
# 151665
# 嵌入层维度大小是：---------------------
# 1536
# 3584
# 隐藏层维度大小是：---------------------
# 1536
# 3584
