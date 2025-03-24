max_seq_length = 2048
dtype = None
load_in_4bit = True


epoch = 20
lr = 0.0005
temperature = 2.0
reduction = "sum"
topk = 150
alpha = 0.5
chunk_size = 4


origin_student_path = "../models/unsloth/Qwen2.5-1.5B"
teacher_path = "../models/unsloth/Qwen2.5-7B"
save_path = "../models/results"
resume_from_checkpoint = False


# 用于evaluate中导出结果记录的名字
run_name = "OT_KD_epoch=15_topk=200"
