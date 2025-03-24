max_seq_length = 2048
dtype = None
load_in_4bit = True

epoch = 10
lr = 0.0005
weight_decay = 0.01
topk = 150
chunk_size = 4
alpha = 0.5
temperature = 2.0
reduction = "sum"

# 即qwen2.5-1.5B的模型存放位置
origin_student_path = "/root/shared-nvme/ComprehensiveExperimentalDesign/fine_tuning_train/save_model/student/Qwen2.5-1.5B"
# 即原始qwen2.5-7B的模型存放位置
teacher_path = "/root/shared-nvme/ComprehensiveExperimentalDesign/fine_tuning_train/save_model/teacher/Qwen2.5-7B"
# 即微调后的qwen2.5-7B的模型存放位置
save_path = "/root/shared-nvme/ComprehensiveExperimentalDesign/fine_tuning_train/save_model/chpt/Qwen2.5-7B_ft"

resume_from_checkpoint = False
# 用于evaluate中导出结果记录的名字
run_name = ""