# 导入unsloth库中的FastLanguageModel模块，用于高效加载和训练大模型
from unsloth import FastLanguageModel

# 导入PyTorch相关模块
# 导入自定义的损失函数模块
# 导入Hugging Face的dataset模块


# 配置参数
max_seq_length = 2048  # 最大序列长度，支持RoPE扩展
dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = True  # 使用4bit量化减少内存占用

text_input = 'Create a MongoDB query to retrieve data within a specific range.'


# 初始化学生模型（使用unsloth的优化实现）
old_student, _ = FastLanguageModel.from_pretrained(
    model_name="/root/shared-nvme/model/Qwen2.5-1.5B-bnb-4bit",  # 1.5B参数的千问模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # 4bit量化加载
)
old_student = FastLanguageModel.get_peft_model(
    old_student,
    r=16,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  # 目标注意力模块
                    "gate_proj", "up_proj", "down_proj", ],  # FFN模块
    lora_alpha=16,  # LoRA alpha参数
    lora_dropout=0,  # 无dropout
    bias="none",  # 不训练偏置参数
    use_gradient_checkpointing="unsloth",  # 使用优化版梯度检查点
    random_state=3407,  # 随机种子
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
FastLanguageModel.for_inference(old_student)


# 初始化学生模型（使用unsloth的优化实现）
student, _ = FastLanguageModel.from_pretrained(
    model_name="/root/shared-nvme/results/checkpoint-620",  # 1.5B参数的千问模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # 4bit量化加载
)
student = FastLanguageModel.get_peft_model(
    student,
    r=16,  # LoRA秩
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  # 目标注意力模块
                    "gate_proj", "up_proj", "down_proj", ],  # FFN模块
    lora_alpha=16,  # LoRA alpha参数
    lora_dropout=0,  # 无dropout
    bias="none",  # 不训练偏置参数
    use_gradient_checkpointing="unsloth",  # 使用优化版梯度检查点
    random_state=3407,  # 随机种子
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# 初始化teacher模型
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/shared-nvme/model/Qwen2.5-7B",  # 7B参数的千问模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # 4bit量化加载
)

FastLanguageModel.for_inference(student)

# 示例输入
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# input_text = alpaca_prompt.format(
#     "Give three tips for staying healthy.",
#     "",
#     ""
# )
input_text = alpaca_prompt.format(
    text_input,
    "",
    ""
)

# 转换为 token
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 假设使用 GPU

# 生成输出
outputs = old_student.generate(**inputs, max_new_tokens=200, temperature=0.75)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("训练前：学生输出： ",response)



# 示例输入
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# input_text = alpaca_prompt.format(
#     "Give three tips for staying healthy.",
#     "",
#     ""
# )
input_text = alpaca_prompt.format(
    text_input,
    "",
    ""
)

# 转换为 token
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 假设使用 GPU

# 生成输出
outputs = student.generate(**inputs, max_new_tokens=200, temperature=0.75)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("学生输出： ",response)


#-----------------------------------------------------
#下列内容作为teacher模型对比

FastLanguageModel.for_inference(teacher)

# 示例输入
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# input_text = alpaca_prompt.format(
#     "Give three tips for staying healthy.",
#     "",
#     ""
# )

input_text = alpaca_prompt.format(
    text_input,
    "",
    ""
)
# 转换为 token
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  # 假设使用 GPU

# 生成输出
outputs = teacher.generate(**inputs, max_new_tokens=200, temperature=0.75)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("教师输出： ",response)