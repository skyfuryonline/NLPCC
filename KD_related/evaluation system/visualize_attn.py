import torch
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel


# 配置参数
max_seq_length = 2048  # 最大序列长度，支持RoPE扩展
dtype = None  # 自动检测数据类型（Tesla T4/V100用float16，Ampere用bfloat16）
load_in_4bit = True  # 使用4bit量化减少内存占用

# Load the models and tokenizer
teacher, tokenizer = FastLanguageModel.from_pretrained(
    model_name="",  # 7B参数的教师模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
student, _ = FastLanguageModel.from_pretrained(
    model_name="",  # 1.5B参数的千问模型
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,  # 4bit量化加载
)

# Ensure the models is in evaluation system mode
teacher.eval()
student.eval()

# Example input text
input_text = ""

# 定义Alpaca格式的prompt模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 获取结束符

# Tokenize the input text
inputs = tokenizer(alpaca_prompt.format(input_text,"","")+EOS_TOKEN, return_tensors="pt")

# Forward pass with attention outputs
with torch.no_grad():
    teacher_outputs = teacher(**inputs, output_attentions=True)
    student_outputs = student(**inputs,output_attentions=True)

# Get the last layer attention scores for teacher and student
teacher_attention_weights = teacher_outputs.attentions[-1].squeeze().cpu().numpy()
student_attention_weights = student_outputs.attentions[-1].squeeze().cpu().numpy()


# Function to visualize attention weights
def visualize_attention(attention_weights, title):
    fig, ax = plt.subplots(figsize=(16, 12))
    cax = ax.matshow(attention_weights, cmap='viridis')
    fig.colorbar(cax)
    plt.title(title)
    plt.xlabel('Token Index')
    plt.ylabel('Token Index')
    plt.show()

# Visualize the attention weights for the teacher models
visualize_attention(teacher_attention_weights, 'Teacher Model Attention Weights')

# Visualize the attention weights for the student models
visualize_attention(student_attention_weights, 'Student Model Attention Weights')