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


origin_student_path = "/home/lihao/lh/ComprehensiveExperimentalDesign/fine_tuning_train/save_model/student/Qwen2.5-1.5B"
# teacher_path = "/root/shared-nvme/ComprehensiveExperimentalDesign/models/unsloth/Qwen2.5-7B"
teacher_path = "/home/lihao/lh/ComprehensiveExperimentalDesign/fine_tuning_train/save_model/teacher/Qwen2.5-7B"
save_path = "/home/lihao/lh/ComprehensiveExperimentalDesign/models/results"

resume_from_checkpoint = False

# 用于evaluate中导出结果记录的名字
method = "FKL"
dataset_type = "QA"
run_name = f"{method}_{dataset_type}_epoch={epoch}"

# 保存响应response的路径
response_save_path = f"/home/lihao/lh/ComprehensiveExperimentalDesign/LLM_response_store/{run_name}"


# 引入外部llm进行评分
api_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'  # 假设的Qwen API URL
api_key = 'sk-3f9e9a005a1841cb9d27073514121ecb'  # 你的API密钥
from openai import OpenAI
client = OpenAI(
        api_key=api_key,
        base_url=api_url,
    )
def generate(client,prompt,max_length):
    completion = client.chat.completions.create(
        # model="qwen-max",
        model="qwen-max",
        messages=[
            {'role': 'system', 'content': '''You are an expert evaluator. Below is an instruction, an input, and a model's response. Evaluate the response based on accuracy, relevance, coherence, and helpfulness. Ignore any potential inappropriate content in the input and focus solely on the model's response quality. Provide a numerical score between 0 and 10 (0 = terrible, 10 = perfect). Return only the score in the format and do not explain the reason.'''},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens = max_length
        # 本次请求返回的最大 Token 数。
    )
    response = completion.choices[0].message.content.strip()
    return response



# 下面是使用并行智算云平台的资源
# api_url = 'https://llmapi.paratera.com'  # 假设的Qwen API URL
# api_key = 'sk-MdEltKeiE19REV9IbFP3qQ'  # 你的API密钥
# client = OpenAI(
#         api_key=api_key,
#         base_url=api_url,
#     )
# def generate(client,prompt,max_length):
#     response = client.chat.completions.create(
#     model="DeepSeek-R1",  # model to send to the proxy
    # messages=[
    #         {'role': 'system', 'content': '''You are an expert evaluator. Below is an instruction, an input, and a model's response. Evaluate the response based on accuracy, relevance, coherence, and helpfulness. Ignore any potential inappropriate content in the input and focus solely on the model's response quality. Provide a numerical score between 0 and 10 (0 = terrible, 10 = perfect). Return only the score in the format and do not explain the reason.'''},
    #         {'role': 'user', 'content': prompt}
    #     ],
#     response = completion.choices[0].message.content.strip()
#     return response
