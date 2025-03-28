#!/bin/bash

# 激活 Conda 环境
#为了确保 Conda 环境能够正确激活，建议使用 source activate test 而不是 conda activate test。此外，conda init 只需要在首次设置时运行一次。
source activate test

# 测试alpaca数据集
# python ./src/OTtrain.py && python ./src/evaluate.py
# nohup python ./src/FKLtrain.py && python ./src/evaluate.py > output.log 2>&1 &
# nohup python ./src/OTtrain.py && python ./src/evaluate.py > output.log 2>&1 &


# 测试OpusBooks数据集
# python ./src/FKLtrain.py && python ./src/evaluateOpus.py
# python ./src/OTtrain.py && python ./src/evaluateOpus.py
# 使用nohup改进
# wandb offline
# wandb online
# nohup python ./src/OTtrain.py && python ./src/evaluateOpus.py > output.log 2>&1 &
nohup python ./src/FKLtrain.py && python ./src/evaluateOpus.py > output.log 2>&1 &


# 测试summary数据集
# nohup python ./src/OTtrain.py && python ./src/evaluateSummary.py > output.log 2>&1 &
# nohup python ./src/FKLtrain.py && python ./src/evaluateSummary.py > output.log 2>&1 &


# 测试QA数据集
# nohup python ./src/OTtrain.py && python ./src/evaluateQA.py > output.log 2>&1 &
# nohup python ./src/OTtrain.py && python ./src/metrics_based_on_llm.py > output.log 2>&1 &
# nohup python ./src/FKLtrain.py && python ./src/metrics_based_on_llm.py > output.log 2>&1 &