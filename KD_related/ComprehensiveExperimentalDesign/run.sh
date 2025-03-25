#!/bin/bash

# 测试alpaca数据集
# python ./src/OTtrain.py && python ./src/evaluate.py
# python ./src/FKLtrain.py && python ./src/evaluate.py


# 测试OpusBooks数据集
# python ./src/FKLtrain.py && python ./src/evaluateOpus.py
# python ./src/OTtrain.py && python ./src/evaluateOpus.py


# 使用nohup改进
nohup python ./src/OTtrain.py && python ./src/evaluateOpus.py > output.log 2>&1 &