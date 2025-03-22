#!/bin/bash
# 获取当前目录名称
dir_name=$(basename "$PWD")
# 打包并压缩所有内容（包括隐藏文件）
tar czvf "${dir_name}.tar.gz" .
echo "压缩完成！压缩文件为: ${dir_name}.tar.gz"