#!/bin/bash

# 1. 定义源目录和输出文件名
SOURCE_DIR="/home/discover/sam3d_gs/data"
OUTPUT_FILE="/home/discover/sam3d_gs/original_rgb.tar.gz"

# 2. 获取绝对路径，避免 cd 后路径失效
SOURCE_DIR=$(realpath "$SOURCE_DIR")
PARENT_DIR=$(dirname "$SOURCE_DIR")
BASE_NAME=$(basename "$SOURCE_DIR")

# 3. 切换到父目录，确保 tar 打包时的相对路径从源目录名开始
cd "$PARENT_DIR" || exit

echo "正在搜索并打包文件，请稍候..."

# 4. 核心修改部分：
# -type f: 只找文件
# -name "*depth_ori_visual.png*": 匹配包含该字符串的文件名
# tar -czf: 创建压缩包
# -T -: 接收来自管道的文件列表
find "$BASE_NAME" -type f -name "*input_image.png*" | tar -czvf "$OUTPUT_FILE" -T -

echo "打包完成！压缩包位置: $OUTPUT_FILE"