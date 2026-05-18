#!/bin/bash

# CUDA开发容器启动脚本
# 用法: ./start_cuda_container.sh [镜像名] [容器名]
# 示例: ./start_cuda_container.sh nvidia/cuda:12.1.0-devel-ubuntu22.04 cuda-dev

# 获取参数
IMAGE_NAME="${1:-nvcr.io/nvidia/pytorch:25.08-py3}"
CONTAINER_NAME="${2:-lsmps-env}"

# 获取当前目录
CURRENT_DIR=$(pwd)

echo "=========================================="
echo "启动CUDA开发容器"
echo "=========================================="
echo "镜像名: $IMAGE_NAME"
echo "容器名: $CONTAINER_NAME"
echo "工作目录: $CURRENT_DIR -> /workspace"
echo "GPU: 全部可用"
echo "共享内存: 1GB"
echo "=========================================="

# 检查容器是否已存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "警告: 容器 '$CONTAINER_NAME' 已存在"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "停止并删除现有容器..."
        docker stop "$CONTAINER_NAME" 2>/dev/null
        docker rm "$CONTAINER_NAME" 2>/dev/null
    else
        echo "使用现有容器..."
        docker start "$CONTAINER_NAME" 2>/dev/null
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit 0
    fi
fi

# 构建docker run命令
docker run -it \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=1g \
    --ipc=host \
    -v "$CURRENT_DIR:/workspace" \
    -w /workspace \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    "$IMAGE_NAME" \
    /bin/bash -c "
        # 检查并安装ncu（如果不存在）
        if ! command -v ncu &> /dev/null; then
            echo '检测到ncu未安装，正在安装...'
            # 尝试从apt安装（适用于Ubuntu镜像）
            if command -v apt-get &> /dev/null; then
                apt-get update -qq && \
                apt-get install -y -qq nsight-compute || \
                echo '警告: 无法通过apt安装ncu，请手动安装或使用包含ncu的镜像'
            else
                echo '警告: 无法自动安装ncu，请手动安装或使用包含ncu的镜像'
            fi
        else
            echo 'ncu已可用'
        fi
        # 启动bash
        exec /bin/bash
    "


