#!/bin/bash
# ========================================
# 张雪峰升学规划AI - 一键训练脚本
# 适用于 AutoDL 等云端训练平台
# GPU推荐: RTX 4090 24GB
# ========================================

set -e

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
# 模型和输出保存到AutoDL持久存储，关机不丢失
MODEL_DIR="/root/autodl-fs/model/Qwen2.5-7B-Instruct"
OUTPUT_DIR="/root/autodl-fs/lora_output"
MERGED_DIR="/root/autodl-fs/merged_model"

echo "======================================"
echo "  张雪峰升学规划AI - QLoRA微调"
echo "======================================"
echo "工作目录: $WORK_DIR"
echo ""

# ========== 第1步：安装依赖 ==========
echo "[步骤1/4] 安装依赖..."
pip install llamafactory bitsandbytes modelscope -q
echo "✓ 依赖安装完成"

# ========== 第2步：下载模型 ==========
if [ -f "$MODEL_DIR/config.json" ]; then
    echo "[步骤2/4] 模型已存在，跳过下载"
else
    echo "[步骤2/4] 下载 Qwen2.5-7B-Instruct 模型（约15GB）..."
    modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir "$MODEL_DIR"
    echo "✓ 模型下载完成"
fi

# ========== 第3步：更新配置中的路径 ==========
echo "[步骤3/4] 更新配置路径..."
sed -i "s|model_name_or_path:.*|model_name_or_path: $MODEL_DIR|" "$WORK_DIR/train_qlora.yaml"
sed -i "s|dataset_dir:.*|dataset_dir: $WORK_DIR|" "$WORK_DIR/train_qlora.yaml"
sed -i "s|output_dir:.*|output_dir: $OUTPUT_DIR|" "$WORK_DIR/train_qlora.yaml"
echo "✓ 配置已更新"

# ========== 第4步：开始训练 ==========
echo "[步骤4/4] 开始QLoRA微调训练..."
echo "训练配置:"
echo "  模型: Qwen2.5-7B-Instruct"
echo "  方法: QLoRA (4-bit)"
echo "  LoRA rank: 16"
echo "  Epochs: 3"
echo "  Batch: 2 x 8 = 16"
echo ""

llamafactory-cli train "$WORK_DIR/train_qlora.yaml"

echo ""
echo "======================================"
echo "  ✓ 训练完成！"
echo "======================================"
echo ""

# ========== 第5步：合并导出 ==========
echo "[额外] 合并LoRA权重到完整模型..."
python "$WORK_DIR/导出模型.py" \
    --model_dir "$MODEL_DIR" \
    --lora_dir "$OUTPUT_DIR" \
    --output_dir "$MERGED_DIR"

echo ""
echo "======================================"
echo "  全部完成！"
echo "  LoRA权重: $OUTPUT_DIR"
echo "  合并模型: $MERGED_DIR"
echo "======================================"
