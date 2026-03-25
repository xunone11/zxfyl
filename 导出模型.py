# ========================================
# 功能说明
# ========================================
# 训练完成后，将LoRA权重合并到基座模型，导出完整模型
# 合并后的模型可以直接用Ollama/vLLM等部署

# ========================================
# 参数配置
# ========================================

# 默认路径（会被命令行参数覆盖）
DEFAULT_MODEL_DIR = "/root/model/Qwen2.5-7B-Instruct"
DEFAULT_LORA_DIR = "./lora_output"
DEFAULT_OUTPUT_DIR = "./merged_model"

# ========================================
# 以下为脚本主体
# ========================================

import argparse
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    parser = argparse.ArgumentParser(description="合并LoRA权重到基座模型")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="基座模型路径")
    parser.add_argument("--lora_dir", default=DEFAULT_LORA_DIR, help="LoRA权重路径")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="合并后模型输出路径")
    args = parser.parse_args()

    print(f"基座模型: {args.model_dir}")
    print(f"LoRA权重: {args.lora_dir}")
    print(f"输出目录: {args.output_dir}")

    # 加载tokenizer
    print("\n[1/4] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 加载基座模型
    print("[2/4] 加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载并合并LoRA
    print("[3/4] 合并LoRA权重...")
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model = model.merge_and_unload()

    # 保存合并后的模型
    print(f"[4/4] 保存合并模型到 {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✓ 合并完成！模型已保存到: {args.output_dir}")
    print("  你可以下载 merged_model 目录，用 Ollama 本地部署")


if __name__ == "__main__":
    main()
