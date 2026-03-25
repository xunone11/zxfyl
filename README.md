# 张雪峰升学规划AI - QLoRA微调训练包

## 推荐配置
- **GPU**: RTX 4090 24GB (AutoDL ¥1.98/时)
- **显存占用**: ~20GB (QLoRA 4-bit)
- **总花费**: ¥1-2

## 一键启动

```bash
# 上传本目录到服务器后，执行：
cd /root/训练包
bash 一键启动.sh
```

脚本会自动完成：安装依赖 → 下载模型 → 开始训练 → 合并导出

## 文件说明

| 文件 | 说明 |
|------|------|
| `一键启动.sh` | 一键安装+训练+导出脚本 |
| `train_qlora.yaml` | QLoRA微调配置（Qwen2.5-7B + LoRA rank16） |
| `dataset_info.json` | LLaMA-Factory数据集注册 |
| `train_data_b.json` | 2445条高质量QA训练集（主数据集） |
| `train_data_a.json` | 1254条备用数据集 |
| `导出模型.py` | 训练完成后合并LoRA权重到完整模型 |

## 训练完成后

训练完成后会在持久存储中生成：
- `/root/autodl-fs/lora_output/` - LoRA权重文件（几十MB）
- `/root/autodl-fs/merged_model/` - 合并后的完整模型（约15GB）

**下载 `merged_model/` 目录到本地**，即可用 Ollama 部署。
关机后这些文件仍保留在 autodl-fs 中。

## 手动执行（如果一键脚本有问题）

```bash
# 1. 安装依赖
pip install llamafactory bitsandbytes modelscope

# 2. 下载模型
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /root/autodl-fs/model/Qwen2.5-7B-Instruct

# 3. 开始训练
llamafactory-cli train train_qlora.yaml

# 4. 合并导出
python 导出模型.py --model_dir /root/autodl-fs/model/Qwen2.5-7B-Instruct --lora_dir /root/autodl-fs/lora_output --output_dir /root/autodl-fs/merged_model
```


## 注意事项

# 1. AutoDL有内置的学术加速代理。你现在直接在服务器终端执行这条命令，立刻加速：
source /etc/network_turbo
# 2.清华源安装依赖
pip install llamafactory bitsandbytes modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3.# 用screen后台运行，SSH断了也不怕
screen -S train
cd /root/训练包 && bash 一键启动.sh
