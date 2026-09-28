# A4 中文慢病健康科普模型微调

基础模型为 `Qwen/Qwen2.5-7B-Instruct`，训练数据来自 Hugging Face 的
`FreedomIntelligence/huatuo_encyclopedia_qa`，固定数据版本为
`2989899dd5bed83cf9bd17cf9fff9889705436e2`。项目使用 LLaMA-Factory v0.9.3
执行 LoRA SFT。

## 远端目录

训练资产位于服务器的 `~/wangzeyu`。密码没有写入任何配置或脚本。

Miniforge 已安装到 `~/wangzeyu/miniforge3`，可使用：

```bash
source ~/wangzeyu/miniforge3/etc/profile.d/conda.sh
conda activate base
```

安装后的 conda 版本为 26.3.2，base 环境 Python 版本为 3.13.13。

按课程流程创建的 ModelScope 环境：

```bash
source ~/wangzeyu/miniforge3/etc/profile.d/conda.sh
conda activate modelscope
python --version
python -c "import modelscope; print(modelscope.__version__)"
```

已验证环境为 Python 3.11.16，ModelScope 版本为 1.40.0。

按课程流程创建的 LLaMA-Factory 环境：

```bash
source ~/wangzeyu/miniforge3/etc/profile.d/conda.sh
conda activate llamafactory
cd ~/wangzeyu/LlamaFactory-0.9.3
llamafactory-cli version
```

该环境使用 Python 3.11.16、LLaMA-Factory 0.9.3 和
PyTorch 2.5.1+cu124，已验证可识别 NVIDIA A800 80GB GPU。

## 本地模型备份

桌面 `a4实战` 副本保留了可直接部署的完整合并模型：

```text
models/Qwen2.5-7B-medical-merged
```

项目中还保留了体积较小的 LoRA 适配器备份：

```text
remote_artifacts/adapter_model.safetensors
remote_artifacts/adapter_config.json
```

如果完整模型丢失，可重新下载 `Qwen/Qwen2.5-7B-Instruct`，再使用
`configs/export.yaml` 合并适配器。

## 可复现命令

```bash
cd ~/wangzeyu
source env.sh

python scripts/prepare_data.py --root .
llamafactory-cli train configs/smoke_lora.yaml
llamafactory-cli train configs/train_lora.yaml
llamafactory-cli export configs/export.yaml
```

按《A4项目部署微调展示平台操作指南》使用独立的 Python 3.10 Conda 环境部署合并模型：

```bash
source ~/wangzeyu/miniforge3/etc/profile.d/conda.sh
conda activate vllm
cd ~/wangzeyu

python3 -m vllm.entrypoints.openai.api_server \
  --model ~/wangzeyu/outputs/Qwen2.5-7B-medical-merged \
  --served-model-name Qwen2.5-7B-ft \
  --gpu-memory-utilization 0.30 \
  --max-model-len 1024 \
  --dtype=half \
  --port 22222
```

另开一个终端，进入独立的 Flask Python 3.10 环境并启动 `/chat` 服务：

```bash
source ~/wangzeyu/miniforge3/etc/profile.d/conda.sh
conda activate flask
cd ~/wangzeyu
PORT=20003 \
VLLM_BASE_URL=http://127.0.0.1:22222/v1 \
MODEL_NAME=Qwen2.5-7B-ft \
python scripts/server.py
```

平台 URL 为：

```text
https://262c5a7cd8a04506b7d6dae5f5b22b0f.proxy.nscc-gz.cn:8443/chat
```

已通过公网 `GET /health` 验证该域名直接转发到 20003 端口。
