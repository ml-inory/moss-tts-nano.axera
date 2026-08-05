#!/usr/bin/env bash
# 下载 MOSS-TTS-Nano-100M LLM 权重 (约 234MB)
set -e
cd "$(dirname "$0")"
mkdir -p models/llm
if [ -f models/llm/pytorch_model.bin ]; then
  echo "权重已存在: models/llm/pytorch_model.bin"
  exit 0
fi
echo "从 HuggingFace 下载 LLM 权重 ..."
pip install -q huggingface_hub
HF_ENDPOINT=https://hf-mirror.com python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import shutil, os
p = snapshot_download(repo_id="OpenMOSS-Team/MOSS-TTS-Nano-100M",
                      allow_patterns=["pytorch_model.bin"])
shutil.copy(os.path.join(p, "pytorch_model.bin"), "models/llm/pytorch_model.bin")
print("done")
PYEOF
echo "LLM 权重就绪"
