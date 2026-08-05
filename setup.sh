#!/usr/bin/env bash
set -e

echo "=============================================="
echo " MOSS-TTS-Nano-100M 中文语音合成 环境安装"
echo "=============================================="

PY=${PYTHON:-python3}
$PY -m pip install --upgrade pip -q
$PY -m pip install -r python/requirements.txt -q
# pyaxengine 仅在 AX650 板上需要; x86 开发机可跳过
$PY -m pip install pyaxengine -q 2>/dev/null || echo "提示: pyaxengine 未安装 (x86 开发机正常, 板端运行前请安装)"

echo "检查 Python 依赖..."
$PY - <<'EOF'
import numpy, soundfile
try:
    import onnxruntime
    print("onnxruntime OK")
except Exception as e:
    print("onnxruntime 缺失 (板端 NPU 或本机解码需要):", e)
try:
    import torch
    print("torch OK", torch.__version__)
except Exception as e:
    print("torch 缺失:", e)
    raise
try:
    import transformers
    print("transformers OK", transformers.__version__)
except Exception as e:
    print("transformers 缺失:", e)
    raise
print("环境安装完成 ✅")
EOF
