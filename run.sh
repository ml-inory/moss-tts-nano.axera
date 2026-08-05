#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo "=============================================="
echo " MOSS-TTS-Nano-100M 中文语音合成"
echo "=============================================="

PY=${PYTHON:-python3}
$PY python/demo.py --model-dir models --text "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。" \
    --out out.wav

echo "生成完成: $(pwd)/out.wav"
