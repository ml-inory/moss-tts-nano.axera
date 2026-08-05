# Python SDK

## 环境安装

```bash
pip install -r requirements.txt
```

依赖：`torch`、`transformers==4.57.1`、`sentencepiece`、`numpy`、`onnxruntime`、
`soundfile`；AX650 板上另需 `pyaxengine`。

## 快速使用

```python
from moss_tts_sdk import MossTTSNano

tts = MossTTSNano("models")                      # provider 默认 auto
result = tts.synthesize("你好，欢迎使用模思智能。", "out.wav")
print(result["duration_s"], "秒")
```

## 命令行

```bash
python demo.py --text "今天天气真好。" --out out.wav
python demo.py --provider axengine --text "中文优先" --out out.wav
```

## API

### MossTTSNano(model_dir, provider="auto", device="cpu")

- `model_dir`：包内 `models/` 目录
- `provider`：`auto`（有 axmodel 且装 pyaxengine 时用 NPU，否则 onnxruntime）| `axengine` | `onnxruntime`

### synthesize(text, output_path, max_new_frames=375, seed=42, do_sample=True)

返回 `dict`：

- `audio_path`：输出 wav 路径
- `sample_rate`：48000
- `channels`：2
- `frames`：音频 token 帧数（12.5 帧/秒）
- `duration_s`：音频时长（秒）
- `provider`：实际解码后端（axengine / onnxruntime）

## 预处理说明

1. 中文文本按标点自动分句，逐句生成
2. 每句按官方 voice-clone 模板拼接文本与参考音频 token（内置 `zh_1.wav` 音色）
3. LLM 自回归生成 16 个 codebook 的音频 token（12.5 Hz）
4. CPU 查表得到 code 嵌入，送入 codec 解码器（NPU AXMODEL 或 ONNX）
5. 输出 48kHz 双声道波形，clamp 到 [-1,1] 后写 int16 WAV

## 数值稳定性说明

LLM 在 x86 高负载环境下偶发 fp32 denormal 导致的 NaN，SDK 已内置四重防护：
固定线程数、`torch.set_flush_denormal(True)`、-1e9 注意力掩码、换种子自动重试。
AX650 板端（aarch64）不受影响。
