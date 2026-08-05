# MOSS-TTS-Nano-100M 中文语音合成 (AX650 / NPU3)

基于复旦大学 OpenMOSS 团队的 **MOSS-TTS-Nano-100M**（Apache-2.0），在爱芯 AX650（NPU3）上部署的
中文优先语音合成包。模型仅 **0.1B 参数**，输出 **48kHz 双声道** 中文语音。

- 100M 自回归 LLM：板端 CPU 运行（PyTorch fp32，原权重，中文合成质量与官方一致）
- NPU3 加速选件: global decode/local 已编译为 AXMODEL (约 4 倍 LLM 加速), 见 reports/
- 音频 codec 解码器（MOSS-Audio-Tokenizer-Nano）：编译为 **AXMODEL，运行在 NPU3**
  （默认，快 2.6 倍）；也可选 FP32 ONNX（参考级精度，cosine=1.0）
- 内置中文参考音色（zh_1.wav），voice-clone 模式合成，无需额外录制

## 快速开始（两步）

```bash
bash setup.sh      # 安装依赖（约几分钟）
bash run.sh        # 合成中文语音到 out.wav
```

## 目录结构

```
models/
  llm/                 # 100M TTS-LLM 权重与运行时代码（CPU）
  codec/               # codec 解码器 AXMODEL/ONNX + 查表权重 + 参考音色
python/                # Python SDK（推荐）
cpp/                   # C++ codec 解码器 SDK（AX Engine 直连）
model_convert/         # 可复现的导出/编译脚本（ONNX + Pulsar2）
reports/               # 导出/编译/仿真/性能报告
```

## 更多用法

```bash
# 自定义中文文本（支持长文本自动分句）
python python/demo.py --model-dir models \
  --text "今天天气真好，我们一起去公园散步吧。" \
  --out my_voice.wav

# 板上使用 NPU 解码 (provider=axengine 自动优先)
python python/demo.py --model-dir models --provider axengine
```

## 性能与限制

- 生成速度：CPU LLM 约 10-13 帧/秒（12.5 帧/秒即实时），codec 解码在 NPU3
- 输出格式：48kHz / 双声道 / 16-bit PCM WAV
- 语言：中文最优（内置 zh 参考音色），也支持官方 20 种语言
- 长文本：SDK 自动按句切分并保留参考音色
- 已知限制：AXMODEL 为混合精度（全部计算 FP32 + LayerNorm S16），波形与 FP32 参考的
  cosine 相似度 0.952±0.019（见 reports/simulate_report.md）；需要参考级精度时用
  `--provider onnxruntime`（板端 CPU 约慢 2.6 倍）

## 许可证

模型与代码遵循 Apache-2.0（OpenMOSS-Team）。商业使用请查阅上游许可证。
