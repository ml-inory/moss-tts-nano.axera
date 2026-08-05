# EXPORT 报告

## 模型

- 来源: `OpenMOSS-Team/MOSS-TTS-Nano-100M` (Apache-2.0) + `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`
- 架构: 100M 自回归 TTS-LLM (12 层 GPT-2 + RoPE + 16 codebook 头 + 1 层 local transformer)
  + 20M 音频 codec 解码器 (CAT, 48kHz 双声道, 16 codebook @12.5Hz)

## 部署拆分

| 组件 | 运行位置 | 说明 |
|------|----------|------|
| TTS-LLM (100M) | AX650 CPU (PyTorch fp32) | 自回归采样无法静态化; 官方定位即 CPU 友好 |
| codec 解码器 (20M) | AX650 NPU3 (AXMODEL) | 纯前馈, Pulsar2 编译 |
| codebook 查表 | CPU (numpy/C++) | Pulsar2 不支持 int64 输入, 查表留在主机侧 |

## ONNX 导出

- `codec_decoder.onnx`: 静态 shape, opset 17
  - 输入 `codes_emb` float32 [1,512,64] (16 codebook 查表求和)
  - 输出 `waveform` float32 [1,2,245760] (48kHz 双声道, 5.12s)
- onnxsim 精简: 240MB → 50MB
- 对分验证: PyTorch vs ONNX, 5 组随机+真实样本
  - min cosine = 1.000000, MAE = 2.5e-7
- CPU 查表权重: `codec_quantizer.npz` (2.2MB) / `codec_quantizer_cpp.bin` (2.4MB),
  numpy 与 torch 逐位一致 (cosine = 1.0, max diff = 0)

## 校准数据

- 11 份样本, 全部来自真实中文 voice-clone TTS 推理 (8 段中文文本)
- 每份: `codes_emb` [1,512,64] float32
- 生成脚本: `gen_chinese_tts_calib.py` (固定种子可复现)

## 中文优先措施

- 内置官方中文参考音色 `zh_1.wav`, voice-clone 模板与官方 demo 一致
- LLM 权重 fp32 原样部署, 中文合成质量无损
- 采样参数对齐官方 demo (text_temperature=1.0, audio_temperature=0.8,
  audio_top_p=0.95, repetition_penalty=1.2)
