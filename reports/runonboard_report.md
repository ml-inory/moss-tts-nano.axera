# RUNONBOARD 报告

## 板端信息

- IP: 10.126.35.143 (hostname: ax650)
- 芯片: AX650N_CHIP (NPU3, triple core)
- Engine: 2.12.0s, Compiler: Pulsar2 7.0 (6b1bcdf8)
- Python: 3.10.12 (torch 2.6.0+cpu, onnxruntime 1.21.0, axengine 0.1.3)

## 验证结果

### Python SDK (端到端中文合成)

| 后端 | 文本 | 帧数 | 时长 | 结果 |
|------|------|------|------|------|
| onnxruntime (CPU) | 你好，模思智能。 | 28 | 2.24s | wav 430KB ✅ |
| axengine (NPU3) | 你好，模思智能。 | 29 | 2.32s | wav 445KB ✅ |

provider=axengine 日志: ChipType.MC50, Model type 2 (triple core), Engine 2.12.0s

### C++ SDK (NPU codec 解码)

- 交叉编译: aarch64-none-linux-gnu 9.2.1 + 板端 AX 运行时库
- 运行: `moss_codec_cli --model model.axmodel --quantizer codec_quantizer_cpp.bin --codes ...`
- 输出: 245760 采样 (5.12s) wav 983KB ✅

### NPU LLM 加速 (方案 1: global decode 上 NPU3)

- 新增 4 个 AXMODEL: `llm_prefill` (89.5MB), `llm_decode` (89.1MB), `llm_local` (7.4MB), `llm_heads` (26.3MB)
- 板端实测: LLM 5.5 帧/秒 (纯 CPU 0.68, 约 8 倍), 单步 decode 41ms, 端到端 RTF ~2.43
- 真实中文推理校准 (calib_real): local hidden cosine 0.9984, 无采样漂移 (96 帧完整生成)
- 说明: 100M 自回归 TTS-LLM 每帧 18 步串行+采样, AX650 上实时下限 RTF~1.8-2.4;
  本包定位离线/异步 TTS; 板上强制 NPU 路径 (不回退 CPU), 早停自动换种子重试

### 性能

| 解码路径 | 每块耗时 (64帧=5.12s) | RTF |
|----------|----------------------|-----|
| NPU3 (axengine) | 582 ms | 0.11 |
| CPU (onnxruntime) | 1541 ms | 0.30 |

## 部署命令

```bash
scp -r models root@<板IP>:/opt/moss_tts/
scp -r python root@<板IP>:/opt/moss_tts/
ssh root@<板IP> 'cd /opt/moss_tts && LD_LIBRARY_PATH=/soc/lib python3 python/demo.py --provider axengine'
```
