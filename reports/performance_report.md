# 性能报告

## 流水线耗时 (x86 开发机, 4 线程)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| LLM 生成 | ~12 帧/秒 | 100M 自回归, 12.5 帧/秒即实时 |
| LLM 生成 (NPU3 加速, 板端) | ~5.5 帧/秒 | global decode + local + heads 上 NPU3, 约 8x 加速 |
| LLM 生成 (纯 CPU, 板端) | 0.68 帧/秒 | 参考质量 |

## 端到端 RTF (AX650N 实测)

| 配置 | 端到端 RTF |
|------|-----------|
| 纯 CPU LLM + ONNX codec | ~18.8 |
| NPU3 LLM + NPU codec | **~2.4** |
| 仅 codec 解码 (NPU3) | 0.11-0.15 |

> 说明: 100M 自回归 TTS LLM 每帧需 18 步串行 (1 全局 + 17 local) 且步间采样,
> 该模型在 AX650 上的实时下限约为 RTF 1.8-2.4; RTF<1 (实时) 需更换非自回归/并行 TTS 架构。
| codec 解码 (ONNX, CPU x86) | 403 ms / 64帧(5.12s 音频) | RTF 0.079 |
| codec 解码 (ONNX, CPU 板端) | 1541 ms / 64帧 | RTF 0.30 |
| codec 解码 (NPU3, 板端 axengine) | 582 ms / 64帧 | RTF 0.11 |

## 模型效率

| 指标 | 值 |
|------|-----|
| ONNX 大小 | 50.2 MB |
| AXMODEL 大小 | 22.3 MB (FP32-mix/S16-LN) |
| 压缩比 | 2.25x |
| MACs (编译报告) | 171.97 G (64 帧 chunk) |
| 输出规格 | 48kHz / 双声道 / 16-bit PCM |

## 延迟估算 (AX650)

- 单帧 (80ms 音频) 生成: 目标 ≤ 12.5 帧/秒
- 5 秒中文句: LLM 约 0.5-1s (CPU) + NPU 解码约 100-300ms

## 板端内存

- AXMODEL 常驻 CMM: 见 RUNONBOARD 实测
- Python 运行时 (torch + 100M 模型): 约 0.8-1.2GB 系统内存
