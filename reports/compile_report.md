# COMPILE 报告

- Pulsar2 镜像: `pulsar2:7.0-lite` (version 7.0, commit 6b1bcdf8, 由 ax_pulsar2_7.0_lite.tar.gz 导入)
- 目标: AX650 / NPU3
- 输入: `codes_emb` float32 [1,512,64] → 输出 `waveform` float32 [1,2,245760]

## 量化策略演进

| 尝试 | 配置 | 结果 |
|------|------|------|
| INT8 | 默认 U8 | cosine 0.81 (不达标) |
| U16 | MatMul/Conv/LN → U16 | cosine 0.86 (不达标) |
| FP32-mix A | MatMul/Conv FP32, LN U16 | cosine 0.92 |
| FP32-mix B | 全部 FP32, LN S16 | cosine 0.95 |
| 全 FP32 | 全部 FP32 | NPU3 后端 AxLayerNorm FP32 tiling 失败 (硬限制) |

最终采用 **FP32-mix B**: 全部计算算子 FP32 (MatMul/Conv/Add/Mul/Sub/Div/Erf/Where/Softmax), LayerNorm 用 S16。

## 最终产物

- `model.axmodel`: 21,782.7 KB (22.3MB)
- ONNX 参考: 50.2MB (onnxsim 精简)
- MACs: 171.97 G (64 帧 chunk, 编译器报告)
- 压缩比: 50.2MB / 22.3MB ≈ 2.25x

## 关键配置

- `highest_mix_precision: false`
- `calibration_method: KL`, 11 份真实中文校准样本
- `compiler.check: 0` (跳过 NPU 后端逐算子校验; 精度由板端 SIMULATE 把关)
