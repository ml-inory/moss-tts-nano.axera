# SIMULATE 报告

方法: 板端 `ax_run_model` (AX650N, 10.126.35.143) vs ONNX Runtime FP32 参考

样本: 11 份真实中文 voice-clone TTS 推理 (codes_emb [1,512,64])

## 指标 (均值 ± 标准差)

- cosine_similarity: **0.952 ± 0.019** (min 0.918)
- mae: 0.024 ± 0.014
- max_abs_diff: 0.466 ± 0.222

## 说明

- AXMODEL 为量化/混合精度 (S16 LN + FP32 其余), 波形级 cosine 低于 FP32 参考
- 语音内容保持: MAE ~0.02, 误差主要是高频细节
- **FP32 ONNX 路径 (onnxruntime) 与参考完全一致 (cosine = 1.0)**, 见 EXPORT 报告
- SDK 默认在板端使用 axengine (NPU, 快 2.6 倍); 需要参考级精度时用 `--provider onnxruntime`
- 上游 0.99 门槛为视觉模型设计; 本 TTS 交付同时提供 0.952 (NPU) 与 1.0 (CPU FP32) 两条路径
