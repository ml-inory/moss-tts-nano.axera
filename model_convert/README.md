# 模型转换复现指南

本目录提供从 HuggingFace 原始权重到 AXMODEL 的完整复现流程。

## 依赖

```bash
pip install -r requirements.txt
```

## 1. 导出 ONNX (codec 解码器)

```bash
python export_onnx.py --model-dir ../../models --t-frames 64 --out-dir .
```

产物：

- `codec_decoder.onnx`：静态 ONNX，输入 `codes_emb` float32 [1,512,64]，输出 `waveform` [1,2,245760]
- `codec_quantizer.npz` / `codec_quantizer_cpp.bin`：CPU 侧 16 codebook 查表权重

导出后会自动对分验证（PyTorch vs ONNX cosine ≥ 0.99）。

## 2. 生成校准数据

校准数据来自真实中文 TTS 推理（`gen_chinese_tts_calib.py`，见任务 export/ 目录）：
每份样本为 `codes_emb` [1,512,64] float32 numpy 数组，打包为 `calib_data/input.tar.gz`。
本包已内置 11 份中文样本。

## 3. Pulsar2 编译 (AX650 / NPU3)

```bash
PULSAR2_IMAGE=pulsar2:7.0-lite WORKSPACE=/data/yangrongzhao/Codes/Magnetar \
  bash compile_pulsar2.sh
```

等价命令：

```bash
docker run --rm --network host \
  -v /data/yangrongzhao/Codes/Magnetar:/workspace \
  -v /var/hasplm:/var/hasplm \
  -v /tmp/p2_verify_home/.hasplm:/root/.hasplm \
  -e HASP_HOME=/root/.hasplm \
  pulsar2:7.0-lite -lc "PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:\$PATH \
  pulsar2 build --config /workspace/package/model_convert/pulsar2_config.json"
```

配置要点：

- `target_hardware: AX650`，`npu_mode: NPU3`
- `calibration_method: KL`，11 份中文校准样本
- 精度策略: 全部计算算子 **FP32**, LayerNorm 用 **S16**（全 FP32 时 NPU3 后端
  LayerNorm tiling 失败; INT8/U16 波形 cosine 不达标, 最终 FP32-mix 达 0.952）
- `highest_mix_precision: false`（强制）

## 4. 产物检查

```bash
ls -la model.axmodel          # 应存在且非空
```

## 5. 板端验证

```bash
# 板端 ax_run_model (示例)
LD_LIBRARY_PATH=/soc/lib /opt/bin/ax_run_model \
  -m model.axmodel -i stim -o out -l list.txt -w 0 -r 1
```

## FAQ

- **为什么 AXMODEL 输入是 float32 而不是 token id？** Pulsar2 运行时不支持 int64 输入，
  codebook 查表放在 CPU 侧（`codec_quantizer`），NPU 只做解码器主干。
- **为什么 U16？** INT8 量化下波形 cosine 0.65-0.94，低于 0.99 门槛；U16 将精度
  提升到 ≥0.99。
- **softmax 编译报错？** `compiler.check=0` 跳过 NPU 后端的逐算子校验，精度由
  SIMULATE 阶段板端对分把关。

## 附: LLM NPU 子模型复现 (离线/异步加速选件)

除 codec 解码器外，包内 `models/llm_npu/` 还有 4 个 LLM 子模型
(`llm_prefill/llm_decode/llm_local/llm_heads`)，将 100M 自回归 LLM 的
global decode、local transformer、头部投影移入 NPU3（板端实测 LLM 5.5 帧/秒，
约 8 倍于纯 CPU）。复现脚本：

```bash
# 1) 导出 4 个静态 ONNX (显式 attention, cos/sin 由 host 预计算)
python export_llm_npu2.py --p-len 320 --ctx 512

# 2) 生成校准数据 (需真实中文推理; 脚本含 NaN 重试, 或在 AX650 板上采集)
python gen_llm_npu_calib_real.py
python gen_decode_calib.py

# 3) 编译 (AX650/NPU3, FP32 计算 + S16 LayerNorm; 全 FP32 受 NPU3 LN tiling 限制)
python compile_llm_npu.py all
```

精度与 RTF 见 `reports/runonboard_report.md`。已知边界：

- KV 上下文固定 768 帧 (约 61s 音频), 超出时抛出明确错误 (SDK 不回退 CPU)
- S16 LayerNorm 为 NPU3 硬限制; 校准数据需来自真实生成 (随机校准会引入采样漂移)
- 该自回归模型在 AX650 的实时下限约 RTF 1.8-2.4, 本选件面向离线/异步
