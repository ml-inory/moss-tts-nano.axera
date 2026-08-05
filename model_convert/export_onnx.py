#!/usr/bin/env python3
"""可复现导出: MOSS-Audio-Tokenizer-Nano 解码器 -> 静态 ONNX (FP32 输入版)。

用法:
    python export_onnx.py --model-dir ../../models --t-frames 64

产物:
    codec_decoder.onnx      静态 ONNX, 输入 codes_emb[1,512,T], 输出 waveform[1,2,T*3840]
    codec_quantizer.npz     CPU 侧 codebook 查表权重 (SDK/Python 用)
    codec_quantizer_cpp.bin C++ 侧查表权重
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def build_export_code(audio_tokenizer_dir: Path, t_frames: int):
    """加载 HF 仓库代码 (trust_remote_code), 返回 (wrapper, quantizer, model)。"""
    sys.path.insert(0, str(audio_tokenizer_dir))
    from modeling_moss_audio_tokenizer import (  # noqa
        MossAudioTokenizerMultiheadAttention,
        MossAudioTokenizerModel,
    )

    class EagerAttentionMHA(MossAudioTokenizerMultiheadAttention):
        def _forward_non_streaming_sdpa(self, x, input_lengths):
            batch_size, max_seqlen, _ = x.shape
            q, k, v = self._project_qkv(x)
            q, k = self._apply_dense_rope(q, k)
            head_dim = self.embed_dim // self.num_heads
            scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
            positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
            delta = positions.view(1, max_seqlen, 1) - positions.view(1, 1, max_seqlen)
            attn_bias = torch.ones((1, max_seqlen, max_seqlen), device=x.device, dtype=torch.bool)
            if self.causal:
                attn_bias = attn_bias & (delta >= 0)
            if self.context is not None:
                attn_bias = attn_bias & (delta < self.context)
            valid_k = positions.view(1, 1, max_seqlen) < input_lengths.view(-1, 1, 1)
            attn_bias = attn_bias & valid_k
            scores = scores.masked_fill(~attn_bias, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out.transpose(1, 2).reshape(batch_size, max_seqlen, self.embed_dim)

    model = MossAudioTokenizerModel.from_pretrained(str(audio_tokenizer_dir), trust_remote_code=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, MossAudioTokenizerMultiheadAttention):
            module.__class__ = EagerAttentionMHA
            module.attention_implementation = "sdpa"

    class DecoderForExport(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.output_proj = m.quantizer.output_proj
            self.decoder = m.decoder
            self.n_channels = m.number_channels
            self.interleave = m.enable_channel_interleave

        def forward(self, codes_emb):
            batch, _, time = codes_emb.shape
            audio = self.output_proj(codes_emb.float())
            lengths = torch.full((batch,), time, dtype=torch.long, device=codes_emb.device)
            for module in self.decoder:
                audio, lengths = module(audio, lengths)
            if self.n_channels > 1 and self.interleave:
                audio = (
                    audio.squeeze(1)
                    .contiguous()
                    .view(audio.shape[0], -1, self.n_channels)
                    .transpose(1, 2)
                    .contiguous()
                    .float()
                )
            else:
                audio = audio.float()
            return audio

    return DecoderForExport(model), model.quantizer, model


def save_quantizer_weights(quantizer, npz_path: Path, cpp_path: Path):
    data = {}
    for i, q in enumerate(quantizer.quantizers):
        data[f"codebook_{i}"] = q.codebook.weight.detach().cpu().numpy().astype(np.float32)
        w = q.out_proj.weight.detach().cpu().numpy().astype(np.float32)  # [512,8,1]
        data[f"out_proj_w_{i}"] = w.transpose(1, 0, 2)[:, :, 0]  # [8,512] = W^T
        data[f"out_proj_b_{i}"] = q.out_proj.bias.detach().cpu().numpy().astype(np.float32)
    w = quantizer.output_proj.weight.detach().cpu().numpy().astype(np.float32)
    data["output_proj_w"] = w.transpose(1, 0, 2)[:, :, 0]  # [512,768] = W^T
    data["output_proj_b"] = quantizer.output_proj.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(str(npz_path), **data)

    blob = b"".join(
        [data[f"codebook_{i}"].astype("<f4").tobytes()
         + data[f"out_proj_w_{i}"].astype("<f4").tobytes()
         + data[f"out_proj_b_{i}"].astype("<f4").tobytes() for i in range(16)]
        + [data["output_proj_w"].astype("<f4").tobytes(),
           data["output_proj_b"].astype("<f4").tobytes()]
    )
    cpp_path.write_bytes(blob)


def validate(wrapper, quantizer, onnx_path, t_frames, nq=16, trials=4):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cosines = []
    for trial in range(trials):
        codes = np.random.randint(0, 1024, (nq, 1, t_frames), dtype=np.int64)
        with torch.no_grad():
            emb = torch.zeros(1, quantizer.rvq_dim, t_frames)
            for i in range(nq):
                emb += quantizer.quantizers[i].decode_code(torch.from_numpy(codes[i])).float()
            ref = wrapper(emb).numpy()
        out = sess.run(None, {"codes_emb": emb.numpy().astype(np.float32)})[0]
        a, b = ref.reshape(-1).astype(np.float64), out.reshape(-1).astype(np.float64)
        cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return min(cosines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("../../models"))
    ap.add_argument("--t-frames", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    audio_tokenizer_dir = args.model_dir / "codec"
    # 说明: 该目录需包含 HF MOSS-Audio-Tokenizer-Nano 的 modeling/configuration 代码
    # (模型包里已内置在 models/codec/ 下的子目录)
    code_dir = audio_tokenizer_dir / "hf_code"
    wrapper, quantizer, model = build_export_code(code_dir, args.t_frames)
    dummy = torch.zeros((1, 512, args.t_frames), dtype=torch.float32)
    onnx_path = args.out_dir / "codec_decoder.onnx"
    torch.onnx.export(wrapper, (dummy,), str(onnx_path), input_names=["codes_emb"],
                      output_names=["waveform"], opset_version=17)
    cos = validate(wrapper, quantizer, onnx_path, args.t_frames)
    print(f"[export] codec_decoder.onnx done, cosine={cos:.6f}")
    assert cos > 0.99
    save_quantizer_weights(quantizer, args.out_dir / "codec_quantizer.npz",
                           args.out_dir / "codec_quantizer_cpp.bin")
    print("[export] quantizer weights saved")


if __name__ == "__main__":
    main()
