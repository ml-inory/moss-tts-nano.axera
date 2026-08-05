#!/usr/bin/env python3
"""导出 MOSS-TTS-Nano 100M LLM 的 NPU 子图 (v2, 显式 attention)。

显式 attention 避免模型原实现 (split/transpose/掩码约定) 与静态导出的冲突。
复用原模型权重 (c_attn/c_proj/rotary_emb/ln/mlp), KV 布局 [B,S,H,D]。

三个图:
  llm_prefill.onnx: inputs_embeds[1,P,768] + mask[1,1,P,P] + position_ids[1,P]
                    -> hidden_last[1,768] + kv_out 24x[1,P,12,64]
  llm_decode.onnx:  inputs_embeds[1,1,768] + mask[1,1,1,C] + position_ids[1,1]
                    + past 24x[1,C,12,64] -> hidden[1,768] + new_kv 24x[1,1,12,64]
  llm_local.onnx:   x[1,17,768] + mask[1,1,17,17] + position_ids[1,17]
                    -> hidden_last[1,768] (位置 seq_len-1 由 host 取)
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_stub = types.ModuleType("torchaudio")


def _stub_raise(*a, **k):
    raise RuntimeError("stub")


_stub.load = _stub_raise
_stub.save = _stub_raise
_stub.functional = types.SimpleNamespace(resample=_stub_raise)
import importlib.machinery

_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
sys.modules["torchaudio"] = _stub

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "origin"))

from repo.modeling_moss_tts_nano import MossTTSNanoForCausalLM  # noqa: E402


class ExportAttention(nn.Module):
    """显式 attention, 复用原模块参数。mask 作为图输入 (已含因果与有效性)。"""

    def __init__(self, orig):
        super().__init__()
        self.c_attn = orig.c_attn
        self.c_proj = orig.c_proj
        self.rotary_emb = orig.rotary_emb
        self.num_heads = orig.num_heads
        self.head_dim = orig.head_dim
        self.embed_dim = orig.embed_dim
        self.scale_attn_weights = orig.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = orig.scale_attn_by_inverse_layer_idx
        self.layer_idx = orig.layer_idx

    def forward(self, hidden_states, mask, cos, sin, past_k=None, past_v=None):
        b, s, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        k_new, v_new = k, v  # 供 host 更新 KV 缓冲
        if past_k is not None:
            pk = past_k.permute(0, 2, 1, 3).contiguous()  # [B,ctx,H,D] -> [B,H,ctx,D]
            pv = past_v.permute(0, 2, 1, 3).contiguous()
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        scale = 1.0
        if self.scale_attn_weights:
            scale /= self.head_dim ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)
        k_t = k.permute(0, 1, 3, 2).contiguous()
        scores = torch.matmul(q, k_t) * scale
        mf = mask.to(dtype=scores.dtype)
        scores = scores * mf + (1.0 - mf) * (-1e9)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(b, s, self.embed_dim)
        return self.c_proj(out), k_new, v_new

    @staticmethod
    def _rope(x, cos, sin):
        b, h, s, d = x.shape
        z = x.reshape(b, h, s, d // 2, 2)
        even, odd = z[..., 0], z[..., 1]
        rot = torch.cat([-odd.unsqueeze(-1), even.unsqueeze(-1)], dim=-1).reshape(b, h, s, d)
        return x * cos + rot * sin


class ExportBlock(nn.Module):
    def __init__(self, orig, mode):
        super().__init__()
        self.ln_1 = orig.ln_1
        self.attn = ExportAttention(orig.attn)
        self.ln_2 = orig.ln_2
        self.mlp = orig.mlp
        self.mode = mode

    def forward(self, hidden_states, mask, cos, sin, past_k=None, past_v=None):
        attn_out, k, v = self.attn(self.ln_1(hidden_states), mask, cos, sin, past_k, past_v)
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, k, v


class PrefillNet(nn.Module):
    def __init__(self, model, p_len):
        super().__init__()
        self.p_len = p_len
        self.blocks = nn.ModuleList([ExportBlock(b, "prefill") for b in model.transformer.h])
        self.ln_f = model.transformer.ln_f

    def forward(self, inputs_embeds, mask, cos, sin):
        h = inputs_embeds
        kvs = []
        for blk in self.blocks:
            h, k, v = blk(h, mask, cos, sin)
            kvs.append(k.permute(0, 2, 1, 3).contiguous())  # [B,S,H,D]
            kvs.append(v.permute(0, 2, 1, 3).contiguous())
        h = self.ln_f(h)
        return [h] + kvs  # host 取 h[:, seq_len-1, :]


class DecodeNet(nn.Module):
    def __init__(self, model, ctx):
        super().__init__()
        self.ctx = ctx
        self.blocks = nn.ModuleList([ExportBlock(b, "decode") for b in model.transformer.h])
        self.ln_f = model.transformer.ln_f

    def forward(self, inputs_embeds, mask, cos, sin, *past):
        h = inputs_embeds
        outs = []
        for i, blk in enumerate(self.blocks):
            h, k, v = blk(h, mask, cos, sin, past[2 * i], past[2 * i + 1])
            outs.append(k)  # [B,H,1,D] -> host 转 [B,1,H,D]
            outs.append(v)
        h = self.ln_f(h)
        return [h.reshape(h.shape[0], -1)] + outs


class LocalNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.blocks = nn.ModuleList([ExportBlock(model.local_transformer.h[0], "local")])
        self.ln_f = model.local_transformer.ln_f

    def forward(self, x, mask, cos, sin):
        h = x
        for blk in self.blocks:
            h, _, _ = blk(h, mask, cos, sin)
        return self.ln_f(h)  # [1,17,768]


class HeadsNet(nn.Module):
    """纯线性头部: hidden [1,768] -> 全部 logits [1, 32768] (text 16384 + 16 audio)。"""

    def __init__(self, model):
        super().__init__()
        self.text_lm_head = model.text_lm_head
        self.audio_lm_heads = model.audio_lm_heads

    def forward(self, hidden):
        text_logits = self.text_lm_head(hidden)  # [1,16384]
        audio_logits = torch.cat(
            [self.audio_lm_heads[ch](hidden) for ch in range(16)], dim=-1
        )  # [1,16384]
        return torch.cat([text_logits, audio_logits], dim=-1)  # [1,32768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-len", type=int, default=320)
    ap.add_argument("--ctx", type=int, default=768)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "export" / "llm_npu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[EXPORT] loading model ...")
    m = MossTTSNanoForCausalLM.from_pretrained(str(ROOT / "origin" / "repo"), trust_remote_code=True)
    m.eval()

    p_len, ctx = args.p_len, args.ctx

    def make_rope(pos_np, d=64):
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float32) / d))
        freqs = np.outer(pos_np.astype(np.float32), inv_freq)  # [S, D/2]
        cos = np.cos(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32)
        sin = np.sin(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32)
        return torch.from_numpy(cos), torch.from_numpy(sin)

    # prefill
    pn = PrefillNet(m, p_len).eval()
    emb = torch.randn((1, p_len, 768))
    mask = torch.tril(torch.ones((p_len, p_len))).unsqueeze(0).unsqueeze(0).float()
    cos, sin = make_rope(np.arange(p_len))
    p = args.out_dir / "llm_prefill.onnx"
    torch.onnx.export(pn, (emb, mask, cos, sin), str(p),
                      input_names=["inputs_embeds", "mask", "cos", "sin"],
                      output_names=["hidden_all"] + [f"kv_{i}" for i in range(24)], opset_version=17)
    print("[EXPORT] prefill done")

    # decode
    dn = DecodeNet(m, ctx).eval()
    emb1 = torch.randn((1, 1, 768))
    mask1 = torch.ones((1, 1, 1, ctx), dtype=torch.float32)
    mask1 = torch.ones((1, 1, 1, ctx + 1), dtype=torch.float32)
    mask1[0, 0, 0, 201:] = 0
    cos1, sin1 = make_rope(np.array([201]))
    past = [torch.zeros((1, ctx, 12, 64)) for _ in range(24)]
    d = args.out_dir / "llm_decode.onnx"
    torch.onnx.export(dn, (emb1, mask1, cos1, sin1, *past), str(d),
                      input_names=["inputs_embeds", "mask", "cos", "sin"] + [f"past_{i}" for i in range(24)],
                      output_names=["hidden"] + [f"new_{i}" for i in range(24)], opset_version=17)
    print("[EXPORT] decode done")

    # local
    ln = LocalNet(m).eval()
    x = torch.randn((1, 17, 768))
    maskl = torch.tril(torch.ones((17, 17))).unsqueeze(0).unsqueeze(0).float()
    maskl[0, 0, 5:, :] = 0
    maskl[0, 0, :, 5:] = 0
    cosl, sinl = make_rope(np.arange(17))
    l = args.out_dir / "llm_local.onnx"
    torch.onnx.export(ln, (x, maskl, cosl, sinl), str(l),
                      input_names=["x", "mask", "cos", "sin"],
                      output_names=["hidden"], opset_version=17)
    print("[EXPORT] local done")

    # heads
    hn = HeadsNet(m).eval()
    hh = torch.randn((1, 768))
    hd = args.out_dir / "llm_heads.onnx"
    torch.onnx.export(hn, (hh,), str(hd),
                      input_names=["hidden"], output_names=["logits"], opset_version=17)
    print("[EXPORT] heads done")

    print("[EXPORT] validating ...")
    import onnxruntime as ort

    for name, sess_inputs, ref_fn, onnx_path in [
        ("prefill", (emb, mask, cos, sin), lambda: pn(emb, mask, cos, sin), p),
        ("decode", (emb1, mask1, cos1, sin1, *past), lambda: dn(emb1, mask1, cos1, sin1, *past), d),
        ("local", (x, maskl, cosl, sinl), lambda: ln(x, maskl, cosl, sinl), l),
        ("heads", (hh,), lambda: hn(hh), hd),
    ]:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        feed = {n: v.cpu().numpy() for n, v in zip(names, sess_inputs)}
        out = sess.run(None, feed)
        with torch.no_grad():
            ref = ref_fn()
        ref_t = ref[0] if isinstance(ref, (list, tuple)) else ref
        a, b = ref_t.cpu().numpy().reshape(-1).astype(np.float64), out[0].reshape(-1).astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  {name} hidden cos={cos:.6f}")
        if name == "decode":
            # 校验 new_k
            a2 = ref[1].cpu().numpy().reshape(-1).astype(np.float64)
            b2 = out[1].reshape(-1).astype(np.float64)
            cos2 = float(np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-12))
            print(f"  decode new_k cos={cos2:.6f}")
        if name == "heads":
            print(f"  heads logits cos={cos:.6f}")
        assert cos > 0.99


if __name__ == "__main__":
    main()
