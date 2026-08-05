#!/usr/bin/env python3
"""生成 decode 子图校准 (ctx=512), 手动前向采集, 带 NaN 重试。"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import export_llm_npu2 as E  # noqa: E402
from repo.modeling_moss_tts_nano import MossTTSNanoForCausalLM  # noqa: E402
from repo.tokenization_moss_tts_nano import MossTTSNanoSentencePieceTokenizer  # noqa: E402

OUT = ROOT / "export" / "llm_npu" / "calib_real"
CTX = 512


def make_rope_np(pos_np, d=64):
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float32) / d))
    freqs = np.outer(np.asarray(pos_np, dtype=np.float32), inv_freq)
    return (np.cos(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32),
            np.sin(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32))


def main():
    torch.set_flush_denormal(True)
    torch.set_num_threads(4)
    m = MossTTSNanoForCausalLM.from_pretrained(str(ROOT / "origin" / "repo"), trust_remote_code=True)
    m.eval()
    for name, mod in m.named_modules():
        if hasattr(mod, "attn_implementation"):
            mod.attn_implementation = "eager"

    def safe_eager(self, query, key, value, attention_mask):
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scale = 1.0
        if self.scale_attn_weights:
            scale /= self.head_dim ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)
        key_t = key.transpose(-1, -2)
        scores = torch.matmul(query, key_t) * scale
        causal_mask = self._causal_attention_mask(attention_mask, query.shape[-2], key.shape[-2], query.device)
        mf = causal_mask.to(dtype=scores.dtype)
        scores = scores * mf + (1.0 - mf) * (-1e9)
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)
        return output.transpose(1, 2).contiguous()

    for name, mod in m.named_modules():
        if hasattr(mod, "_eager_attention"):
            mod._eager_attention = safe_eager.__get__(mod, type(mod))

    tok = MossTTSNanoSentencePieceTokenizer(vocab_file=str(ROOT / "origin" / "repo" / "tokenizer.model"))
    OUT.mkdir(parents=True, exist_ok=True)

    samples = {"emb": [], "mask": [], "cos": [], "sin": [], "past": [[] for _ in range(24)]}
    for text in ("欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。",
                 "今天天气真好，我们一起去公园散步吧。"):
        inp, am = m.build_inference_input_ids(text=text, text_tokenizer=tok, mode="continuation")
        got = 0
        for attempt in range(6):
            try:
                with torch.no_grad():
                    emb0 = m._build_inputs_embeds(inp)
                    out = m.transformer(input_ids=None, past_key_values=None, attention_mask=am,
                                        position_ids=None, inputs_embeds=emb0, use_cache=True)
                kv = list(out.past_key_values)
                S = kv[0][0].shape[1]
                buffer = [np.zeros((1, CTX, 12, 64), np.float32) for _ in range(24)]
                for i in range(24):
                    buffer[i][:, :S, :, :] = kv[i // 2][i % 2].numpy()
                cur_am = am
                for step in range(8):
                    row = torch.full((1, 1, 17), m.config.audio_pad_token_id, dtype=torch.long)
                    row[0, 0, 0] = m.config.audio_start_token_id
                    row[0, 0, 1:] = torch.randint(0, 1024, (1, 16))
                    emb1 = m._build_inputs_embeds(row)
                    mask = np.ones((1, 1, 1, CTX + 1), np.float32)
                    mask[0, 0, 0, S + step:] = 0
                    c, s = make_rope_np(np.array([S + step]))
                    samples["emb"].append(emb1.detach().numpy().astype(np.float32))
                    samples["mask"].append(mask)
                    samples["cos"].append(c)
                    samples["sin"].append(s)
                    for i in range(24):
                        samples["past"][i].append(buffer[i].copy())
                    got += 1
                    with torch.no_grad():
                        out = m.transformer(input_ids=None, past_key_values=kv, attention_mask=cur_am,
                                            position_ids=None, inputs_embeds=emb1, use_cache=True)
                    kv = list(out.past_key_values)
                    for i in range(24):
                        buffer[i][:, S + step + 1, :, :] = kv[i // 2][i % 2].numpy()[:, -1, :, :]
                    cur_am = torch.cat([cur_am, torch.ones(1, 1, dtype=torch.bool)], dim=1)
                break
            except RuntimeError as exc:
                if "Non-finite" not in str(exc):
                    raise
                print(f"[calib] text NaN (attempt {attempt}), retry")
        print(f"[calib] '{text[:12]}...' captured {got} decode steps")
        if len(samples["emb"]) >= 8:
            break

    def save(arrs, name):
        with tarfile.open(OUT / f"{name}.tar.gz", "w:gz") as tar:
            for idx, arr in enumerate(arrs):
                np_path = OUT / f"{name}_{idx}.npy"
                np.save(np_path, arr)
                tar.add(np_path, arcname=f"{name}_{idx}.npy")

    save(samples["emb"], "decode_inputs_embeds")
    save(samples["mask"], "decode_mask")
    save(samples["cos"], "decode_cos")
    save(samples["sin"], "decode_sin")
    for i in range(24):
        save(samples["past"][i], f"decode_past_{i}")
    print(f"[calib] decode saved: {len(samples['emb'])} samples")


if __name__ == "__main__":
    main()
