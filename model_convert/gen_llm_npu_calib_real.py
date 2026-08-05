#!/usr/bin/env python3
"""从真实中文 TTS 生成中采集 LLM NPU 子图校准样本 (更准确的 LN 量化尺度)。"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import export_llm_npu2 as E  # noqa: E402  (torchaudio stub)
from repo.modeling_moss_tts_nano import MossTTSNanoForCausalLM  # noqa: E402
from repo.tokenization_moss_tts_nano import MossTTSNanoSentencePieceTokenizer  # noqa: E402

OUT = ROOT / "export" / "llm_npu" / "calib_real"
CTX = 512


def make_rope_np(pos_np, d=64):
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float32) / d))
    freqs = np.outer(np.asarray(pos_np, dtype=np.float32), inv_freq)
    cos = np.cos(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32)
    sin = np.sin(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32)
    return cos, sin


def main():
    torch.set_flush_denormal(True)
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

    cap = {
        "loc_x": [], "loc_mask": [], "loc_cos": [], "loc_sin": [], "heads_hidden": [],
        "loc_select": [],
        "dec_emb": [], "dec_mask": [], "dec_cos": [], "dec_sin": [], "dec_past": [],
        "pre_emb": [], "pre_mask": [], "pre_cos": [], "pre_sin": [],
    }

    orig_local = m._decode_local_last_hidden_state
    orig_trans = m.transformer.forward

    def capture_local(x):
        hl = orig_local(x)
        k = x.shape[1]
        if len(cap["loc_x"]) < 24:
            xp = np.zeros((1, 17, 768), np.float32)
            xp[0, :k, :] = x.detach().numpy()
            mask = np.tril(np.ones((1, 1, 17, 17), np.float32))
            mask[0, 0, k:, :] = 0
            mask[0, 0, :, k:] = 0
            c, s = make_rope_np(np.arange(17))
            cap["loc_x"].append(xp)
            cap["loc_mask"].append(mask)
            cap["loc_cos"].append(c)
            cap["loc_sin"].append(s)
            sel = np.zeros((1, 17, 1), np.float32)
            sel[0, k - 1, 0] = 1.0
            cap["loc_select"].append(sel)
            cap["heads_hidden"].append(hl.detach().numpy().astype(np.float32))
        return hl

    def capture_trans(*a, **k):
        r = orig_trans(*a, **k)
        past = k.get("past_key_values")
        emb = k.get("inputs_embeds")
        if past is not None and emb is not None and len(cap["dec_emb"]) < 8:
            s = int(past[0][0].shape[1])
            buf = [np.zeros((1, CTX, 12, 64), np.float32) for _ in range(24)]
            for i in range(24):
                buf[i][:, :s, :, :] = past[i // 2][i % 2].detach().numpy()
            mask = np.ones((1, 1, 1, CTX + 1), np.float32)
            mask[0, 0, 0, s:] = 0
            c, s_ = make_rope_np(np.array([s]))
            cap["dec_emb"].append(emb[:, -1:, :].detach().numpy().astype(np.float32))
            cap["dec_mask"].append(mask)
            cap["dec_cos"].append(c)
            cap["dec_sin"].append(s_)
            cap["dec_past"].append(buf)
        return r

    m._decode_local_last_hidden_state = capture_local
    m.transformer.forward = capture_trans

    texts = [
        "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。",
        "今天天气真好，我们一起去公园散步吧。",
        "人工智能正在改变我们生活的方方面面。",
        "夜深了，城市的灯一盏一盏慢慢安静下来，愿你有一个好梦。",
    ]
    for text in texts:
        inp, am = m.build_inference_input_ids(text=text, text_tokenizer=tok, mode="continuation")
        p_len = 320
        padded = torch.full((1, p_len, 17), m.config.audio_pad_token_id, dtype=torch.long)
        padded[:, :, 0] = m.config.pad_token_id
        padded[:, -inp.shape[1]:, :] = inp
        with torch.no_grad():
            pemb = m._build_inputs_embeds(padded)
            pmask = torch.tril(torch.ones((p_len, p_len))).unsqueeze(0).unsqueeze(0).float()
            pmask[0, 0, inp.shape[1]:, :] = 0
            pmask[0, 0, :, inp.shape[1]:] = 0
            c, s = make_rope_np(np.arange(p_len))
            cap["pre_emb"].append(pemb.numpy().astype(np.float32))
            cap["pre_mask"].append(pmask.numpy().astype(np.float32))
            cap["pre_cos"].append(c)
            cap["pre_sin"].append(s)
            gen = None
            for seed in (42, 43, 44, 45, 46):
                torch.manual_seed(seed)
                try:
                    with torch.no_grad():
                        gen = m.generate(input_ids=inp, attention_mask=am, max_new_frames=24, do_sample=True,
                                         audio_temperature=0.8, audio_top_p=0.95, audio_top_k=25,
                                         text_temperature=1.0, text_top_k=50, audio_repetition_penalty=1.2)
                    break
                except RuntimeError as exc:
                    if "Non-finite" not in str(exc):
                        raise
            if gen is None:
                raise RuntimeError("calib generation failed (NaN)")
        print(f"[calib_real] '{text[:14]}...' frames={gen.audio_token_ids.shape[1]} "
              f"loc={len(cap['loc_x'])} dec={len(cap['dec_emb'])}")
        if len(cap["loc_x"]) >= 24 and len(cap["dec_emb"]) >= 8 and len(cap["pre_emb"]) >= 3:
            break

    def save(arrs, name):
        with tarfile.open(OUT / f"{name}.tar.gz", "w:gz") as tar:
            for idx, arr in enumerate(arrs):
                np_path = OUT / f"{name}_{idx}.npy"
                np.save(np_path, arr)
                tar.add(np_path, arcname=f"{name}_{idx}.npy")

    naming = {
        "pre_emb": "prefill_inputs_embeds", "pre_mask": "prefill_mask",
        "pre_cos": "prefill_cos", "pre_sin": "prefill_sin",
        "loc_x": "local_x", "loc_mask": "local_mask", "loc_select": "local_select",
        "loc_cos": "local_cos", "loc_sin": "local_sin",
        "dec_emb": "decode_inputs_embeds", "dec_mask": "decode_mask",
        "dec_cos": "decode_cos", "dec_sin": "decode_sin",
    }
    for key, name in naming.items():
        save(cap[key], name)
    save(cap["heads_hidden"], "heads_hidden")
    for i in range(24):
        save([p[i] for p in cap["dec_past"]], f"decode_past_{i}")
    print(f"[calib_real] saved pre={len(cap['pre_emb'])} loc={len(cap['loc_x'])} "
          f"heads={len(cap['heads_hidden'])} dec={len(cap['dec_emb'])}")


if __name__ == "__main__":
    main()
