"""NPU 加速版 LLM 推理: prefill 在 CPU (torch), 逐帧 decode/local 在 NPU3。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

_CTX = 768
_HEAD_DIM = 64
_NUM_HEADS = 12
_HIDDEN = 768
_MAX_LOCAL = 17


def _make_rope_np(positions, d=_HEAD_DIM):
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float32) / d))
    freqs = np.outer(np.asarray(positions, dtype=np.float32), inv_freq)
    return (np.cos(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32),
            np.sin(freqs).repeat(2, axis=-1)[None, None, :, :].astype(np.float32))


class NpuLlmRuntime:
    """加载 llm_decode/llm_local AXMODEL, 用 NPU 加速逐帧推理。

    与 CPU 版 generate 完全对齐的采样/模板逻辑 (top-k/top-p/temperature/penalty),
    仅把 transformer 计算移到 NPU3。
    """

    def __init__(self, model_dir: Path, llm, tokenizer, config):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config
        try:
            from .runtime import _ensure_ax_runtime_path, _load_axengine_module

            _ensure_ax_runtime_path()
            ax = _load_axengine_module()
            self.decode_sess = ax.InferenceSession(
                str(model_dir / "llm_npu" / "llm_decode.axmodel"),
                providers=["AxEngineExecutionProvider"],
            )
            self.local_sess = ax.InferenceSession(
                str(model_dir / "llm_npu" / "llm_local.axmodel"),
                providers=["AxEngineExecutionProvider"],
            )
            self.heads_sess = ax.InferenceSession(
                str(model_dir / "llm_npu" / "llm_heads.axmodel"),
                providers=["AxEngineExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(f"NPU LLM 运行时加载失败: {exc}") from exc
        self._kv = None
        self._past_len = 0
        self.local_on_cpu = False  # True 时 local transformer 回退 CPU (保质量)
        self._local_cos, self._local_sin = _make_rope_np(np.arange(_MAX_LOCAL))
        self._x_buf = np.zeros((1, _MAX_LOCAL, _HIDDEN), np.float32)
        self._masks = {}
        for k in range(1, _MAX_LOCAL + 1):
            mask = np.tril(np.ones((1, 1, _MAX_LOCAL, _MAX_LOCAL), np.float32))
            mask[0, 0, k:, :] = 0
            mask[0, 0, :, k:] = 0
            self._masks[k] = mask

    # ---------- prefill ----------
    def _prefill(self, input_ids, attention_mask):
        with torch.no_grad():
            emb = self.llm._build_inputs_embeds(input_ids)
            out = self.llm.transformer(
                input_ids=None, past_key_values=None, attention_mask=attention_mask,
                position_ids=None, inputs_embeds=emb, use_cache=True,
            )
        self._kv = [np.zeros((1, _CTX, _NUM_HEADS, _HEAD_DIM), np.float32) for _ in range(24)]
        s = int(out.past_key_values[0][0].shape[1])
        for i in range(24):
            self._kv[i][:, :s, :, :] = out.past_key_values[i // 2][i % 2].numpy()
        self._past_len = s
        return out.last_hidden_state[:, -1, :]  # [1,768] 用于首帧 local

    # ---------- 逐帧 ----------
    def _decode_step(self, inputs_embeds, position):
        if self._past_len >= _CTX:
            raise RuntimeError("NPU_CTX_OVERFLOW")
        mask = np.zeros((1, 1, 1, _CTX + 1), np.float32)
        mask[0, 0, 0, : self._past_len] = 1.0
        cos, sin = _make_rope_np([position])
        feed = {
            "inputs_embeds": np.asarray(inputs_embeds, np.float32),
            "mask": mask,
            "cos": cos,
            "sin": sin,
        }
        for i in range(24):
            feed[f"past_{i}"] = self._kv[i]
        out = self.decode_sess.run(None, feed)
        hidden = out[0]  # [1,768]
        for i in range(24):
            nk = np.asarray(out[1 + i], np.float32).reshape(1, 1, _NUM_HEADS, _HEAD_DIM)
            self._kv[i][:, self._past_len, :, :] = nk[:, 0, :, :]
        self._past_len += 1
        return hidden

    def _local_step(self, x, seq_len, position_ids):
        if self.local_on_cpu:
            with torch.no_grad():
                h = self.llm._decode_local_last_hidden_state(
                    torch.from_numpy(np.asarray(x[:, :seq_len], np.float32))
                )
            return h.numpy()  # [1,768]
        mask = self._masks[seq_len]
        out = self.local_sess.run(None, {"x": np.asarray(x, np.float32), "mask": mask,
                                         "cos": self._local_cos, "sin": self._local_sin})
        return out[0][:, seq_len - 1, :]  # [1,768]

    def _heads(self, hidden):
        out = self.heads_sess.run(None, {"hidden": np.asarray(hidden, np.float32).reshape(1, -1)})
        return torch.from_numpy(out[0])  # [1,32768]

    def generate(self, input_ids, attention_mask, max_new_frames, do_sample=True,
                 text_temperature=1.0, text_top_p=1.0, text_top_k=50,
                 audio_temperature=0.8, audio_top_p=0.95, audio_top_k=25,
                 audio_repetition_penalty=1.2, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        global_hidden = self._prefill(input_ids, attention_mask)
        generated = []

        for step in range(max_new_frames):
            # 文本 token (slot/end): local 序列 = [global_hidden]
            local_embs = [np.asarray(global_hidden, np.float32).reshape(-1)]
            hidden_local = self._local_forward(local_embs)
            logits_all = self._heads(hidden_local)
            text_logits = logits_all[:, :16384]
            next_text_idx = self._sample_next_token(
                text_logits[:, [self.config.audio_assistant_slot_token_id, self.config.audio_end_token_id]],
                do_sample, text_temperature, text_top_k, text_top_p,
            )
            next_text = torch.tensor(
                [self.config.audio_assistant_slot_token_id, self.config.audio_end_token_id])[next_text_idx]
            if next_text.eq(self.config.audio_end_token_id).all():
                break

            frame_codes = []
            local_embs.append(self.llm.transformer.wte(next_text).float().numpy().reshape(-1))
            for ch in range(16):
                hidden_local = self._local_forward(local_embs)
                logits_all = self._heads(hidden_local)
                logits = logits_all[:, 16384 + ch * 1024 : 16384 + (ch + 1) * 1024]
                token = self._sample_next_token(
                    logits, do_sample, audio_temperature, audio_top_k, audio_top_p,
                    previous=torch.tensor([g[ch] for g in generated]) if generated else None,
                    repetition_penalty=audio_repetition_penalty,
                )
                frame_codes.append(int(token.item()))
                local_embs.append(self.llm.audio_embeddings[ch](
                    torch.tensor([token.item()])).float().numpy().reshape(-1))

            generated.append(frame_codes)
            # 下一帧 global decode
            row = torch.full((1, 1, 17), self.config.audio_pad_token_id, dtype=torch.long)
            row[0, 0, 0] = self.config.audio_assistant_slot_token_id
            row[0, 0, 1:] = torch.tensor(frame_codes)
            emb1 = self.llm._build_inputs_embeds(row)
            hidden = self._decode_step(emb1.numpy(), self._past_len)
            global_hidden = hidden

        return torch.tensor(generated, dtype=torch.long).unsqueeze(0) if generated \
            else torch.empty((1, 0, 16), dtype=torch.long)

    def _local_forward(self, seq_embs):
        """local transformer 前向: seq_embs 为 [embed...], 取最后有效 hidden。"""
        k = len(seq_embs)
        x = self._x_buf
        for j, e in enumerate(seq_embs):
            x[0, j, :] = e
        out = self._local_step(x, k, list(range(k)))
        return self._local_step(x, k, list(range(k)))  # [1,768]

    @staticmethod
    def _sample_next_token(logits, do_sample, temperature, top_k, top_p,
                           previous=None, repetition_penalty=1.0):
        if not do_sample:
            return torch.tensor([int(np.argmax(logits.detach().numpy()))])
        scores = logits.detach().float().numpy().reshape(-1).astype(np.float64)
        if previous is not None and repetition_penalty != 1.0:
            for t in previous.detach().numpy().reshape(-1):
                t = int(t)
                if 0 <= t < len(scores):
                    scores[t] = scores[t] * repetition_penalty if scores[t] < 0 \
                        else scores[t] / repetition_penalty
        scores = scores / temperature
        if top_k is not None and top_k > 0:
            k = min(int(top_k), len(scores))
            keep_idx = np.argpartition(scores, len(scores) - k)[-k:]
            keep = np.zeros_like(scores, dtype=bool)
            keep[keep_idx] = True
            scores[~keep] = -np.inf
        if top_p is not None and 0.0 < top_p < 1.0:
            order = np.argsort(-scores)
            sorted_scores = scores[order]
            m = sorted_scores.max()
            exp = np.exp(sorted_scores - m)
            probs_sorted = exp / exp.sum()
            cum = np.cumsum(probs_sorted)
            remove = cum > top_p
            remove[1:] = remove[:-1].copy()
            remove[0] = False
            sorted_scores[remove] = -np.inf
            scores[order] = sorted_scores
        m = scores.max()
        probs = np.exp(scores - m)
        probs = probs / probs.sum()
        return torch.tensor([int(np.random.choice(len(probs), p=probs))])
