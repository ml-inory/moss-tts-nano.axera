"""MOSS-TTS-Nano-100M 推理运行时 (CPU LLM + NPU/ONNX codec decoder)。"""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import numpy as np

# torchaudio 仅在官方 modeling 的 inference() 路径使用; SDK 自实现音频 I/O, 用 stub 满足 import
if "torchaudio" not in sys.modules:
    _stub = types.ModuleType("torchaudio")

    def _stub_raise(*args, **kwargs):
        raise RuntimeError("torchaudio stub: SDK 使用 soundfile 处理音频")

    _stub.load = _stub_raise
    _stub.save = _stub_raise
    _stub.functional = types.SimpleNamespace(resample=_stub_raise)
    import importlib.machinery

    _stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = _stub

import torch  # noqa: E402


def _ensure_ax_runtime_path() -> None:
    """板端 AX 运行时库通常位于 /soc/lib; 提前注入 LD_LIBRARY_PATH 供 cffi 查找。"""
    soc_lib = "/soc/lib"
    if os.path.isdir(soc_lib):
        current = os.environ.get("LD_LIBRARY_PATH", "")
        entries = [e for e in current.split(":") if e]
        if soc_lib not in entries:
            os.environ["LD_LIBRARY_PATH"] = soc_lib + (":" + current if current else "")
        try:
            import ctypes

            for name in ("libax_engine.so", "libax_interpreter.so", "libax_sys.so"):
                path = os.path.join(soc_lib, name)
                if os.path.exists(path):
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass


_ensure_ax_runtime_path()


def _load_axengine_module():
    try:
        import axengine

        return axengine
    except ImportError:
        import pyaxengine  # noqa: F401

        return pyaxengine


class MossTTSNano:
    """中文优先的 MOSS-TTS-Nano 推理入口。

    参数:
        model_dir: 模型目录 (包含 llm/ 与 codec/)
        provider: "auto" | "axengine" | "onnxruntime"
        device: torch 设备 ("cpu")
    """

    def __init__(self, model_dir, provider: str = "auto", device: str = "cpu"):
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.provider = provider
        # 线程数过多时 oneDNN fp32 softmax 偶发 NaN; 固定少量线程保证数值稳定
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
        # 关键: fp32 denormal/FTZ 处理不一致会导致偶发 NaN (多核迁移时 MXCSR 状态漂移),
        # 强制 flush-to-zero 彻底消除
        try:
            torch.set_flush_denormal(True)
        except Exception:
            pass

        self.llm = self._load_llm(model_dir / "llm")
        self.tokenizer = self._load_tokenizer(model_dir / "llm")
        self.quantizer = np.load(model_dir / "codec" / "codec_quantizer.npz")
        self.decoder = self._load_decoder(model_dir / "codec", provider)
        self.reference_wav = model_dir / "codec" / "zh_1.wav"

    # ---------- 加载 ----------
    def _load_llm(self, llm_dir: Path):
        sys.path.insert(0, str(llm_dir))
        from repo.modeling_moss_tts_nano import MossTTSNanoForCausalLM

        model = MossTTSNanoForCausalLM.from_pretrained(str(llm_dir), trust_remote_code=True)
        model.eval()
        self._patch_safe_attention(model)
        return model

    @staticmethod
    def _patch_safe_attention(model):
        """eager attention 掩码用 -1e9, 规避 oneDNN fp32 多线程 softmax 偶发 NaN。"""
        for name, module in model.named_modules():
            if hasattr(module, "attn_implementation"):
                module.attn_implementation = "eager"

        def safe_eager(self, query, key, value, attention_mask):
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            scale = 1.0
            if self.scale_attn_weights:
                scale /= self.head_dim ** 0.5
            if self.scale_attn_by_inverse_layer_idx:
                scale /= float(self.layer_idx + 1)
            scores = torch.matmul(query, key.transpose(-1, -2)) * scale
            causal_mask = self._causal_attention_mask(
                attention_mask, query.shape[-2], key.shape[-2], query.device
            )
            scores = scores.masked_fill(~causal_mask, -1e9)
            probs = torch.softmax(scores, dim=-1)
            output = torch.matmul(probs, value)
            return output.transpose(1, 2).contiguous()

        for name, module in model.named_modules():
            if hasattr(module, "_eager_attention"):
                module._eager_attention = safe_eager.__get__(module, type(module))

    @staticmethod
    def _load_tokenizer(llm_dir: Path):
        sys.path.insert(0, str(llm_dir))
        from repo.tokenization_moss_tts_nano import MossTTSNanoSentencePieceTokenizer

        return MossTTSNanoSentencePieceTokenizer(vocab_file=str(llm_dir / "tokenizer.model"))

    def _load_decoder(self, codec_dir: Path, provider: str):
        axmodel = codec_dir / "codec_decoder.axmodel"
        onnx = codec_dir / "codec_decoder.onnx"
        resolved = provider
        if provider == "auto":
            resolved = "axengine" if axmodel.exists() and self._axengine_available() else "onnxruntime"
        if resolved == "axengine":
            if not axmodel.exists():
                raise FileNotFoundError(f"缺少 axmodel: {axmodel}")
            try:
                _load_axengine_module()
            except ImportError as exc:
                raise RuntimeError("provider=axengine 需要安装 axengine/pyaxengine (AXERA)") from exc
            return {
                "type": "axengine",
                "axmodel": str(axmodel),
                "session": None,  # 首次推理时创建
            }
        if not onnx.exists():
            raise FileNotFoundError(f"缺少 onnx: {onnx}")
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return {
            "type": "onnxruntime",
            "session": ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"]),
        }

    @staticmethod
    def _axengine_available() -> bool:
        try:
            _load_axengine_module()
            return True
        except Exception:
            return False

    # ---------- codebook 查表 (CPU) ----------
    def codes_to_emb(self, codes: np.ndarray) -> np.ndarray:
        """codes [16, T] int64 -> [1, 512, T] float32。"""
        nq, t = codes.shape
        emb = np.zeros((1, 512, t), dtype=np.float32)
        for i in range(nq):
            cb = self.quantizer[f"codebook_{i}"]
            w = self.quantizer[f"out_proj_w_{i}"]  # [8, 512]
            b = self.quantizer.get(f"out_proj_b_{i}")
            vec = cb[codes[i]]  # [T, 8]
            out = vec @ w  # [T, 8] @ [8, 512] -> [T, 512]
            if b is not None:
                out += b
            emb[0] += out.T
        return emb

    # ---------- 解码器 ----------
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """codes [16, T] -> waveform [2, T*3840] float32 (按 64 帧分块)。"""
        t = codes.shape[1]
        chunk_frames = 64
        n_chunks = (t + chunk_frames - 1) // chunk_frames
        wav_chunks = []
        for c in range(n_chunks):
            src = codes[:, c * chunk_frames : (c + 1) * chunk_frames]
            padded = np.zeros((16, chunk_frames), dtype=np.int64)
            padded[:, : src.shape[1]] = src
            wav = self._decode_chunk(padded)
            wav_chunks.append(wav[:, : src.shape[1] * 3840])
        return np.concatenate(wav_chunks, axis=-1)

    def _decode_chunk(self, codes: np.ndarray) -> np.ndarray:
        """codes [16, 64] -> waveform [2, 245760] float32。"""
        emb = self.codes_to_emb(codes)
        decoder = self.decoder
        if decoder["type"] == "onnxruntime":
            out = decoder["session"].run(None, {"codes_emb": emb})[0]
        else:
            if decoder["session"] is None:
                axengine = _load_axengine_module()
                decoder["session"] = axengine.InferenceSession(
                    decoder["axmodel"], providers=["AxEngineExecutionProvider"]
                )
            out = np.asarray(decoder["session"].run(["waveform"], {"codes_emb": emb})[0])
        return out[0]  # [2, 245760]

    # ---------- 参考音频 ----------
    def _encode_reference(self) -> np.ndarray:
        import soundfile as sf

        wav, sr = sf.read(str(self.reference_wav), dtype="float32", always_2d=True)
        if sr != 48000:
            x = torch.from_numpy(wav.T).unsqueeze(0)
            out_len = int(round(x.shape[-1] * 48000 / sr))
            x = torch.nn.functional.interpolate(x, size=out_len, mode="linear", align_corners=False)
            wav = x[0].T.numpy()
        if wav.shape[1] == 1:
            wav = np.repeat(wav, 2, axis=1)
        # 编码器也需要 codec 权重; 这里用预计算好的 prompt codes 文件
        prompt_path = self.model_dir / "codec" / "prompt_codes.npy"
        return np.load(str(prompt_path))

    # ---------- 合成 ----------
    def synthesize(
        self,
        text: str,
        output_path: str,
        max_new_frames: int = 375,
        seed: int | None = 42,
        do_sample: bool = True,
    ) -> dict:
        """中文语音合成 (voice_clone, 内置 zh_1.wav 参考音色)。"""
        prompt_codes = self._encode_reference()  # [T, 16]
        npu_llm = self._try_npu_llm()
        input_ids, attention_mask = self.llm.build_inference_input_ids(
            text=text,
            text_tokenizer=self.tokenizer,
            mode="voice_clone",
            prompt_audio_codes=torch.from_numpy(prompt_codes),
        )
        frames = 0
        generation = None
        if npu_llm is not None:
            # NPU LLM 强制路径 (不回退 CPU): 早停/溢出时换种子重试, 失败直接报错
            for attempt in range(3):
                with torch.no_grad():
                    generation = npu_llm.generate(
                        input_ids=input_ids, attention_mask=attention_mask,
                        max_new_frames=max_new_frames, do_sample=do_sample,
                        seed=(seed or 42) + attempt * 101,
                    )
                frames = int(generation.shape[1])
                if frames >= 5:
                    break
            frames = int(generation.shape[1])
            llm_backend = "npu3"
            if frames < 5:
                raise RuntimeError("NPU LLM 生成过早停止 (已重试 3 个种子), 请换文本或检查 AXMODEL")
        else:
            # 无 axengine (x86 开发机) 时才走 CPU 路径
            generation, frames = self._cpu_generate_with_retry(
                input_ids, attention_mask, max_new_frames, do_sample, seed
            )
            llm_backend = "cpu"
        if frames < 5:
            raise RuntimeError("生成过早停止, 请换文本或种子重试")

        gen_tensor = generation.audio_token_ids if hasattr(generation, "audio_token_ids") else generation
        codes = gen_tensor[0].transpose(0, 1).contiguous().numpy()  # [16, T]
        waveform = self.decode(codes)  # [2, N]
        waveform = np.clip(waveform, -1.0, 1.0)
        import soundfile as sf

        sf.write(str(output_path), waveform.T, 48000, subtype="PCM_16")
        return {
            "audio_path": str(output_path),
            "sample_rate": 48000,
            "channels": 2,
            "frames": frames,
            "duration_s": float(waveform.shape[1]) / 48000.0,
            "provider": self.decoder["type"],
            "llm_backend": llm_backend,
        }

    def _try_npu_llm(self):
        llm_npu_dir = self.model_dir / "llm_npu"
        decode_ax = llm_npu_dir / "llm_decode.axmodel"
        local_ax = llm_npu_dir / "llm_local.axmodel"
        if not (decode_ax.exists() and local_ax.exists()):
            return None
        if not self._axengine_available():
            return None
        try:
            from .npu_llm import NpuLlmRuntime

            return NpuLlmRuntime(self.model_dir, self.llm, self.tokenizer, self.llm.config)
        except Exception:
            return None

    def _cpu_generate_with_retry(self, input_ids, attention_mask, max_new_frames, do_sample, seed):
        # 配置兜底: 默认线程+oneDNN; 若连续失败, 关闭 oneDNN 并用单线程重试
        # (共享 x86 主机高负载下 oneDNN fp32 内核偶发 NaN; 板端 aarch64 不受影响)
        frames = 0
        generation = None
        for config in ("default", "no_mkldnn"):
            if config == "no_mkldnn":
                torch.backends.mkldnn.enabled = False
                torch.set_num_threads(1)
            for attempt in range(5):
                if seed is not None:
                    torch.manual_seed(seed + attempt * 101)
                try:
                    torch.set_flush_denormal(True)
                except Exception:
                    pass
                try:
                    with torch.no_grad():
                        generation = self.llm.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_frames=max_new_frames,
                            do_sample=do_sample,
                            audio_temperature=0.8,
                            audio_top_p=0.95,
                            audio_top_k=25,
                            text_temperature=1.0,
                            text_top_p=1.0,
                            text_top_k=50,
                            audio_repetition_penalty=1.2,
                        )
                except RuntimeError as exc:
                    if "Non-finite" not in str(exc):
                        raise
                    continue
                frames = int(generation.audio_token_ids.shape[1])
                if frames >= 5:
                    break
            if frames >= 5:
                break
        return generation, frames
