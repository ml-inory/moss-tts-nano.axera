"""MOSS-TTS-Nano-100M AX650 Python SDK.

架构:
- 100M 自回归 LLM: 板端 CPU (PyTorch, fp32, 原权重)
- 音频 codec 解码器: AXMODEL (pyaxengine.AxEngineExecutionProvider) 或 ONNX (onnxruntime)
- codebook 查表: CPU numpy (codec_quantizer.npz)
"""

from .runtime import MossTTSNano  # noqa: F401

__all__ = ["MossTTSNano"]
__version__ = "1.0.0"
