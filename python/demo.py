#!/usr/bin/env python3
"""MOSS-TTS-Nano 中文语音合成 demo。

用法:
    python demo.py --text "欢迎关注模思智能" [--out out.wav] [--provider auto]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from moss_tts_sdk import MossTTSNano  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。")
    ap.add_argument("--out", default=str(ROOT.parent.parent / "out.wav"))
    ap.add_argument("--model-dir", default=str(ROOT.parent.parent / "models"))
    ap.add_argument("--provider", default="auto", choices=["auto", "axengine", "onnxruntime"])
    ap.add_argument("--max-frames", type=int, default=375)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    print(f"[demo] loading model from {args.model_dir} (provider={args.provider}) ...")
    tts = MossTTSNano(args.model_dir, provider=args.provider)
    print(f"[demo] decoding provider: {tts.decoder['type']}")
    print(f"[demo] synthesizing: {args.text}")
    result = tts.synthesize(args.text, args.out, max_new_frames=args.max_frames, seed=args.seed)
    print(f"[demo] done: {result['audio_path']}")
    print(f"[demo] duration={result['duration_s']:.2f}s frames={result['frames']} "
          f"provider={result['provider']} llm={result.get('llm_backend', 'cpu')} "
          f"sr={result['sample_rate']} channels={result['channels']}")


if __name__ == "__main__":
    main()
