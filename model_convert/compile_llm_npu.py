#!/usr/bin/env python3
"""Pulsar2 编译 LLM NPU 子图 (AX650/NPU3, FP32-mix + S16 LN)。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/data/yangrongzhao/Codes/Magnetar")

from magnetar.docker_util import docker_pulsar2  # noqa: E402

LAYER_CONFIGS = [
    {"op_type": "MatMul", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Conv", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "LayerNormalization", "data_type": "S16", "weight_data_type": "S16", "output_data_type": "S16"},
    {"op_type": "Mul", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Add", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Sub", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Div", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Where", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Erf", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
    {"op_type": "Softmax", "data_type": "FP32", "weight_data_type": "FP32", "output_data_type": "FP32"},
]


def build_config(model_name: str, input_shapes: str, input_names: list[str], calib_prefix: str, calib_size: int,
                 calib_dir: str = "calib_real"):
    return {
        "input": f"/workspace/export/llm_npu/{model_name}.onnx",
        "output_dir": f"/workspace/export/llm_npu",
        "output_name": f"{model_name}.axmodel",
        "work_dir": f"/workspace/export/llm_npu/work_{model_name}",
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": "NPU3",
        "input_shapes": input_shapes,
        "input_processors": [
            {"tensor_name": n, "tensor_format": "AutoColorSpace", "tensor_layout": "NCHW",
             "src_format": "AutoColorSpace", "src_layout": "NCHW", "src_dtype": "FP32"}
            for n in input_names
        ],
        "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": False, "model_check": True},
        "quant": {
            "input_configs": [
                {"tensor_name": n,
                 "calibration_dataset": f"/workspace/export/llm_npu/{calib_dir}/{calib_prefix}_{n}.tar.gz",
                 "calibration_format": "Numpy",
                 "calibration_size": calib_size,
                 "calibration_mean": [], "calibration_std": []}
                for n in input_names
            ],
            "calibration_method": "KL",
            "precision_analysis": False,
            "highest_mix_precision": False,
            "layer_configs": LAYER_CONFIGS,
        },
        "compiler": {"check": 0},
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = {
        "decode": build_config(
            "llm_decode",
            "inputs_embeds:1x1x768,mask:1x1x1x513,cos:1x1x1x64,sin:1x1x1x64," +
            ",".join(f"past_{i}:1x512x12x64" for i in range(24)),
            ["inputs_embeds", "mask", "cos", "sin"] + [f"past_{i}" for i in range(24)],
            "decode", 8,
        ),
        "local": build_config(
            "llm_local",
            "x:1x17x768,mask:1x1x17x17,cos:1x1x17x64,sin:1x1x17x64",
            ["x", "mask", "cos", "sin"], "local", 24,
        ),
        "heads": build_config(
            "llm_heads",
            "hidden:1x768",
            ["hidden"], "heads", 24,
        ),
        "prefill": build_config(
            "llm_prefill",
            "inputs_embeds:1x320x768,mask:1x1x320x320,cos:1x1x320x64,sin:1x1x320x64",
            ["inputs_embeds", "mask", "cos", "sin"], "prefill", 3,
        ),
    }
    order = ["decode", "local", "heads", "prefill"] if which == "all" else [which]
    for name in order:
        cfg = jobs[name]
        cfg_path = ROOT / "export" / "llm_npu" / f"{name}_config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[compile] {name} ...")
        log = docker_pulsar2(
            "pulsar2:7.0-lite", str(ROOT),
            f"pulsar2 build --config /workspace/export/llm_npu/{name}_config.json",
            timeout=7200,
        )
        (ROOT / "export" / "llm_npu" / f"{name}_compile.log").write_text(log, encoding="utf-8")
        ax = ROOT / "export" / "llm_npu" / f"{name}.axmodel"
        print(f"[compile] {name} done: {ax.stat().st_size / 1e6:.1f} MB" if ax.exists() else f"[compile] {name} FAILED")


if __name__ == "__main__":
    main()
