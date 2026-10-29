"""Store an anima_ref_kv/1 adapter's LoRA matrices as FP8 (e4m3fn) with one float32 scale per row.

    python deploy/quantize_adapter_fp8.py IN.safetensors OUT.safetensors

Keys: <name>.lora_down / .lora_up become float8_e4m3fn, plus <name>.lora_down_scale / .lora_up_scale [rows, 1];
alpha, mode_embed and metadata are copied. The ComfyUI node (comfyui_nodes/snapref) dequantizes at load,
so compute stays bf16: this halves the file and RAM, it does not speed anything up.
"""
from __future__ import annotations

import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FP8_MAX = 448.0  # largest finite float8_e4m3fn


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return (w / scale).to(torch.float8_e4m3fn), scale


def main(src: str, dst: str) -> int:
    tensors = load_file(src)
    with safe_open(src, "pt") as handle:
        metadata = dict(handle.metadata() or {})
    out, worst = {}, 0.0
    for key, value in tensors.items():
        if key.endswith((".lora_down", ".lora_up")):
            q, scale = quantize(value)
            out[key], out[key + "_scale"] = q, scale
            back = q.float() * scale
            worst = max(worst, ((back - value.float()).norm() / value.float().norm().clamp_min(1e-12)).item())
        else:
            out[key] = value
    meta = json.loads(metadata.get("snapref") or metadata.get("anima_ref_kv") or "{}")
    meta["lora_dtype"] = "float8_e4m3fn, per-row float32 scale"
    metadata["anima_ref_kv"] = json.dumps(meta)
    save_file(out, dst, metadata=metadata)
    assert worst < 0.05, f"per-matrix relative error {worst:.4f} too large"
    print(json.dumps({"out": dst, "matrices": sum(k.endswith(("lora_down", "lora_up")) for k in out),
                      "worst_matrix_rel_err": round(worst, 5)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
