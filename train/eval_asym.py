"""Held-out loss gap for the asymmetric adapter (the original author's metric, fixed noise and sigmas).

    CUDA_VISIBLE_DEVICES=0 python train/eval_asym.py --prep ~/train/asym_prep --ours runs/v1a/adapter_002000.safetensors \
        --out ~/train/compare/v1a_002000

For every held-out group (characters never trained on): target = a body-pose image when there is one,
reference = another image of the group. At sigmas 0.15 / 0.35 / 0.6 / 0.85 with fixed noise, MSE of the
predicted velocity for: base (no adapter), ours with the right reference, ours with a wrong reference
(another held-out character). identity gap = wrong - right (higher is better: the adapter uses the reference).
Writes heldout_loss.json next to the comparison sheets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402

SIGMAS = (0.15, 0.35, 0.6, 0.85)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=96)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    anima_env.setup_comfy(args.comfy_root)
    index = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    cfg = json.loads((args.ours.parent / "config.json").read_text(encoding="utf-8"))
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    mods = asym.inject(dm, cfg["adapter"]["lora"])
    modes = list(cfg["adapter"]["modes"])
    mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
    asym.load(mods, args.ours, {"mode_embed": mode_embed})
    ctx = load_file(str(args.prep / "contexts.safetensors"))
    data, offsets = ctx["data"], ctx["offsets"].tolist()

    def context(i):
        return anima_env.pad_contexts([data[offsets[i]:offsets[i + 1]]]).to(device, torch.bfloat16)

    lf = anima_env.latent_format()
    needed, cases = set(), []
    groups = sorted(g for g, v in index["groups"].items() if v["split"] == "eval")[: args.groups]
    recs = index["records"]
    for gid in groups:
        ids = index["groups"][gid]["record_ids"]
        target = next((r for r in ids if recs[r]["focus"] == "body_pose"), ids[-1])
        ref = next(r for r in ids if r != target)
        cases.append((gid, target, ref))
        needed |= {target, ref}
    latents = {}
    for name in index["files"]["latents"]:
        part = load_file(str(args.prep / name))
        latents.update({k: v for k, v in part.items() if k in needed})
    lat = {k: lf.process_in(v[None].float().to(device)).to(torch.bfloat16) for k, v in latents.items()}
    empty = context(0)
    memb = mode_embed[modes.index("cs")][None]
    sums = {m: [0.0] * len(SIGMAS) for m in ("base", "right", "wrong")}
    gen = torch.Generator(device="cpu").manual_seed(4321)
    with torch.inference_mode():
        for i, (gid, target, ref) in enumerate(cases):
            x0 = lat[target]
            wrong_ref = lat[cases[(i + 1) % len(cases)][2]]
            ctx_t = context(recs[target]["prompts"]["stripped"])
            noise = torch.randn(x0.shape, generator=gen).to(device)
            caches = {"right": asym.encode_reference(dm, lat[ref], empty, (x0.shape[-2], x0.shape[-1]), mode_emb=memb),
                      "wrong": asym.encode_reference(dm, wrong_ref, empty, (x0.shape[-2], x0.shape[-1]), mode_emb=memb)}
            for j, s in enumerate(SIGMAS):
                xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
                tv = noise - x0.float()
                sig = torch.full((1,), s, device=device)
                asym.set_enabled(mods, False)
                sums["base"][j] += torch.nn.functional.mse_loss(asym.target_forward(dm, xt, sig, ctx_t, None).float(), tv).item()
                asym.set_enabled(mods, True)
                for m in ("right", "wrong"):
                    sums[m][j] += torch.nn.functional.mse_loss(asym.target_forward(dm, xt, sig, ctx_t, caches[m]).float(), tv).item()
    n = len(cases)
    result = {m: [round(v / n, 5) for v in vals] for m, vals in sums.items()}
    result["mean"] = {m: round(sum(v) / len(v), 5) for m, v in result.items()}
    result["identity_gap"] = round(result["mean"]["wrong"] - result["mean"]["right"], 5)
    result["gain_vs_base"] = round(result["mean"]["base"] - result["mean"]["right"], 5)
    result.update({"sigmas": SIGMAS, "groups": n, "adapter": str(args.ours)})
    (args.out / "heldout_loss.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
