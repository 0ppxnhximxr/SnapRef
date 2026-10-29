"""Low- and high-resolution target latents for multi-resolution training (v1.x), from the generator's images.

    python train/prepare_multires.py --prep ~/train/asym_prep --data-dir ~/data/scenes3x4 --out ~/train/asym_prep_lowres \
        [--part 0 --parts 2]

For every training record, the original image is resized by x0.75 per side (~0.56 MP, Lanczos) and VAE-encoded.
Those are the "low" targets. References keep their own latents at ~1 MP, so the target is smaller than the
reference, as at inference with a small output. "High" targets (~1.3 MP) need fresh renders and come from a
separate datagen run; they are not made here.
Writes latents_low_<part>.safetensors {record_id: fp16 raw VAE latent [16, 1, h, w]}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402

LOW = {(1024, 1024): (768, 768), (896, 1152): (672, 864), (832, 1216): (624, 912), (1216, 832): (912, 624)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    idx = json.loads((args.prep / "index.json").read_text(encoding="utf-8"))
    cfg = idx["config"]
    anima_env.setup_comfy(args.comfy_root)
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    train_groups = {g for g, v in idx["groups"].items() if v["split"] == "train"}
    ids = sorted(k for k, r in idx["records"].items() if r["group_id"] in train_groups)[args.part::args.parts]
    args.out.mkdir(parents=True, exist_ok=True)
    out, done = {}, 0
    for k in ids:
        r = idx["records"][k]
        size = LOW.get((r["width"], r["height"]))
        if size is None:
            continue
        img = Image.open(args.data_dir / r["image"]).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        lat = anima_env.vae_encode_image(vae, img)
        assert lat.shape[-2:] == (size[1] // 8, size[0] // 8), (k, lat.shape)
        out[k] = lat.to(torch.float16).contiguous()
        done += 1
        if done % 2000 == 0:
            print(f"part {args.part}: {done}/{len(ids)}", flush=True)
    path = args.out / f"latents_low_{args.part}.safetensors"
    save_file(out, str(path.with_suffix(".partial")))
    path.with_suffix(".partial").rename(path)
    print(json.dumps({"part": args.part, "latents": len(out), "file": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
