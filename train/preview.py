"""In-training preview: our adapter's images for a fixed set of cases, rendered inside the training process
(model already loaded) every eval_every steps, next to cached competitor images rendered once.

  * the fixed cases and the competitor/base panels come from one compare_refs.py run (its summary.json lists
    the cases with prompt, negative, size, seed and reference path; the per-method panels are saved by
    compare_refs.py --save-panels as <case>/<method>.jpg);
  * preview(...) samples ours for every case at the same seed and steps, then writes one row per case
    (reference | OURS step N | pilot_r32 | Role-Ref-KV | base | target) into <out>/<run>_<step>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anima_env  # noqa: E402
import asym  # noqa: E402
from sample import labeled_row, to_pil  # noqa: E402

ORDER = [("v2", "v2_r64 (ElectronGirl)"), ("pilot", "pilot_r32 (ElectronGirl)"), ("rrkv", "Role-Ref-KV (AnticPan)"), ("base", "base, no reference"),
         ("truth", "generator target (full tags)")]


class Preview:
    def __init__(self, fixed_dir: Path, clip_name: str, vae_name: str, device):
        self.dir = Path(fixed_dir)
        summary = json.loads((self.dir / "summary.json").read_text(encoding="utf-8"))
        self.cases = summary["cases"]
        self.steps, self.cfg = summary["steps"], summary["cfg"]
        self.device = device
        self.clip_name, self.vae_name = clip_name, vae_name
        self.ctx = None

    def _ref_path(self, c: dict) -> Path:
        local = self.dir / c["name"] / "ref.png"
        return local if local.exists() else Path(c["ref_image"])

    def _prepare(self, dm):
        """Text contexts and reference latents, encoded once (text encoder and VAE freed afterwards except VAE)."""
        clip = anima_env.load_text_encoder(self.clip_name)
        enc = lambda t: anima_env.encode_prompt(clip, dm, t, self.device)  # noqa: E731
        self.ctx = {"": enc("")}
        for c in self.cases:
            for key in ("prompt", "negative"):
                if c[key] not in self.ctx:
                    self.ctx[c[key]] = enc(c[key])
        del clip
        self.vae = anima_env.load_vae(self.vae_name)
        lf = anima_env.latent_format()
        self.refs = {}
        for c in self.cases:
            img = Image.open(self._ref_path(c)).convert("RGB")
            s = min(1.0, (1024 * 1024 / (img.width * img.height)) ** 0.5)
            img = img.resize(tuple(max(16, int(n * s) // 16 * 16) for n in img.size), Image.Resampling.LANCZOS)
            self.refs[c["name"]] = lf.process_in(anima_env.vae_encode_image(self.vae, img)[None].to(self.device)).to(torch.bfloat16)
        self.lf = lf

    @torch.inference_mode()
    def render(self, dm, mode_emb: torch.Tensor, out_dir: Path, label: str) -> None:
        if self.ctx is None:
            self._prepare(dm)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for c in self.cases:
            shape = (1, 16, 1, int(c["height"]) // 8, int(c["width"]) // 8)
            x = asym.sample(dm, self.ctx[c["prompt"]], self.ctx[c["negative"]], self.ctx[""], shape, int(c["seed"]),
                            self.steps, self.cfg, 3.0, self.device, self.refs[c["name"]], mode_emb=mode_emb)
            ours = to_pil(anima_env.vae_decode(self.vae, self.lf.process_out(x.to(self.device)).float())[0])
            panels = [(f"reference ({c['character']})", Image.open(self._ref_path(c)).convert("RGB")), (f"OURS {label}", ours)]
            for key, title in ORDER:
                p = self.dir / c["name"] / f"{key}.jpg"
                if p.exists():
                    panels.append((title, Image.open(p).convert("RGB")))
            row = labeled_row(panels)
            row.save(out_dir / f"{c['name']}.jpg", quality=88)
            rows.append(row)
        # preview.json, not summary.json: the pipeline's full comparison checks for summary.json
        (out_dir / "preview.json").write_text(json.dumps({"label": label, "fixed": str(self.dir), "steps": self.steps,
                                                          "cfg": self.cfg, "cases": len(self.cases)}), encoding="utf-8")
        torch.cuda.empty_cache()


def main() -> int:
    """Standalone: python train/preview.py --fixed DIR --adapter adapter_000500.safetensors --out DIR --label 'step 500'"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    device = torch.device("cuda")
    anima_env.setup_comfy(args.comfy_root)
    cfg = json.loads((args.adapter.parent / "config.json").read_text(encoding="utf-8"))
    _, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    mods = asym.inject(dm, cfg["adapter"]["lora"])
    modes = list(cfg["adapter"]["modes"])
    mode_embed = torch.zeros(len(modes), dm.model_channels, device=device)
    asym.load(mods, args.adapter, {"mode_embed": mode_embed})
    p = Preview(args.fixed, cfg["comfy"]["clip_name"], cfg["comfy"]["vae_name"], device)
    p.render(dm, mode_embed[modes.index("cs")][None], args.out, args.label or args.adapter.stem)
    print(f"preview written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
