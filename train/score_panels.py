"""Score compare_refs --save-panels output with image-embedding similarity.

    python train/score_panels.py RUN_DIR [RUN_DIR ...] --label v1a v1b --out scores.json

For each case dir (<run>/<case>/{ref.png, truth.jpg?, <method>.jpg}) and method:
  dino_truth  DINOv2 CLS cosine to the held-out truth image (same character, the prompted pose) - identity
  dino_ref    DINOv2 CLS cosine to the reference image - identity, but also rewards copying the ref pose
  clip_truth  CLIP image cosine to the truth image
  clip_text   CLIP image-text cosine to the prompt (minus quality tags) - prompt/pose following
Methods from later runs get prefixed by their label so ours-from-v1a and ours-from-v1b can be told apart.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

QUALITY = {"masterpiece", "best quality", "score_7", "newest", "highres", "absurdres", "safe"}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--label", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert len(args.label) == len(args.runs)
    dev = "cuda"
    dproc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval().half()
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()

    def d_emb(p):
        x = dproc(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"].to(dev).half()
        return torch.nn.functional.normalize(dino(pixel_values=x).pooler_output.float(), dim=-1)[0]

    def c_emb(p):
        x = cproc(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"].to(dev).half()
        return torch.nn.functional.normalize(feat(clip.get_image_features(pixel_values=x)), dim=-1)[0]

    def t_emb(text):
        x = cproc(text=[text], return_tensors="pt", padding=True, truncation=True).to(dev)
        return torch.nn.functional.normalize(feat(clip.get_text_features(**x)), dim=-1)[0]

    def feat(o):   # transformers 5 returns a model output from get_*_features, 4.x a tensor
        return (o if torch.is_tensor(o) else o.pooler_output).float()

    rows = []
    for label, run in zip(args.label, args.runs):
        cases = {c["name"]: c for c in json.loads((run / "summary.json").read_text(encoding="utf-8"))["cases"]}
        for name, c in cases.items():
            cd = run / name
            if not cd.is_dir():
                continue
            ref, truth = cd / "ref.png", cd / "truth.jpg"
            dr, cr = d_emb(ref), c_emb(ref)
            dt, ct = (d_emb(truth), c_emb(truth)) if truth.exists() else (None, None)
            text = ", ".join(t for t in (s.strip() for s in c["prompt"].split(",")) if t not in QUALITY)
            te = t_emb(text)
            for img in sorted(cd.glob("*.jpg")):
                if img.stem == "truth":
                    continue
                method = img.stem if img.stem in ("base", "v2") else f"{label}_{img.stem}"
                di, ci = d_emb(img), c_emb(img)
                rows.append({"case": name, "held_out": name.startswith("held_out"), "method": method,
                             "dino_ref": float(di @ dr), "clip_text": float(ci @ te),
                             "dino_truth": float(di @ dt) if dt is not None else None,
                             "clip_truth": float(ci @ ct) if ct is not None else None})
    # base / v2 appear in several runs with identical images; keep the first copy
    seen, uniq = set(), []
    for r in rows:
        k = (r["case"], r["method"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    summary = {}
    for m in sorted({r["method"] for r in uniq}):
        rs = [r for r in uniq if r["method"] == m]
        held = [r for r in rs if r["held_out"]]
        avg = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
        summary[m] = {"n": len(rs), "n_held_out": len(held),
                      "dino_truth": avg([r["dino_truth"] for r in held]),
                      "clip_truth": avg([r["clip_truth"] for r in held]),
                      "dino_ref": avg([r["dino_ref"] for r in rs]),
                      "clip_text": avg([r["clip_text"] for r in rs])}
    args.out.write_text(json.dumps({"summary": summary, "rows": uniq}, indent=1), encoding="utf-8")
    for m, s in summary.items():
        print(f"{m:10s}", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
