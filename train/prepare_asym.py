"""Training tensors for the asymmetric (K/V-cache) reference adapter.

    python train/prepare_asym.py --config configs/train_asym.json --results ~/data/scenes3x4/results_gpu*.jsonl \
        --data-dir ~/data/scenes3x4 --scenes data/scenes.jsonl data/pose_scenes.jsonl --out ~/train/asym_prep

Differences from prepare.py:
  * the reference is the image itself at its own size (no letterbox to the target size): its latent is the
    generator's saved latent, so no VAE pass is needed and target and reference tensors are the same tensor;
  * records whose scene is in an eval split (scene-prep eval scenes, held-out pose combos) are not trained on;
  * held-out characters (by hash, as prepare.py) and every group of theirs are eval only;
  * prompts: stripped / partial / minimal variants as prepare.py plus with_artist (for character-only pairs);
  * reference text: an identity caption of the reference record (its generator appearance and outfit tags,
    no character name, no copyright, no artist), or the empty string (train_asym chooses per sample);
  * "character" pairs: same character, other artist combo (another group of the same character); the target
    prompt then names the target's artists so style comes from text, identity from the reference.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
from prepare import SCENE_KEYS, anima_tag, prompt_variants, read_results, split_groups  # noqa: E402
from prepare_dual import artist_prompt  # noqa: E402


def write_shards(prefix: Path, tensors: dict[str, torch.Tensor], max_bytes: int = 4 << 30) -> list[str]:
    files, shard, size = [], {}, 0
    for key, value in tensors.items():
        shard[key] = value.contiguous()
        size += value.numel() * value.element_size()
        if size >= max_bytes:
            files.append(f"{prefix.name}_{len(files):03d}.safetensors")
            save_file(shard, str(prefix.parent / files[-1]))
            shard, size = {}, 0
    if shard:
        files.append(f"{prefix.name}_{len(files):03d}.safetensors")
        save_file(shard, str(prefix.parent / files[-1]))
    return files


def encode_part(texts: list[str], part: int, parts: int, path: Path, config: dict, dm, device) -> None:
    clip = anima_env.load_text_encoder(config["comfy"]["clip_name"])
    out, started = {}, time.perf_counter()
    for i in range(part, len(texts), parts):
        out[i] = anima_env.encode_prompt(clip, dm, texts[i], device)
        if len(out) % 5000 == 0:
            print(f"text part {part}: {len(out)}/{len(texts) // parts} {time.perf_counter() - started:.0f}s", flush=True)
    torch.save(out, path)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--encode-part":
        part, parts, out = int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
        rest = argparse.ArgumentParser()
        rest.add_argument("--config", type=Path)
        rest.add_argument("--comfy-root")
        a = rest.parse_args(sys.argv[5:])
        config = json.loads(a.config.read_text(encoding="utf-8"))
        anima_env.setup_comfy(a.comfy_root)
        device = torch.device("cuda")
        _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
        texts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
        encode_part(texts, part, parts, out / f"contexts_part{part}.pt", config, dm, device)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train_asym.json")
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, nargs="+", required=True, help="scene lists with a split field")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    args.out.mkdir(parents=True, exist_ok=True)

    eval_scenes = set()
    for path in args.scenes:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "eval":
                    eval_scenes.add(row["scene_key"])
    records = read_results(args.results)
    seen, unique = set(), []
    for r in records:
        if r["record_id"] not in seen:
            seen.add(r["record_id"])
            unique.append(r)
    records = unique
    dropped_scene = [r for r in records if r["scene"].get("scene_key") in eval_scenes]
    records = [r for r in records if r["scene"].get("scene_key") not in eval_scenes]
    by_group = collections.defaultdict(list)
    for r in records:
        by_group[r["group_id"]].append(r)
    groups = {gid: rows for gid, rows in by_group.items() if len(rows) >= 2}
    records = [r for rows in groups.values() for r in rows]
    held_out, eval_groups, seen_groups = split_groups(groups, data_cfg["eval_characters"], data_cfg["seen_eval_groups"])
    print(f"records={len(records)} groups={len(groups)} eval-scene records dropped={len(dropped_scene)} "
          f"held_out_characters={len(held_out)}", flush=True)

    anima_env.setup_comfy(args.comfy_root)
    device = torch.device("cuda")
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)

    prompts: dict[str, int] = {"": 0}
    index_records = {}
    for r in records:
        variants = prompt_variants(r, tuple(data_cfg["partial_keep"]), config["train"]["seed"])
        prefix = [r["quality"], r["rating"]]
        scene = [r["scene"][k] for k in SCENE_KEYS if r["scene"].get(k)]
        variants["with_artist"] = ", ".join(prefix + [artist_prompt(r["artists"]), r["identity"]["gender"], "solo"] + scene)
        ident = r["identity"]
        variants["ref_caption"] = ", ".join([ident["gender"], "solo"] + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]])
        entry = {k: prompts.setdefault(t, len(prompts)) for k, t in variants.items()}
        entry["negative"] = prompts.setdefault(r["negative"], len(prompts))
        index_records[r["record_id"]] = {
            "group_id": r["group_id"], "character": r["character"], "width": r["width"], "height": r["height"],
            "image": r["image"], "seed": r["seed"], "focus": r["scene"].get("scene_focus"),
            "framing": r["scene"].get("framing"), "prompt_original": r["prompt"], "prompts": entry,
        }
    texts = list(prompts)
    workers = max(1, torch.cuda.device_count())
    parts = [args.out / f"contexts_part{w}.pt" for w in range(workers)]
    started = time.perf_counter()
    if workers > 1:
        # one encoder process per GPU (single-prompt encoding is overhead-bound); this process does part 0
        import subprocess
        (args.out / "prompts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        procs = [subprocess.Popen([sys.executable, __file__, "--encode-part", str(w), str(workers), str(args.out),
                                   "--config", str(args.config), "--comfy-root", args.comfy_root],
                                  env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(w)}) for w in range(1, workers)]
    encode_part(texts, 0, workers, parts[0], config, dm, device)
    if workers > 1:
        for proc in procs:
            if proc.wait() != 0:
                raise RuntimeError("an encoder process failed")
    contexts = [None] * len(texts)
    for w, part in enumerate(parts):
        for i, c in torch.load(part).items():
            contexts[i] = c
    assert all(c is not None for c in contexts)
    offsets = torch.tensor([0] + [c.shape[0] for c in contexts]).cumsum(0)
    save_file({"data": torch.cat(contexts).contiguous(), "offsets": offsets}, str(args.out / "contexts.safetensors"))
    print(f"contexts: {len(prompts)} prompts in {time.perf_counter() - started:.0f}s", flush=True)
    del contexts

    latents = {}
    for r in records:
        lat = load_file(str(args.data_dir / r["latent"]))["latent"]
        lat = lat[0] if lat.shape[0] == 1 else lat
        latents[r["record_id"]] = (lat.unsqueeze(1) if lat.ndim == 3 else lat).to(torch.float16)
    files = write_shards(args.out / "latents", latents)
    print(f"latents: {len(latents)} in {len(files)} shards", flush=True)

    split_of = {gid: ("eval" if rows[0]["character"] in held_out else "train") for gid, rows in groups.items()}
    eval_cases = []
    for split, gids in (("held_out", eval_groups), ("seen", seen_groups)):
        for gid in gids:
            rows = sorted(groups[gid], key=lambda r: r["record_id"])
            ref = next((r for r in rows if r["scene"].get("scene_focus") != "body_pose"), rows[0])
            target = next((r for r in rows if r is not ref and r["scene"].get("scene_focus") == "body_pose"),
                          next(r for r in rows if r is not ref))
            eval_cases.append({"split": split, "group_id": gid, "character": ref["character"], "ref": ref["record_id"],
                               "target": target["record_id"], "width": target["width"], "height": target["height"],
                               "seed": 777 + len(eval_cases)})
    index = {
        "created": time.time(), "config": config, "data_dir": str(args.data_dir), "records": index_records,
        "groups": {gid: {"split": split_of[gid], "character": rows[0]["character"],
                         "artists": rows[0]["artists"], "record_ids": sorted(r["record_id"] for r in rows)}
                   for gid, rows in groups.items()},
        "held_out_characters": sorted(held_out), "prompts": list(prompts), "eval_cases": eval_cases,
        "files": {"latents": files}, "dropped_eval_scene_records": len(dropped_scene),
    }
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    train_groups = sum(1 for s in split_of.values() if s == "train")
    print(f"done: train_groups={train_groups} eval_cases={len(eval_cases)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
