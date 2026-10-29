"""Add body-pose images to every group of a run: same character and artist combo, new records
<group>_04, <group>_05, ... with scenes from data/pose_scenes.jsonl (datagen/mine_pose_scenes.py).

    python datagen/extend_pose.py --run scenes3x4 --per-group 2

Rules (as repose_pending.py): train-split pose scenes only; leg poses only in full body / wide / cowboy
shots; no scene overlapping another image of the group by tag Jaccard > 0.5; body tags differ from the
group's other pose images when possible; round-robin so scenes are used about equally. Framing favours
shots that show the body, counted down by half for each use inside the group. Expression from
vocab.EXPRESSIONS, camera from vocab.CAMERA when the scene has none. Old files are kept as *.pre-extend.*.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vocab  # noqa: E402
from build_manifest import ROOT, weighted  # noqa: E402
from repose_pending import CAMERA_TAGS, LEG_FRAMINGS, jaccard, prompt  # noqa: E402

POSE_FRAMINGS = {"full body": 0.4, "cowboy shot": 0.3, "full body, wide shot": 0.15, "upper body": 0.15}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "datagen_scenes.json")
    ap.add_argument("--pose-scenes", type=Path, default=ROOT / "data" / "pose_scenes.jsonl")
    args = ap.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / args.run
    records = read_jsonl(run_dir / "manifest.jsonl")
    groups = read_jsonl(run_dir / "groups.jsonl")
    poses_all = read_jsonl(args.pose_scenes)
    tags_of = {s["scene_key"]: set(s["danbooru_tags"]) for s in read_jsonl(ROOT / "data" / "scenes.jsonl") + poses_all}
    legs_of = {p["scene_key"]: p["needs_legs"] for p in poses_all}
    poses = [p for p in poses_all if p["split"] == "train"]
    pools = {True: poses, False: [p for p in poses if not p["needs_legs"]]}
    queues = {True: [], False: []}
    rng = random.Random(config["seed"] + 7919)

    def draw(legs_ok: bool, used: list[set], used_body: set) -> dict:
        fallback = None
        for _ in range(400):
            q = queues[legs_ok]
            if not q:
                q.extend(pools[legs_ok])
                rng.shuffle(q)
            p = q.pop()
            tags = set(p["danbooru_tags"])
            if any(jaccard(tags, u) > 0.5 for u in used):
                continue
            if not (tags & used_body) or not p["needs_legs"]:
                return p
            fallback = fallback or p
        if fallback is None:
            raise RuntimeError("no pose scene fits the group")
        return fallback

    by_group = collections.defaultdict(list)
    for r in records:
        by_group[r["group_id"]].append(r)
    new_records = []
    for g in groups:
        rows = sorted(by_group[g["group_id"]], key=lambda r: r["record_id"])
        base = rows[0]
        used = [tags_of[r["scene"]["scene_key"]] for r in rows]
        used_body = {t for r in rows if r["scene"]["scene_focus"] == "body_pose" and legs_of.get(r["scene"]["scene_key"])
                     for t in tags_of[r["scene"]["scene_key"]]}
        framings = [r["scene"]["framing"] for r in rows]
        start = 1 + max(int(r["record_id"].rsplit("_", 1)[1]) for r in rows)
        for i in range(start, start + args.per_group):
            names = list(POSE_FRAMINGS)
            framing = rng.choices(names, weights=[POSE_FRAMINGS[f] * 0.5 ** framings.count(f) for f in names])[0]
            framings.append(framing)
            p = draw(framing in LEG_FRAMINGS, used, used_body)
            tags = set(p["danbooru_tags"])
            used.append(tags)
            if p["needs_legs"]:
                used_body |= tags
            width, height = rng.choice(vocab.FRAMINGS[framing][1])
            record_id = f"{g['group_id']}_{i:02d}"
            scene = {"framing": framing, "camera": "" if CAMERA_TAGS & set(p["tags"]) else weighted(rng, vocab.CAMERA),
                     "pose": ", ".join(p["tags"]), "expression": weighted(rng, vocab.EXPRESSIONS),
                     "background": weighted(rng, vocab.BACKGROUNDS), "lighting": weighted(rng, vocab.LIGHTING),
                     "width": width, "height": height, "scene_key": p["scene_key"], "scene_focus": "body_pose"}
            r = {k: base[k] for k in ("group_id", "character", "character_danbooru", "artists", "identity", "rating",
                                       "quality", "negative")}
            r.update(record_id=record_id, scene=scene, width=width, height=height,
                     seed=int.from_bytes(hashlib.sha256(f"{config['seed']}:{record_id}".encode()).digest()[:8], "little"))
            r["prompt"] = prompt(r)
            new_records.append(r)
            g["record_ids"].append(record_id)
    ids = [r["record_id"] for r in records + new_records]
    assert len(ids) == len(set(ids)), "duplicate record id"
    for name in ("manifest.jsonl", "groups.jsonl", "manifest_summary.json"):
        src = run_dir / name
        src.rename(src.with_name(src.stem + ".pre-extend" + src.suffix))
    all_records = records + new_records
    (run_dir / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_records),
                                           encoding="utf-8", newline="\n")
    (run_dir / "groups.jsonl").write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in groups),
                                         encoding="utf-8", newline="\n")
    summary = json.loads((run_dir / "manifest_summary.pre-extend.json").read_text(encoding="utf-8"))
    summary["records"] = len(all_records)
    summary["framings"] = dict(collections.Counter(r["scene"]["framing"] for r in all_records))
    summary["focus"] = dict(collections.Counter(r["scene"]["scene_focus"] for r in all_records))
    summary["pose_extend"] = {"per_group": args.per_group, "records_added": len(new_records)}
    (run_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    use = collections.Counter(r["scene"]["scene_key"] for r in all_records if r["scene"]["scene_focus"] == "body_pose")
    print(json.dumps({"records": len(all_records), "added": len(new_records), "pose_scenes_used": len(use),
                      "pose_uses_min_max": [min(use.values()), max(use.values())],
                      "added_framings": dict(collections.Counter(r["scene"]["framing"] for r in new_records)),
                      "focus": summary["focus"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
