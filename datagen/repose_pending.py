"""Replace the scene of every record that is not complete yet with a mined body-pose scene
(data/pose_scenes.jsonl, datagen/mine_pose_scenes.py). Character, artists, framing, size, seed,
background and lighting stay; the prompt is rebuilt with the same tag order as build_manifest_scenes.py.

    python datagen/repose_pending.py --run scenes3x4 --results /tmp/res/*.jsonl

Rules: only train-split pose scenes; scenes that need legs (sitting, lying, walking...) only for full body,
wide and cowboy shots; inside a group no scene overlaps another image of the group by tag Jaccard > 0.5,
and the body tags differ from the other images when possible; scenes are dealt round-robin so each is
used about equally often. An expression (vocab.EXPRESSIONS) is added, and a camera tag (vocab.CAMERA)
when the pose scene has none. The old manifest is kept as manifest.pre-pose.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vocab  # noqa: E402
from build_manifest import ROOT, artist_prompt, weighted  # noqa: E402
from catalog import anima_tag  # noqa: E402

LEG_FRAMINGS = ("full body", "full body, wide shot", "cowboy shot")
CAMERA_TAGS = {"from side", "from behind", "from above", "from below", "profile", "dutch angle", "looking to the side",
               "looking up", "looking down", "looking back", "upside-down", "foreshortening", "facing away"}


def identity_tags(r: dict) -> list[str]:
    ident = r["identity"]
    return [ident["gender"], "solo", r["character"]] + ([anima_tag(ident["copyright"])] if ident["copyright"] else []) \
        + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]]


def prompt(r: dict) -> str:
    s = r["scene"]
    scene = [s[k] for k in ("framing", "camera", "pose", "expression", "background", "lighting") if s.get(k)]
    return ", ".join([r["quality"], "safe", artist_prompt(r["artists"]), *identity_tags(r), *scene])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--results", type=Path, nargs="+", required=True)
    ap.add_argument("--pose-scenes", type=Path, default=ROOT / "data" / "pose_scenes.jsonl")
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run
    records = [json.loads(l) for l in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    for r in records:  # the rebuilt prompt must equal the original for untouched scenes
        assert prompt(r) == r["prompt"], r["record_id"]
    scene_tags = {json.loads(l)["scene_key"]: set(json.loads(l)["danbooru_tags"])
                  for l in (ROOT / "data" / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if l}
    done = set()
    for path in args.results:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "complete":
                done.add(row["record_id"])
    poses = [json.loads(l) for l in args.pose_scenes.read_text(encoding="utf-8").splitlines() if l]
    poses = [p for p in poses if p["split"] == "train"]
    pools = {True: poses, False: [p for p in poses if not p["needs_legs"]]}
    queues = {True: [], False: []}
    rng = random.Random(args.seed)

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
    changed = 0
    for gid in sorted(by_group):
        group = by_group[gid]
        used = [scene_tags.get(r["scene"]["scene_key"], set()) for r in group if r["record_id"] in done]
        used_body = set()
        for r in group:
            if r["record_id"] in done:
                continue
            p = draw(r["scene"]["framing"] in LEG_FRAMINGS, used, used_body)
            tags = set(p["danbooru_tags"])
            used.append(tags)
            if p["needs_legs"]:
                used_body |= tags
            s = r["scene"]
            camera = "" if CAMERA_TAGS & set(p["tags"]) else weighted(rng, vocab.CAMERA)
            s.update({"camera": camera, "pose": ", ".join(p["tags"]), "expression": weighted(rng, vocab.EXPRESSIONS),
                      "scene_key": p["scene_key"], "scene_focus": "body_pose"})
            r["prompt"] = prompt(r)
            changed += 1
    (run_dir / "manifest.jsonl").rename(run_dir / "manifest.pre-pose.jsonl")
    (run_dir / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                           encoding="utf-8", newline="\n")
    summary_path = run_dir / "manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["focus"] = dict(collections.Counter(r["scene"]["scene_focus"] for r in records))
    summary["pose_scenes"] = {"file": str(args.pose_scenes.relative_to(ROOT)), "train_scenes": len(poses),
                              "records_reposed": changed, "complete_before": len(done)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    use = collections.Counter(r["scene"]["scene_key"] for r in records if r["scene"]["scene_focus"] == "body_pose")
    print(json.dumps({"records": len(records), "complete": len(done), "reposed": changed, "pose_scenes_used": len(use),
                      "uses_min_max": [min(use.values()), max(use.values())] if use else None,
                      "focus": summary["focus"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
