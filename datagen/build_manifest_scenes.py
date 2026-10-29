"""Manifest for the scene-prep data run: research-repo groups (character x artist combo), scenes from
scene-prep/out/scenes.jsonl instead of datagen/vocab.py poses and expressions.

    python build_manifest_scenes.py --config ../configs/datagen_scenes.json

Same record format as datagen/build_manifest.py (run_worker.py, prepare*.py read it unchanged).
Differences:
  * scene = one curated scene (its tags kept together) + framing + background + lighting from vocab;
    record["scene"] keeps the vocab keys: framing, camera (""), pose (scene tags), expression (""),
    background, lighting, plus scene_key / scene_focus for bookkeeping;
  * scenes are dealt round-robin from per-focus queues, each reshuffled when exhausted, so every scene
    is used about equally often and assignment is independent of character and style;
  * framing: the reviewed framings plus full body / wide shot for every scene, weighted as vocab.FRAMINGS;
    every group gets at least one full-body image;
  * within a group no two images share a scene or a similar one (tag Jaccard > max_group_overlap), and
    framings differ when the scene allows it; by default every scene is used equally often
    (focus_weights "uniform"; "sqrt" boosts rare focus classes);
  * only rating "safe" train-split scenes are used (rating stays "safe" for every record).
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
from build_manifest import (ROOT, artist_combo, artist_prompt, eligible_characters, popularity_sample,  # noqa: E402
                            weighted, write_jsonl)
from catalog import anima_tag, load_artists  # noqa: E402


def load_scenes(path: Path, split: str, exclude_tags: set[str]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["split"] == split and r["rating"] == "safe" and r["framings"]
            and not exclude_tags & set(r["danbooru_tags"])]


class SceneDealer:
    """Round-robin over scenes, balanced across focus classes by weight."""

    def __init__(self, scenes: list[dict], focus_weights: dict[str, float], rng: random.Random):
        self.rng = rng
        self.pools = collections.defaultdict(list)
        for s in scenes:
            self.pools[s["focus"]].append(s)
        self.weights = {f: focus_weights.get(f, 1.0) for f in self.pools}
        self.queues = {f: [] for f in self.pools}

    def _next(self, focus: str) -> dict:
        queue = self.queues[focus]
        if not queue:
            queue.extend(self.pools[focus])
            self.rng.shuffle(queue)
        return queue.pop()

    def deal(self, used: list[set[str]], max_overlap: float) -> dict:
        """Next scene whose tags overlap no scene already in the group by more than max_overlap (Jaccard)."""
        focuses = list(self.weights)
        for _ in range(200):
            focus = self.rng.choices(focuses, weights=[self.weights[f] for f in focuses])[0]
            scene = self._next(focus)
            tags = set(scene["danbooru_tags"])
            if all(len(tags & u) / len(tags | u) <= max_overlap for u in used):
                return scene
        raise RuntimeError("could not draw a scene that differs from the rest of the group")


FULL_BODY = ("full body", "full body, wide shot")


def allowed_framings(scene: dict) -> list[str]:
    """Reviewed framings plus full-body shots: any scene reads at full body, only the face gets smaller."""
    return [f for f in vocab.FRAMINGS if f in scene["framings"] or f in FULL_BODY]


def pick_framing(rng: random.Random, scene: dict, used: list[str], need_full_body: bool) -> str:
    allowed = allowed_framings(scene)
    if need_full_body:
        allowed = [f for f in allowed if f in FULL_BODY]
    # vocab weights, each framing already in the group counted down by half: groups stay mixed without
    # inflating the rare wide shot the way "only unused framings" would
    weights = [vocab.FRAMINGS[f][0] * 0.5 ** used.count(f) for f in allowed]
    return rng.choices(allowed, weights=weights)[0]


def build(config: dict, existing: list[dict] | None = None) -> tuple[list[dict], list[dict], dict]:
    """existing: groups of an earlier manifest of this run. They are kept as they are (their records
    are not rebuilt here); new groups bring every character up to uses_per_character, never repeating
    a (character, artist combo) pair, and are numbered after the existing ones."""
    rng = random.Random(config["seed"] + (len(existing) if existing else 0))
    chars = eligible_characters(config["characters"])
    if config["characters"].get("limit"):
        chars = chars[: int(config["characters"]["limit"])]
    artists = load_artists(ROOT / config["artists"]["catalog"], config["artists"]["min_post_count"])
    # The character pool includes characters who are minors, so suggestive expressions are never generated.
    scenes = load_scenes(ROOT / config["scenes"]["path"], config["scenes"].get("split", "train"),
                         set(config["scenes"].get("exclude_tags", [])))
    weights = config["scenes"].get("focus_weights", "uniform")
    if isinstance(weights, str):  # "uniform": every scene equally often; "sqrt": rare focus classes boosted
        power = {"uniform": 1.0, "sqrt": 0.5}[weights]
        sizes = collections.Counter(s["focus"] for s in scenes)
        weights = {f: n ** power for f, n in sizes.items()}
    dealer = SceneDealer(scenes, weights, rng)
    max_overlap = float(config["scenes"].get("max_group_overlap", 0.5))

    combos_per_char = int(config["characters"]["uses_per_character"])
    combo_uses = int(config["artists"]["uses_per_combo"])
    have = collections.Counter(g["character"] for g in existing or [])
    char_slots = [c for c in chars for _ in range(max(0, combos_per_char - have[c["tag"]]))]
    n_groups = len(char_slots)
    combos = [artist_combo(rng, artists, config["artists"]) for _ in range(-(-n_groups // combo_uses))]
    combo_slots = [c for c in combos for _ in range(combo_uses)][:n_groups]
    rng.shuffle(char_slots)
    rng.shuffle(combo_slots)
    pair_key = lambda i: (char_slots[i]["tag"], json.dumps(combo_slots[i], sort_keys=True))  # noqa: E731
    seen = {(g["character"], json.dumps(g["artists"], sort_keys=True)) for g in existing or []}
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if pair_key(i) not in seen:
                break
            combo_slots[i], combo_slots[j] = combo_slots[j], combo_slots[i]
        seen.add(pair_key(i))

    quality, negative = config["prompt"]["quality"], config["prompt"]["negative"]
    images_per_group = int(config["images_per_group"])
    records, groups = [], []
    scene_use, framing_use = collections.Counter(), collections.Counter()
    for g, (char, combo) in enumerate(zip(char_slots, combo_slots), start=len(existing or [])):
        group_id = f"{config.get('id_prefix', 's')}{g:06d}"
        ident = char["identity"]
        identity_tags = [ident["gender"], "solo", char["tag"]] + ([anima_tag(ident["copyright"])] if ident["copyright"] else []) \
            + [anima_tag(t) for t in ident["appearance"] + ident["outfit"]]
        used_tags, used_framings, group_records = [], [], []
        for i in range(images_per_group):
            scene = dealer.deal(used_tags, max_overlap)
            used_tags.append(set(scene["danbooru_tags"]))
            # last image of a group is full body when none came up, so every group has a full-body reference
            need_full = i == images_per_group - 1 and not any(f in FULL_BODY for f in used_framings)
            framing = pick_framing(rng, scene, used_framings, need_full)
            used_framings.append(framing)
            width, height = rng.choice(vocab.FRAMINGS[framing][1])
            fields = {
                "framing": framing, "camera": "", "pose": ", ".join(scene["tags"]), "expression": "",
                "background": weighted(rng, vocab.BACKGROUNDS), "lighting": weighted(rng, vocab.LIGHTING),
                "width": width, "height": height, "scene_key": scene["scene_key"], "scene_focus": scene["focus"],
            }
            scene_tags = [fields[k] for k in ("framing", "pose", "background", "lighting") if fields[k]]
            prompt = ", ".join([quality, "safe", artist_prompt(combo), *identity_tags, *scene_tags])
            record_id = f"{group_id}_{i:02d}"
            seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{record_id}".encode()).digest()[:8], "little")
            group_records.append({
                "record_id": record_id, "group_id": group_id, "character": char["tag"],
                "character_danbooru": char["danbooru_tag"], "artists": combo, "identity": ident, "rating": "safe",
                "quality": quality, "scene": fields, "prompt": prompt, "negative": negative,
                "width": width, "height": height, "seed": seed,
            })
            scene_use[scene["scene_key"]] += 1
            framing_use[framing] += 1
        records.extend(group_records)
        groups.append({"group_id": group_id, "character": char["tag"], "artists": combo,
                       "record_ids": [r["record_id"] for r in group_records]})
    uses = sorted(scene_use.values())
    stats = {"scenes_available": len(scenes), "scenes_used": len(scene_use),
             "scene_uses_min_median_max": [uses[0], uses[len(uses) // 2], uses[-1]] if uses else None,
             "framings": dict(framing_use),
             "focus": dict(collections.Counter(r["scene"]["scene_focus"] for r in records))}
    return records, groups, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit-characters", type=int, help="first N eligible characters (smoke test)")
    parser.add_argument("--extend", action="store_true",
                        help="keep runs/<run>/manifest.jsonl as it is and append groups for the new config")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.workers:
        config["workers"] = args.workers
    if args.limit_characters:
        config["characters"]["limit"] = args.limit_characters
    run_dir = ROOT / "runs" / config["run_name"]
    old_records, old_groups = [], []
    if args.extend:
        old_records = [json.loads(l) for l in (run_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
        old_groups = [json.loads(l) for l in (run_dir / "groups.jsonl").read_text(encoding="utf-8").splitlines() if l]
    new_records, new_groups, stats = build(config, old_groups)
    records, groups = old_records + new_records, old_groups + new_groups
    stats["extended_from"] = {"groups": len(old_groups), "records": len(old_records)} if args.extend else None
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes = {"manifest": write_jsonl(run_dir / "manifest.jsonl", records),
              "groups": write_jsonl(run_dir / "groups.jsonl", groups)}
    by_group = collections.defaultdict(list)
    for record in records:
        by_group[record["group_id"]].append(record)
    group_ids = sorted(by_group)
    for k in range(int(config["workers"])):
        shard = [r for gid in group_ids[k::int(config["workers"])] for r in by_group[gid]]
        hashes[f"shard_{k}"] = write_jsonl(run_dir / f"shard_{k}.jsonl", shard)
    summary = {
        "run_name": config["run_name"], "groups": len(groups), "records": len(records),
        "unique_characters": len({g["character"] for g in groups}),
        "unique_artist_combos": len({json.dumps(g["artists"], sort_keys=True) for g in groups}),
        "unique_artists": len({a["tag"] for g in groups for a in g["artists"]}),
        "resolutions": dict(collections.Counter(f"{r['width']}x{r['height']}" for r in records)),
        **stats, "workers": int(config["workers"]), "sha256": hashes, "config": config,
    }
    (run_dir / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "sha256")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
