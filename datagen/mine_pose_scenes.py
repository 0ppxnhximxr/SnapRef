"""Pose scenes mined from Danbooru posts: combinations of body-pose tags that real solo, general-rated
illustrations carry together, so base Anima knows each combination.

    python datagen/mine_pose_scenes.py --posts ~/hf/danbooru2025-metadata/data \
        --taxonomy /tmp/danbooru_tag_taxonomy.csv --out data/pose_scenes.jsonl

Per post (rating g, tagged solo, no multi-person tag, no excluded tag): the set of its pose tags
(posture, kinetic, gesture, a combat and a camera allowlist). Combos of 1-4 tags with at least one
body tag and at least --min-posts posts become scenes. Suggestive and fanservice pose tags are excluded
outright because the character pool contains minors. 10% of combos (by hash) are held out for eval.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import anima_tag  # noqa: E402

COMBAT = {"fighting_stance", "punching", "kicking", "martial_arts", "blocking", "flying_kick"}
CAMERA = {"from_side", "from_behind", "from_above", "from_below", "profile", "dutch_angle", "looking_to_the_side",
          "looking_up", "upside-down", "foreshortening", "facing_away"}
EXCLUDE_EXACT = {
    "holding", "spread_legs", "straddling", "cowgirl_position", "all_fours", "bent_over", "arched_back", "legs_up",
    "leg_up", "leg_lift", "knees_together_feet_apart", "arm_under_breasts", "covering_breasts", "hand_between_own_legs",
    "top-down_bottom-up", "pole_dancing", "ass_shake", "strap_lift", "bound_arms", "high_kick", "adjusting_clothes",
    "presenting", "squatting_cowgirl_position", "reverse_cowgirl_position", "fetal_position", "tachi-e", "swimming",
    "wading", "afloat", "freediving", "bathing", "driving", "kneepits", "sleeping_upright", "crotch_grab",
    "skirt_hold", "dress_hold", "clothes_pull", "shirt_lift", "skirt_lift", "dress_lift", "clothes_lift",
    "finger_sucking", "finger_in_own_mouth", "mouth_pull", "tongue_out", "licking", "licking_lips", "trembling",
    "head_out_of_frame", "out_of_frame", "feet_out_of_frame", "cropped_legs", "hand_on_own_crotch",
    "hand_on_own_ass", "hand_on_own_thigh", "between_legs", "leg_hug", "pantyhose_pull", "thighhighs_pull",
    "undressing", "masturbation", "humping", "bouncing_breasts", "breast_hold", "self_hug", "kabedon",
    "hand_in_own_hair", "the_pose", "reclining", "arms_behind_head", "arm_behind_head", "arms_between_legs",
    "kine", "hanging", "dropping",  # pin-up poses and ambiguous tags: keep the set conservative
}
EXCLUDE_SUBSTR = ("breast", "ass", "crotch", "pant", "skirt", "panties", "thigh", "feet", "foot", "toe", "leg_lock",
                  "bound", "tied", "restrain", "grab", "grop", "lift", "pull", "spread", "nude", "naked", "bath",
                  "lick", "suck", "kiss", "bondage", "cuff", "choke", "stab", "violence", "blood")
MULTI = {"2girls", "3girls", "4girls", "5girls", "6+girls", "2boys", "3boys", "4boys", "5boys", "6+boys",
         "multiple_girls", "multiple_boys", "1boy_1girl", "group_picture", "everyone", "crowd"}
# tags whose body part sits below the waist: only full body / wide / cowboy shots show them
NEEDS_LEGS = {"standing", "sitting", "lying", "on_back", "on_side", "on_stomach", "kneeling", "squatting", "seiza",
              "wariza", "yokozuwari", "crossed_legs", "standing_on_one_leg", "hugging_own_legs", "on_one_knee",
              "knee_up", "legs_together", "legs_apart", "walking", "running", "jumping", "crawling", "climbing",
              "dancing", "ballet", "midair", "falling", "flying", "tripping", "kicking", "flying_kick", "contrapposto",
              "indian_style", "sitting_on_stairs", "against_wall", "leaning_back", "stretching", "levitation",
              "spinning", "dynamic_pose", "exercising", "dodging", "fighting_stance", "martial_arts", "upside-down"}


def banned(tag: str) -> bool:
    return tag in EXCLUDE_EXACT or any(s in tag for s in EXCLUDE_SUBSTR)


def pose_vocab(taxonomy: Path, min_tag_posts: int) -> tuple[set[str], set[str]]:
    rows = list(csv.DictReader(taxonomy.open(encoding="utf-8")))
    ok = lambda r: r["status"] == "approved" and int(r["post_count"] or 0) >= min_tag_posts and not banned(r["tag"])  # noqa: E731
    body = {r["tag"] for r in rows if ok(r) and r["slot"] in ("pose:posture", "pose:kinetic")} | COMBAT
    gesture = {r["tag"] for r in rows if ok(r) and r["slot"] == "pose:gesture"}
    body -= {"head_tilt", "looking_down", "looking_back", "arm_at_side", "arms_at_sides", "sleeping"}
    return body, gesture | CAMERA | {"head_tilt", "looking_down", "looking_back", "sleeping"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-tag-posts", type=int, default=2000)
    ap.add_argument("--min-posts", type=int, default=40)
    ap.add_argument("--max-tags", type=int, default=4)
    ap.add_argument("--eval-fraction", type=float, default=0.1)
    args = ap.parse_args()
    body, other = pose_vocab(args.taxonomy, args.min_tag_posts)
    pose = sorted(body | other)
    files = sorted(glob.glob(str(args.posts / "*.parquet")))
    df = (pl.scan_parquet(files).filter(pl.col("rating") == "g").select("tag_string_general")
          .with_columns(pl.col("tag_string_general").str.split(" ").alias("t")).select("t").collect())
    total_g = df.height
    # post level: solo, no multi-person or excluded pose tag (clothing tags are fine; substrings only filter the pose vocab)
    df = df.filter(pl.col("t").list.contains("solo")
                   & ~pl.col("t").list.eval(pl.element().is_in(sorted(MULTI | EXCLUDE_EXACT))).list.any())
    solo_clean = df.height
    combos = (df.with_columns(pl.col("t").list.eval(pl.element().filter(pl.element().is_in(pose))).list.sort().alias("p"))
              .filter((pl.col("p").list.len() >= 1) & (pl.col("p").list.len() <= args.max_tags)
                      & pl.col("p").list.eval(pl.element().is_in(sorted(body))).list.any())
              .with_columns(pl.col("p").list.join(" ").alias("k")).group_by("k").len()
              .filter(pl.col("len") >= args.min_posts).sort("len", descending=True))
    out = []
    for key, n in combos.iter_rows():
        tags = key.split(" ")
        assert not any(banned(t) for t in tags), tags
        legs = bool(set(tags) & NEEDS_LEGS)
        scene_key = "p" + hashlib.sha1(key.encode()).hexdigest()[:11]
        split = "eval" if int(scene_key[1:9], 16) / 0xFFFFFFFF < args.eval_fraction else "train"
        out.append({"scene_key": scene_key, "split": split, "danbooru_tags": tags, "tags": [anima_tag(t) for t in tags],
                    "needs_legs": legs, "posts": n, "focus": "body_pose", "rating": "safe"})
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(json.dumps({"rating_g_posts": total_g, "solo_clean": solo_clean, "pose_vocab": len(pose), "combos": len(out),
                      "train": sum(r["split"] == "train" for r in out), "needs_legs": sum(r["needs_legs"] for r in out),
                      "size_hist": {k: sum(len(r["tags"]) == k for r in out) for k in range(1, args.max_tags + 1)},
                      "top": [r["danbooru_tags"] for r in out[:12]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
