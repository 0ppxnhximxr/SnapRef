"""Reference comparison: our asymmetric adapter vs ElectronGirl pilot_r32 vs AnticPan Role-Ref-KV vs base.

    CUDA_VISIBLE_DEVICES=1 python train/compare_refs.py --prep ~/train/asym_prep --ours runs/v1a/adapter_004000.safetensors \
        --out ~/train/compare/v1a_4000 [--extra-refs ~/competitors/electrongirl/showcase/ref_girl.jpg ...]

Cases: held-out characters of the prep (reference = an expression scene of the group, target prompt = a body-pose
scene of the same group, prompt without character / appearance / artist tags), plus optional external reference
images with a fixed set of pose prompts. Same seed, size and steps for every method; each row: reference |
ours | pilot_r32 | role-ref-kv | base | generator target. Writes per-case jpgs, a contact sheet and timings.

Role-Ref-KV runs through its own released engine (comfyui/native.py) on ComfyUI's Anima modules: its own
Euler loop, shift 3, positive branch only reads the reference. Ours and pilot use Euler with the same schedule.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402
from sample import labeled_row, to_pil  # noqa: E402

POSE_PROMPTS = [
    "full body, sitting, crossed legs, looking at viewer, smile, park, bench, day",
    "full body, from behind, looking back, standing, city, street, night",
    "full body, running, outstretched arm, open mouth, beach, ocean, sunlight",
    "cowboy shot, lying, on side, closed eyes, bed, indoors",
    "full body, squatting, hand on own cheek, pout, classroom",
    "upper body, from side, profile, head tilt, light smile, cafe",
]
NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"


def load_rrkv(path: Path, comfy_dir: Path):
    spec = importlib.util.spec_from_file_location("rrkv_native", comfy_dir / "native.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import comfy.utils
    return mod, comfy.utils.load_torch_file(str(path), safe_load=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, help="prepare_asym output (eval cases); optional with --extra-refs")
    ap.add_argument("--ours", type=Path, help="our adapter (not needed with --skip ours)")
    ap.add_argument("--ours-config", type=Path, help="train config of ours (default: config.json next to the adapter)")
    ap.add_argument("--pilot", type=Path, default=Path.home() / "competitors/electrongirl/anima_ref_adapter_pilot_r32.safetensors")
    ap.add_argument("--v2", type=Path, default=Path.home() / "competitors/electrongirl/v2/anima_ref_adapter_v2_r64.safetensors")
    ap.add_argument("--v2-code", type=Path, default=Path.home() / "competitors/electrongirl/v2/anima_env_eg.py",
                    help="ElectronGirl research repo train/anima_env.py at 119677b (asym_forward, encode_ref_asym)")
    ap.add_argument("--rrkv", type=Path, default=Path.home() / "competitors/role-ref-kv/comfy/anima-role-ref-kv-v1.0.safetensors")
    ap.add_argument("--rrkv-code", type=Path, default=Path.home() / "competitors/role-ref-kv/comfyui")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--extra-refs", type=Path, nargs="*", default=[])
    ap.add_argument("--poses-per-ref", type=int, default=len(POSE_PROMPTS), help="pose prompts per extra reference")
    ap.add_argument("--cases-json", type=Path, help="take the cases from an earlier run's summary.json (e.g. the fixed preview set)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--skip", nargs="*", default=[], help="methods to skip: pilot v2 rrkv base ours")
    ap.add_argument("--save-panels", action="store_true", help="also save every method's image as <case>/<method>.jpg "
                    "(fixed set for the in-training preview)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    anima_env.setup_comfy(args.comfy_root)
    prep_index = json.loads((args.prep / "index.json").read_text(encoding="utf-8")) if args.prep else {"eval_cases": []}
    cfg_path = args.ours_config or (args.ours.parent / "config.json" if args.ours else HERE.parent / "configs" / "train_asym.json")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_model, dm = anima_env.load_diffusion_model(cfg["comfy"]["unet_name"], device)
    vae = anima_env.load_vae(cfg["comfy"]["vae_name"])
    clip = anima_env.load_text_encoder(cfg["comfy"]["clip_name"])
    lf = anima_env.latent_format()
    shift = 3.0

    ours_mods = asym.inject(dm, cfg["adapter"]["lora"])
    mode_names = list(cfg["adapter"]["modes"])
    mode_embed = torch.zeros(len(mode_names), dm.model_channels, device=device)
    meta = asym.load(ours_mods, args.ours, {"mode_embed": mode_embed}) if args.ours else {}
    asym.set_enabled(ours_mods, False)

    def enc(text):
        return anima_env.encode_prompt(clip, dm, text, device)

    def decode(x):
        return to_pil(anima_env.vae_decode(vae, lf.process_out(x.to(device)).float())[0])

    timings = {"ours": [], "pilot": [], "v2": [], "rrkv": [], "base": []}

    def run_ours(ref_lat, ctx_pos, ctx_neg, shape, seed):
        asym.set_enabled(ours_mods, True)
        t0 = time.perf_counter()
        x = asym.sample(dm, ctx_pos, ctx_neg, enc(""), shape, seed, args.steps, args.cfg, shift, device, ref_lat,
                        mode_emb=mode_embed[mode_names.index("cs")][None])
        torch.cuda.synchronize()
        timings["ours"].append(time.perf_counter() - t0)
        asym.set_enabled(ours_mods, False)
        return decode(x)

    def run_base(ctx_pos, ctx_neg, shape, seed):
        t0 = time.perf_counter()
        x = asym.sample(dm, ctx_pos, ctx_neg, enc(""), shape, seed, args.steps, args.cfg, shift, device, None)
        torch.cuda.synchronize()
        timings["base"].append(time.perf_counter() - t0)
        return decode(x)

    def add_plain_lora(path: Path, rank: int) -> dict:
        """An ElectronGirl-format LoRA on the innermost base Linears (under our DualLoRALinear and any other
        plain LoRA already added), disabled until its method runs."""
        mods = {}
        for index, block in enumerate(dm.blocks):
            for target in anima_env.LORA_TARGETS:
                parent_name, name = target.rsplit(".", 1)
                holder = block.get_submodule(parent_name)
                while hasattr(getattr(holder, name), "base"):
                    holder, name = getattr(holder, name), "base"
                lora = anima_env.LoRALinear(getattr(holder, name), rank, rank)
                setattr(holder, name, lora)
                mods[f"blocks.{index}.{target}"] = lora
        anima_env.load_lora(mods, path)
        anima_env.set_lora_enabled(mods, False)
        return mods

    pilot_mods = add_plain_lora(args.pilot, 32) if "pilot" not in args.skip and args.pilot.exists() else None

    def run_pilot(ref_raw_img, ctx_pos, ctx_neg, shape, seed):
        h, w = shape[-2] * 8, shape[-1] * 8
        ref = lf.process_in(anima_env.vae_encode_image(vae, anima_env.letterbox(ref_raw_img, w, h, 255))[None].to(device))
        anima_env.set_lora_enabled(pilot_mods, True)
        t0 = time.perf_counter()
        x = anima_env.sample_euler(dm, ctx_pos, ctx_neg, shape, seed, args.steps, args.cfg, shift, device,
                                   ref=ref.to(torch.bfloat16), ref_offset=(shape[-2] // 2, 0))
        torch.cuda.synchronize()
        timings["pilot"].append(time.perf_counter() - t0)
        anima_env.set_lora_enabled(pilot_mods, False)
        return decode(x)

    v2_mods = v2_env = None
    if "v2" not in args.skip and args.v2.exists() and args.v2_code.exists():
        spec = importlib.util.spec_from_file_location("eg_anima_env", args.v2_code)
        v2_env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_env)
        v2_mods = add_plain_lora(args.v2, 64)

    @torch.inference_mode()
    def run_v2(ref_raw_img, ctx_pos, ctx_neg, shape, seed):
        """v2 as its one-node default: reference resized to 1 MP (aspect kept), shared LoRA on both streams,
        ref_guidance 4.5 + APG in x0 space (3 model calls per step: pos+ref, neg+ref, neg on the plain base)."""
        rg = 4.5
        img = ref_raw_img.convert("RGB")
        s = (1024 * 1024 / (img.width * img.height)) ** 0.5
        img = img.resize(tuple(max(16, round(n * s / 16) * 16) for n in img.size), Image.Resampling.LANCZOS)
        lat = lf.process_in(anima_env.vae_encode_image(vae, img)[None].to(device)).to(torch.bfloat16)
        x = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32).to(device)
        ctx2 = anima_env.pad_contexts([ctx_pos, ctx_neg]).to(device, torch.bfloat16)
        neg = anima_env.pad_contexts([ctx_neg]).to(device, torch.bfloat16)
        sigmas = anima_env.flow_sigmas(args.steps, shift)
        t0 = time.perf_counter()
        anima_env.set_lora_enabled(v2_mods, True)
        cache = v2_env.encode_ref_asym(dm, lat, (-(-shape[-2] // 2), 0))
        cache = [(k.expand(2, *k.shape[1:]), v.expand(2, *v.shape[1:])) for k, v in cache]
        for i in range(args.steps):
            sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
            xb = x.to(torch.bfloat16)
            anima_env.set_lora_enabled(v2_mods, True)
            v = v2_env.asym_forward(dm, xb.expand(2, *xb.shape[1:]), torch.full((2,), sigma, device=device), ctx2,
                                    ref_kv=cache).float()
            anima_env.set_lora_enabled(v2_mods, False)
            v_base = asym.target_forward(dm, xb, torch.full((1,), sigma, device=device), neg, None).float()
            pos, neg_r, base = (x - sigma * vel for vel in (v[0:1], v[1:2], v_base))
            guided = base + rg * (neg_r - base) + args.cfg * (pos - neg_r)
            update = (guided - pos).double()  # APG: drop the update's component along the cond prediction
            unit = torch.nn.functional.normalize(pos.double().flatten(1), dim=1).view_as(pos)
            guided = pos + (update - (update * unit).sum(dim=(1, 2, 3, 4), keepdim=True) * unit).float()
            x = x + (sigma_next - sigma) * (x - guided) / sigma
        torch.cuda.synchronize()
        timings["v2"].append(time.perf_counter() - t0)
        anima_env.set_lora_enabled(v2_mods, False)
        return decode(x)

    rrkv = None
    if "rrkv" not in args.skip and args.rrkv.exists():
        native, rrkv_state = load_rrkv(args.rrkv, args.rrkv_code)
        rrkv = native.NativeEngine(dm, rrkv_state, device=device)

    def run_rrkv(ref_img, prompt_pos, prompt_neg, shape, seed):
        img = ref_img.convert("RGB")
        s = min(1.0, (768 * 768 / (img.width * img.height)) ** 0.5)
        size = tuple(max(16, int(n * s) // 16 * 16) for n in img.size)
        img = img.resize(size, Image.Resampling.LANCZOS)
        lat = lf.process_in(anima_env.vae_encode_image(vae, img)[None].to(device)).to(torch.bfloat16)

        def ctx(text):
            return anima_env.pad_contexts([enc(text)]).to(device, torch.bfloat16)
        asym.set_enabled(ours_mods, False)
        t0 = time.perf_counter()
        x = rrkv.sample(ctx(prompt_pos), ctx(prompt_neg), lat, seed=seed, width=shape[-1] * 8, height=shape[-2] * 8,
                        steps=args.steps, cfg=args.cfg)
        torch.cuda.synchronize()
        timings["rrkv"].append(time.perf_counter() - t0)
        return decode(x.float())

    cases = []
    for case in prep_index["eval_cases"][: args.cases]:
        ref_id, tgt_id = case["ref"], case["target"]
        rec = prep_index["records"][tgt_id]
        cases.append({"name": f"{case['split']}_{case['group_id']}", "ref_image": Path(prep_index["data_dir"]) / prep_index["records"][ref_id]["image"],
                      "prompt": prep_index["prompts"][rec["prompts"]["stripped"]], "negative": prep_index["prompts"][rec["prompts"]["negative"]],
                      "width": case["width"], "height": case["height"], "seed": case["seed"],
                      "truth": Path(prep_index["data_dir"]) / rec["image"], "character": case["character"]})
    quality = "masterpiece, best quality, score_7, newest, highres, absurdres, safe"
    for i, path in enumerate(args.extra_refs):
        # shortcut: gender from the file name (..._man / ..._boy -> 1boy); pass tags explicitly once refs get real names
        gender = "1boy" if re.search(r"(^|[_-])(man|boy)", path.stem) else "1girl"
        for j, pose in enumerate(POSE_PROMPTS[: args.poses_per_ref]):
            cases.append({"name": f"ext{i}_{path.stem}_{j}", "ref_image": path, "prompt": f"{quality}, {gender}, solo, {pose}",
                          "negative": NEG, "width": 832, "height": 1216, "seed": 1000 + 10 * i + j, "truth": None,
                          "character": path.stem})
    if args.cases_json:
        cases = [{**c, "width": int(c["width"]), "height": int(c["height"]), "seed": int(c["seed"])}
                 for c in json.loads(args.cases_json.read_text(encoding="utf-8"))["cases"]]

    summary = []
    for c in cases:
        shape = (1, 16, 1, c["height"] // 8, c["width"] // 8)
        ref_img = Image.open(c["ref_image"]).convert("RGB")
        ctx_pos, ctx_neg = enc(c["prompt"]), enc(c["negative"])
        s = min(1.0, (1024 * 1024 / (ref_img.width * ref_img.height)) ** 0.5)
        ours_img = ref_img.resize(tuple(max(16, int(n * s) // 16 * 16) for n in ref_img.size), Image.Resampling.LANCZOS)
        ref_lat = lf.process_in(anima_env.vae_encode_image(vae, ours_img)[None].to(device)).to(torch.bfloat16)
        named = []
        if "ours" not in args.skip:
            named.append(("ours", "OURS asym", run_ours(ref_lat, ctx_pos, ctx_neg, shape, c["seed"])))
        if pilot_mods is not None:
            named.append(("pilot", "pilot_r32 (ElectronGirl)", run_pilot(ref_img, ctx_pos, ctx_neg, shape, c["seed"])))
        if v2_mods is not None:
            named.append(("v2", "v2_r64 (ElectronGirl)", run_v2(ref_img, ctx_pos, ctx_neg, shape, c["seed"])))
        if rrkv is not None:
            named.append(("rrkv", "Role-Ref-KV (AnticPan)", run_rrkv(ref_img, c["prompt"], c["negative"], shape, c["seed"])))
        if "base" not in args.skip:
            named.append(("base", "base, no reference", run_base(ctx_pos, ctx_neg, shape, c["seed"])))
        if c["truth"] is not None and Path(c["truth"]).exists():
            named.append(("truth", "generator target (full tags)", Image.open(c["truth"]).convert("RGB")))
        if args.save_panels:
            (args.out / c["name"]).mkdir(exist_ok=True)
            ref_img.save(args.out / c["name"] / "ref.png")
            for key, _, im in named:
                im.save(args.out / c["name"] / f"{key}.jpg", quality=92)
        panels = [(f"reference ({c['character']})", ref_img)] + [(title, im) for _, title, im in named]
        row = labeled_row(panels)
        row.save(args.out / f"{c['name']}.jpg", quality=90)
        summary.append({k: (v if isinstance(v, (int, float)) or v is None else str(v)) for k, v in c.items()})
        print(f"{c['name']} done", flush=True)
    rows = [Image.open(args.out / f"{c['name']}.jpg") for c in cases]
    rows = rows[:24]
    if rows:
        width = max(r.width for r in rows)
        sheet = Image.new("RGB", (width, sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        sheet.save(args.out / "contact_sheet.jpg", quality=80)
    times = {k: round(sum(v) / len(v), 2) for k, v in timings.items() if v}
    (args.out / "summary.json").write_text(json.dumps({"adapter": str(args.ours), "meta": meta, "steps": args.steps,
                                                      "cfg": args.cfg, "seconds_per_image": times, "cases": summary},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print("seconds per image (sampling only, 1 image):", times, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
