"""Train the asymmetric (K/V-cache) Anima reference adapter on prepare_asym.py output.

    torchrun --nproc_per_node 2 train/train_asym.py --prep ~/train/asym_prep --out ~/train/runs/v1a

Per step and GPU: a batch of (target, reference) pairs sharing target size and reference size.
  * the reference latent (clean, its own size) runs through the reference stream at sigma 0 with the
    "ref" LoRA and its own text (identity caption 70% / empty 30%); every block's K/V are recorded;
  * the noised target runs through ComfyUI's blocks with the "tgt" LoRA, its self-attention reading
    [K_ref; K_t]; loss = flow-matching MSE on the target velocity;
  * modes: "cs" (another image of the same group: same character, same style; prompt without character,
    appearance and artist tags) and "char" (same character in another style; prompt names the target's
    artists, so style comes from text); a learned mode embedding is added to the reference's adaLN input;
  * reference RoPE offset uniform in [-max, max] patches per axis (inference: 0);
  * dropout per batch: no reference 10% (plus a preservation loss to the base model), empty target text 5%,
    both 5%.
Gradients of the LoRA weights are all-reduced across GPUs by hand (no DDP wrapper on the frozen model).
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anima_env  # noqa: E402
import asym  # noqa: E402


class Prep:
    def __init__(self, prep_dir: Path):
        self.dir = Path(prep_dir)
        self.index = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        ctx = load_file(str(self.dir / "contexts.safetensors"))
        self._data, self._offsets = ctx["data"], ctx["offsets"].tolist()
        self.latents = {}
        for name in self.index["files"]["latents"]:
            self.latents.update(load_file(str(self.dir / name)))

    def context(self, i: int) -> torch.Tensor:
        return self._data[self._offsets[i]:self._offsets[i + 1]]


def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    progress = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    floor = cfg["min_lr_ratio"]
    return cfg["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


class PairSampler:
    """Buckets of (target, ref) pairs keyed by (mode, target size, ref size); pairs whose images differ
    in scene type (expression / body pose) or framing get a higher weight."""

    def __init__(self, index: dict, mode_probs: dict[str, float], cross_weight: float, batch: int, seed: int,
                 pose_weight: float = 1.0):
        self.rng = random.Random(seed)
        self.batch = batch
        recs = index["records"]
        train_groups = {g: v for g, v in index["groups"].items() if v["split"] == "train"}
        by_char = collections.defaultdict(list)
        for gid, g in train_groups.items():
            by_char[g["character"]].append(gid)
        size = lambda r: (recs[r]["width"], recs[r]["height"])  # noqa: E731

        def weight(t, r):
            a, b = recs[t], recs[r]
            differs = (a["focus"] == "body_pose") != (b["focus"] == "body_pose") or a["framing"] != b["framing"]
            # pose_weight > 1: targets that are body-pose scenes are drawn more often (pose following)
            return (cross_weight if differs else 1.0) * (pose_weight if a["focus"] == "body_pose" else 1.0)

        buckets = collections.defaultdict(lambda: ([], []))
        for gid, g in train_groups.items():
            ids = g["record_ids"]
            for t in ids:
                for r in ids:
                    if r != t:
                        pairs, ws = buckets[("cs", size(t), size(r))]
                        pairs.append((t, r))
                        ws.append(weight(t, r))
                for other in by_char[g["character"]]:
                    if other == gid:
                        continue
                    for r in train_groups[other]["record_ids"]:
                        pairs, ws = buckets[("char", size(t), size(r))]
                        pairs.append((t, r))
                        ws.append(weight(t, r))
        self.modes = [m for m in mode_probs if any(k[0] == m for k in buckets)]
        self.mode_w = [mode_probs[m] for m in self.modes]
        self.keys = {m: [k for k in buckets if k[0] == m] for m in self.modes}
        self.key_w = {m: [sum(buckets[k][1]) for k in self.keys[m]] for m in self.modes}
        self.buckets = {k: (pairs, list(itertools.accumulate(ws))) for k, (pairs, ws) in buckets.items()}
        self.counts = {m: sum(len(buckets[k][0]) for k in self.keys[m]) for m in self.modes}

    def next(self):
        mode = self.rng.choices(self.modes, weights=self.mode_w)[0]
        key = self.rng.choices(self.keys[mode], weights=self.key_w[mode])[0]
        pairs, cum = self.buckets[key]
        return mode, self.rng.choices(pairs, cum_weights=cum, k=self.batch)


def spawn_preview(args, ckpt: Path, step: int) -> None:
    """Render the in-training preview for one checkpoint in a separate process on PREVIEW_GPU (default: the last
    visible GPU). It loads its own model copy (~15 GB), so a GPU with that much free memory is needed; if it
    fails, training is unaffected."""
    import subprocess
    gpu = os.environ.get("PREVIEW_GPU")
    if gpu is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        gpu = visible[-1]
    out = args.preview_out / f"{args.out.name}_{step:06d}"
    log_file = Path.home() / "logs" / "train" / f"preview_{args.out.name}_{step}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "preview.py"), "--fixed", str(args.preview_fixed), "--adapter", str(ckpt),
           "--out", str(out), "--label", f"step {step}", "--comfy-root", args.comfy_root]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE", "GROUP_RANK",
              "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
        env.pop(k, None)
    subprocess.Popen(cmd, env=env, stdout=log_file.open("w"), stderr=subprocess.STDOUT, start_new_session=True)


def all_reduce_grads(params, world: int) -> None:
    # every rank must reduce the same tensors: a microbatch without reference leaves the reference LoRA
    # (and the mode embedding) without grad on one rank only
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if world == 1:
        return
    grads = [p.grad for p in params]
    flat = torch.cat([g.flatten() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="defaults to the config in the prep index")
    ap.add_argument("--comfy-root", default=str(Path.home() / "ComfyUI"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--resume", type=Path, help="training state file to continue from")
    ap.add_argument("--init", type=Path, help="start from these adapter weights (new optimizer and schedule), "
                    "e.g. a finished run continued with a new data mix")
    ap.add_argument("--smoke", action="store_true", help="few steps, print memory and speed")
    ap.add_argument("--stop-at", type=int, help="save and exit after this step (the LR schedule still uses train.steps)")
    ap.add_argument("--preview-fixed", type=Path, default=Path.home() / "train" / "compare" / "fixed_preview", help="compare_refs --save-panels output: render our images for its "
                    "cases every train.eval_every steps (rank 0) next to the cached competitor panels")
    ap.add_argument("--preview-out", type=Path, default=Path.home() / "train" / "compare")
    args = ap.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        import datetime
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))  # rank 0 renders previews meanwhile
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    prep = Prep(args.prep)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else prep.index["config"]
    tcfg, acfg = config["train"], config["adapter"]
    if args.steps:
        tcfg["steps"] = args.steps
    if args.smoke:
        tcfg["steps"] = 30
        tcfg["log_every"] = 5
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    seed = tcfg["seed"] + 1000 * rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    anima_env.setup_comfy(args.comfy_root)
    _, dm = anima_env.load_diffusion_model(config["comfy"]["unet_name"], device)
    log(f"self-test: {asym.self_test(dm, device)}")

    modules = asym.inject(dm, acfg["lora"])
    mode_names = list(acfg["modes"])
    mode_embed = torch.nn.Parameter(torch.zeros(len(mode_names), dm.model_channels, device=device))
    params = asym.lora_params(modules) + [mode_embed]
    log(f"lora: ref {sum(p.numel() for p in asym.lora_params(modules, 'ref')) / 1e6:.1f}M, "
        f"tgt {sum(p.numel() for p in asym.lora_params(modules, 'tgt')) / 1e6:.1f}M")
    log(f"grad-test: {asym.grad_test(dm, modules, device)}")
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], betas=(0.9, 0.999), weight_decay=tcfg["weight_decay"])

    lf = anima_env.latent_format()
    latents = {k: lf.process_in(v[None].float())[0].to(torch.bfloat16) for k, v in prep.latents.items()}
    del prep.latents
    recs = prep.index["records"]
    sampler = PairSampler(prep.index, acfg["modes"], config["data"]["cross_weight"], tcfg["batch_size"], seed,
                          config["data"].get("pose_weight", 1.0))
    # multi-resolution: a share of the steps uses ~0.56 MP targets (prepare_multires.py); references stay ~1 MP
    low_p = config["data"].get("low_res_p", 0.0)
    low_latents = {}
    if low_p > 0:
        from safetensors.torch import load_file as _load
        for f in sorted(Path(config["data"]["low_res_dir"]).expanduser().glob("latents_low_*.safetensors")):
            low_latents.update({k: lf.process_in(v[None].float())[0].to(torch.bfloat16) for k, v in _load(str(f)).items()})
        log(f"low-res targets: {len(low_latents)}, p={low_p}")
    log(f"pairs per mode: {sampler.counts}, buckets {sum(len(v) for v in sampler.keys.values())}")
    mix = config["data"]["prompt_mix"]
    char_mix = config["data"]["char_prompt_mix"]
    drop = config["data"]["dropout"]
    empty_ctx = 0  # prompt index of "" (prepare_asym puts it first)
    rng = random.Random(seed + 7)
    # without a target-stream LoRA (reference-only adapter) a batch without reference trains nothing
    has_tgt = bool(asym.lora_params(modules, "tgt"))

    start_step = 0
    if args.init and not args.resume:
        asym.load(modules, args.init, {"mode_embed": mode_embed})
        log(f"initialised from {args.init}")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        asym.load(modules, Path(str(args.resume).replace("state_", "adapter_").replace(".pt", ".safetensors")),
                  {"mode_embed": mode_embed})
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        rng.setstate(state["rng"][rank])
        sampler.rng.setstate(state["sampler_rng"][rank])
        torch.set_rng_state(state["torch_rng"][rank].cpu())
        log(f"resumed at step {start_step}")

    metadata = {"format": "anima_ref_kv/1", "base": config["comfy"]["unet_name"], "ref_t": asym.REF_T,
                "modes": mode_names, "lora": acfg["lora"], "ref_text": "identity caption or empty",
                "ref_offset_max": acfg["ref_offset_max"]}
    shift, ckpt_blocks = tcfg["shift"], tcfg["checkpoint_blocks"]
    eval_every = tcfg.get("eval_every", 0)
    if eval_every and tcfg["save_every"] % eval_every and eval_every % tcfg["save_every"]:
        log("warning: eval_every and save_every do not line up; previews only run on saved steps")
    accum = tcfg.get("grad_accum", 1)
    log_path = args.out / "train_log.jsonl"
    running = collections.defaultdict(list)
    started = window = time.perf_counter()

    end_step = min(tcfg["steps"], args.stop_at or tcfg["steps"])
    for step in range(start_step, end_step):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, tcfg)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            mode, batch = sampler.next()
            use_low = low_p > 0 and rng.random() < low_p and all(t in low_latents for t, _ in batch)
            u = rng.random()
            no_ref = u < drop["ref"] + drop["both"] and has_tgt
            no_text = drop["ref"] <= u < drop["ref"] + drop["both"] + drop["text"]
            x0 = torch.stack([(low_latents if use_low else latents)[t] for t, _ in batch]).to(device)
            ctxs = []
            for t, _ in batch:
                p = recs[t]["prompts"]
                if no_text:
                    ctxs.append(prep.context(empty_ctx))
                    continue
                table = char_mix if mode == "char" else mix
                key = rng.choices(list(table), weights=list(table.values()))[0]
                ctxs.append(prep.context(p[key]))
            context = anima_env.pad_contexts(ctxs).to(device, torch.bfloat16)
            uu = torch.rand(len(batch), device=device)
            sigma = (shift * uu / (1 + (shift - 1) * uu)).clamp(1e-3, 1.0)
            noise = torch.randn_like(x0, dtype=torch.float32)
            s = sigma.view(-1, 1, 1, 1, 1)
            xt = ((1 - s) * x0.float() + s * noise).to(torch.bfloat16)
            target_v = noise - x0.float()
            cache = None
            if not no_ref:
                ref = torch.stack([latents[r] for _, r in batch]).to(device)
                rctx = [prep.context(recs[r]["prompts"]["ref_caption"]) if rng.random() < config["data"]["ref_caption_p"]
                        else prep.context(empty_ctx) for _, r in batch]
                rcontext = anima_env.pad_contexts(rctx).to(device, torch.bfloat16)
                m = acfg["ref_offset_max"]
                offset = (rng.randint(-m, m), rng.randint(-m, m))
                memb = mode_embed[mode_names.index(mode)].expand(len(batch), -1)
                cache = asym.encode_reference(dm, ref, rcontext, (x0.shape[-2], x0.shape[-1]), offset=offset,
                                              checkpoint_blocks=ckpt_blocks, mode_emb=memb)
            pred = asym.target_forward(dm, xt, sigma, context, cache, checkpoint_blocks=ckpt_blocks)
            loss = torch.nn.functional.mse_loss(pred.float(), target_v)
            total = loss
            if no_ref and config["data"]["preserve_weight"] > 0:
                with torch.no_grad():
                    asym.set_enabled(modules, False)
                    v_base = asym.target_forward(dm, xt, sigma, context, None).float()
                    asym.set_enabled(modules, True)
                total = total + config["data"]["preserve_weight"] * torch.nn.functional.mse_loss(pred.float(), v_base)
            (total / accum).backward()
            running["loss_" + ("noref" if no_ref else mode)].append(loss.item())
            if use_low:
                running["loss_low"].append(loss.item())
        all_reduce_grads(params, world)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        optimizer.step()

        done = step + 1
        if done % tcfg["log_every"] == 0 or done == tcfg["steps"]:
            now = time.perf_counter()
            n = sum(len(v) for v in running.values())
            line = {"step": done, **{k: round(sum(v) / len(v), 5) for k, v in running.items()},
                    "lr": optimizer.param_groups[0]["lr"], "grad_norm": round(float(grad_norm), 4),
                    "s_per_step": round((now - window) / max(1, n / accum), 3),
                    "elapsed_h": round((now - started) / 3600, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 1)}
            line["eta_h"] = round(line["s_per_step"] * (tcfg["steps"] - done) / 3600, 2)
            log(json.dumps(line))
            if rank == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line) + "\n")
            running, window = collections.defaultdict(list), time.perf_counter()
        if (done % tcfg["save_every"] == 0 or done == end_step) and not args.smoke:
            rngs = [None] * world
            state_local = (rng.getstate(), sampler.rng.getstate(), torch.get_rng_state())
            if distributed:
                dist.all_gather_object(rngs, state_local)
            else:
                rngs = [state_local]
            if rank == 0:
                asym.save(modules, args.out / f"adapter_{done:06d}.safetensors", {**metadata, "step": done},
                          {"mode_embed": mode_embed})
                torch.save({"step": done, "optimizer": optimizer.state_dict(), "rng": [r[0] for r in rngs],
                            "sampler_rng": [r[1] for r in rngs], "torch_rng": [r[2] for r in rngs]},
                           args.out / f"state_{done:06d}.pt")
                for old in sorted(args.out.glob("state_*.pt"))[:-1]:
                    old.unlink()  # optimizer state is large; keep only the latest
            if distributed:
                dist.barrier()
        if eval_every and done % eval_every == 0 and args.preview_fixed and not args.smoke and done < end_step:
            # a preview subprocess renders our images from the checkpoint just saved; training never waits for it
            ckpt = args.out / f"adapter_{done:06d}.safetensors"
            if rank == 0 and ckpt.exists():
                spawn_preview(args, ckpt, done)
    log(f"training done in {(time.perf_counter() - started) / 3600:.2f} h")
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
