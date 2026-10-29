"""Asymmetric reference attention for the Anima adapter (v1 of the K/V-cache design).

The reference stream runs once at sigma 0 and only attends to itself (and, through cross-attention, to
its own text context). Every block's self-attention K/V of the reference are kept; the target stream
reads [K_ref; K_t] in self-attention and is otherwise ComfyUI's own block. At inference the reference
K/V are computed once per generation and reused for every step (and both CFG branches if wanted).

Separate LoRA per stream: the "ref" LoRA is active only while the reference stream runs, the "tgt" LoRA
only in the target stream. Both wrap the same base Linear (DualLoRALinear, switched by MODE.stream).

RoPE: the target keeps ComfyUI's own positions (frame t = 0, h, w from 0). Reference tokens get t = REF_T
and (h, w) = local grid + (target centre - reference centre) + offset. Training draws the offset per axis
uniformly in [-max, max] patches; inference uses 0. Only relative positions matter.

Built on ComfyUI's Anima modules (comfy/ldm/cosmos/predict2.py): Block.forward calls
self.self_attn(x, None, rope_emb=..., transformer_options=...), and Attention.forward is
compute_qkv + compute_attention. We replace each block's self_attn.forward with the same two calls plus
the record/prepend step, so adaLN, cross-attention and MLP stay ComfyUI's code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import anima_env

REF_T = 10
SELF_ATTN_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj")
MLP_TARGETS = ("mlp.layer1", "mlp.layer2")
CROSS_TARGETS = ("cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.output_proj")


class _Mode:
    stream = "tgt"  # "ref" while the reference stream runs, "tgt" otherwise


MODE = _Mode()


class DualLoRALinear(nn.Module):
    """Base Linear plus up to two LoRAs; the one named MODE.stream is applied."""

    def __init__(self, base: nn.Linear, ranks: dict[str, int], alphas: dict[str, float]):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        for name, rank in ranks.items():
            if not rank:
                continue
            ad = nn.Module()
            ad.lora_down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
            ad.lora_up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
            nn.init.kaiming_uniform_(ad.lora_down, a=math.sqrt(5))
            ad.scale = alphas[name] / rank
            ad.alpha = alphas[name]
            self.adapters[name] = ad
        self.enabled = True

    def _linear(self):
        m = self.base
        while not isinstance(m, nn.Linear) and hasattr(m, "base"):
            m = m.base
        return m

    @property
    def in_features(self):  # other engines (e.g. Role-Ref-KV's) read these from the wrapped Linear
        return self._linear().in_features

    @property
    def out_features(self):
        return self._linear().out_features

    def forward(self, x):
        out = self.base(x)
        if not self.enabled or MODE.stream not in self.adapters:
            return out
        ad = self.adapters[MODE.stream]
        return out + F.linear(F.linear(x, ad.lora_down.to(x.dtype)), ad.lora_up.to(x.dtype)) * ad.scale


def inject(dm, cfg: dict) -> dict[str, DualLoRALinear]:
    """cfg: {"ref": {"rank", "alpha", "targets"}, "tgt": {...}}; a module gets both LoRAs if listed in both."""
    wanted = {}
    for stream in ("ref", "tgt"):
        spec = cfg.get(stream) or {}
        if not spec.get("rank"):
            continue
        for target in spec["targets"]:
            wanted.setdefault(target, {})[stream] = spec
    modules = {}
    for index, block in enumerate(dm.blocks):
        for target, streams in wanted.items():
            parent_name, attr = target.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            mod = DualLoRALinear(getattr(parent, attr), {s: v["rank"] for s, v in streams.items()},
                                 {s: float(v["alpha"]) for s, v in streams.items()})
            setattr(parent, attr, mod)
            modules[f"blocks.{index}.{target}"] = mod
    return modules


def lora_params(modules: dict[str, DualLoRALinear], stream: str | None = None) -> list[nn.Parameter]:
    return [p for m in modules.values() for name, ad in m.adapters.items() if stream in (None, name)
            for p in (ad.lora_down, ad.lora_up)]


def set_enabled(modules: dict[str, DualLoRALinear], enabled: bool) -> None:
    for m in modules.values():
        m.enabled = enabled


# ---------------------------------------------------------------- attention hooks

def _install_hooks(dm) -> None:
    """Wrap each block's self_attn.forward once. Its _asym_state decides what happens:
    {"record": list} stores (k, v) after norm + RoPE; {"cache": list} prepends the cached (k, v)."""
    for index, block in enumerate(dm.blocks):
        attn = block.self_attn
        if getattr(attn, "_asym_installed", False):
            continue
        attn._asym_installed = True
        attn._asym_index = index
        attn._asym_state = None

        def forward(x, context=None, rope_emb=None, transformer_options={}, _attn=attn):
            q, k, v = _attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
            state = _attn._asym_state
            if state is not None and "record" in state:
                state["record"][_attn._asym_index] = (k, v)
            elif state is not None and state.get("cache") is not None:
                k_ref, v_ref = state["cache"][_attn._asym_index]
                if k_ref.shape[0] != k.shape[0]:
                    k_ref = k_ref.expand(k.shape[0], *k_ref.shape[1:])
                    v_ref = v_ref.expand(v.shape[0], *v_ref.shape[1:])
                k = torch.cat([k_ref.to(k.dtype), k], dim=1)
                v = torch.cat([v_ref.to(v.dtype), v], dim=1)
            return _attn.compute_attention(q, k, v, transformer_options=transformer_options)

        attn.forward = forward


def _block_fn(block, stream: str, state):
    """Block call that sets the stream flag and cache state inside, so gradient-checkpoint recomputation
    in backward replays exactly the same computation."""
    def fn(x, t, ctx, rope, adaln):
        prev_stream, prev_state = MODE.stream, block.self_attn._asym_state
        MODE.stream, block.self_attn._asym_state = stream, state
        try:
            return block(x, t, ctx, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None,
                         transformer_options={})
        finally:
            MODE.stream, block.self_attn._asym_state = prev_stream, prev_state
    return fn


def _run_blocks(dm, x_emb, t_emb, adaln, context, rope, checkpoint_blocks: int, stream: str, state):
    rope = rope.unsqueeze(1).unsqueeze(0)
    for index, block in enumerate(dm.blocks):
        fn = _block_fn(block, stream, state)
        if index < checkpoint_blocks and torch.is_grad_enabled():
            x_emb = torch.utils.checkpoint.checkpoint(fn, x_emb, t_emb, context, rope, adaln, use_reentrant=False)
        else:
            x_emb = fn(x_emb, t_emb, context, rope, adaln)
    return x_emb


def _positions(hp: int, wp: int, t: float, h0: int, w0: int, device) -> tuple[torch.Tensor, ...]:
    hh, ww = torch.meshgrid(torch.arange(hp, device=device) + h0, torch.arange(wp, device=device) + w0, indexing="ij")
    return torch.full_like(hh, int(t)).flatten(), hh.flatten(), ww.flatten()


def encode_reference(dm, ref: torch.Tensor, context: torch.Tensor, target_hw: tuple[int, int],
                     offset: tuple[int, int] = (0, 0), checkpoint_blocks: int = 0,
                     mode_emb: torch.Tensor | None = None, t_index: int = REF_T) -> list:
    """Reference K/V per block. ref [B, 16, 1, h, w] model-space latent, context [B, L, 1024] the reference
    stream's text, target_hw (H, W) of the target latent. Returns [(k, v)] per block, each [B, N_ref, heads, dim]."""
    _install_hooks(dm)
    x_emb, _, extra = dm.prepare_embedded_sequence(ref, fps=None, padding_mask=None)
    if extra is not None:
        raise NotImplementedError("extra per-block position embeddings are not supported")
    _, _, hp, wp, _ = x_emb.shape
    h0 = round((target_hw[0] // 2) / 2 - hp / 2) + int(offset[0])
    w0 = round((target_hw[1] // 2) / 2 - wp / 2) + int(offset[1])
    rope = anima_env.rope_from_positions(dm.pos_embedder, *_positions(hp, wp, t_index, h0, w0, ref.device))
    sigma0 = torch.zeros(ref.shape[0], 1, device=ref.device, dtype=torch.float32)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](sigma0).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    if mode_emb is not None:  # reference mode (character & style / character only), on the adaLN input
        t_emb = t_emb + mode_emb.to(t_emb.dtype)[:, None, :]
    record = [None] * len(dm.blocks)
    # the last block's attention output is not needed; running it keeps the code simple (cost ~1/28)
    _run_blocks(dm, x_emb, t_emb, adaln, context, rope, checkpoint_blocks, "ref", {"record": record})
    return record


def target_forward(dm, x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor, cache: list | None,
                   checkpoint_blocks: int = 0) -> torch.Tensor:
    """Velocity for x [B, 16, 1, H, W] reading the reference cache (None = plain target-only forward)."""
    _install_hooks(dm)
    x_emb, rope, extra = dm.prepare_embedded_sequence(x, fps=None, padding_mask=None)
    if extra is not None:
        raise NotImplementedError("extra per-block position embeddings are not supported")
    sigma = sigma.to(device=x.device, dtype=torch.float32).reshape(x.shape[0], 1)
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](sigma).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    x_emb = _run_blocks(dm, x_emb, t_emb, adaln, context, rope, checkpoint_blocks, "tgt", {"cache": cache})
    out = dm.final_layer(x_emb.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln)
    return dm.unpatchify(out)[:, :, :1]


# ---------------------------------------------------------------- save / load

def save(modules: dict[str, DualLoRALinear], path: Path, metadata: dict, extra: dict | None = None) -> None:
    """Keys anima_ref_kv.<stream>.<module>.lora_down / lora_up / alpha and anima_ref_kv.<extra name>. Not a plain
    ComfyUI LoRA on purpose: the reference-stream LoRA must not be merged into the base model."""
    tensors = {f"anima_ref_kv.{k}": v.detach().float().cpu().contiguous() for k, v in (extra or {}).items()}
    for name, module in modules.items():
        for stream, ad in module.adapters.items():
            key = f"anima_ref_kv.{stream}.{name}"
            tensors[f"{key}.lora_down"] = ad.lora_down.detach().to(torch.bfloat16).cpu().contiguous()
            tensors[f"{key}.lora_up"] = ad.lora_up.detach().to(torch.bfloat16).cpu().contiguous()
            tensors[f"{key}.alpha"] = torch.tensor(float(ad.alpha))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"anima_ref_kv": json.dumps(metadata)})


def load(modules: dict[str, DualLoRALinear], path: Path, extra: dict | None = None) -> dict:
    """Loads LoRA weights; extra maps names to tensors filled in place (e.g. {"mode_embed": E})."""
    tensors = load_file(str(path))
    with torch.no_grad():
        for k, v in (extra or {}).items():
            v.copy_(tensors[f"anima_ref_kv.{k}"])
        for name, module in modules.items():
            for stream, ad in module.adapters.items():
                key = f"anima_ref_kv.{stream}.{name}"
                ad.lora_down.copy_(tensors[f"{key}.lora_down"].float())
                ad.lora_up.copy_(tensors[f"{key}.lora_up"].float())
    from safetensors import safe_open
    with safe_open(str(path), "pt") as handle:
        return json.loads((handle.metadata() or {}).get("anima_ref_kv", "{}"))


# ---------------------------------------------------------------- sampling and self-tests

@torch.inference_mode()
def sample(dm, ctx_pos, ctx_neg, ctx_ref, shape, seed: int, steps: int, cfg: float, shift: float, device,
           ref: torch.Tensor | None, ref_in_uncond: bool = False, mode_emb: torch.Tensor | None = None) -> torch.Tensor:
    """Euler flow sampling with CFG. The reference cache is built once; by default only the positive branch
    reads it (Fidelity 1). Returns model-space x0 [1, 16, 1, H, W]."""
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=torch.float32)
    x = noise.to(device)
    cache = None
    if ref is not None:
        ctx_r = anima_env.pad_contexts([ctx_ref]).to(device, torch.bfloat16)
        cache = encode_reference(dm, ref.to(device, torch.bfloat16), ctx_r, (shape[-2], shape[-1]), mode_emb=mode_emb)
    sigmas = anima_env.flow_sigmas(steps, shift)
    pos = anima_env.pad_contexts([ctx_pos]).to(device, torch.bfloat16)
    neg = anima_env.pad_contexts([ctx_neg]).to(device, torch.bfloat16)
    for i in range(steps):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        xb = x.to(torch.bfloat16)
        s = torch.full((1,), sigma, device=device)
        v_pos = target_forward(dm, xb, s, pos, cache).float()
        v_neg = target_forward(dm, xb, s, neg, cache if ref_in_uncond else None).float()
        x = x + (sigma_next - sigma) * (v_neg + cfg * (v_pos - v_neg))
    return x


@torch.inference_mode()
def self_test(dm, device) -> dict:
    """(1) without a cache, target_forward equals ComfyUI's forward; (2) a batched cache equals per-sample ones."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(2, 16, 1, 16, 24, generator=gen).to(device, torch.bfloat16)
    ctx = torch.randn(2, 512, 1024, generator=gen).to(device, torch.bfloat16) * 0.1
    sigma = torch.tensor([0.9, 0.3], device=device)
    ours = target_forward(dm, x, sigma, ctx, None).float()
    theirs = dm(x, sigma, ctx).float()
    d1 = (ours - theirs).abs().max().item()
    scale = theirs.abs().max().item()
    if d1 > 2e-3 * max(1.0, scale):
        raise RuntimeError(f"target_forward without reference differs from ComfyUI: {d1} (scale {scale})")
    ref = torch.randn(2, 16, 1, 12, 12, generator=gen).to(device, torch.bfloat16)
    cache = encode_reference(dm, ref, ctx, (16, 24))
    single = encode_reference(dm, ref[:1], ctx[:1], (16, 24))
    d2 = max((a[0][:1].float() - b[0].float()).abs().max().item() for a, b in zip(cache, single))
    out = target_forward(dm, x, sigma, ctx, cache).float()
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite output with reference cache")
    return {"no_ref_vs_comfy": d1, "batch_vs_single_ref_k": d2, "ref_changes_output": (out - ours).abs().max().item(),
            "scale": scale}


def grad_test(dm, modules, device) -> dict:
    """LoRA gradients with and without gradient checkpointing must match (stream flags replayed correctly)."""
    gen = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(1, 16, 1, 16, 16, generator=gen).to(device, torch.bfloat16)
    ref = torch.randn(1, 16, 1, 16, 12, generator=gen).to(device, torch.bfloat16)
    ctx = torch.randn(1, 512, 1024, generator=gen).to(device, torch.bfloat16) * 0.1
    sigma = torch.tensor([0.5], device=device)
    params = lora_params(modules)
    saved = [p.detach().clone() for p in params]
    with torch.no_grad():  # lora_up starts at zero; give it values so both factors carry gradient
        for p in params:
            p.add_(torch.randn_like(p) * 1e-3)
    grads = []
    for ckpt in (0, len(dm.blocks)):
        for p in params:
            p.grad = None
        cache = encode_reference(dm, ref, ctx, (16, 16), checkpoint_blocks=ckpt)
        out = target_forward(dm, x, sigma, ctx, cache, checkpoint_blocks=ckpt)
        out.float().pow(2).mean().backward()
        # the last block's reference-stream output is unused (only its K/V are), so those LoRAs get no grad
        grads.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in params]))
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s)
            p.grad = None
    rel = ((grads[0] - grads[1]).norm() / grads[0].norm().clamp_min(1e-12)).item()
    if not rel < 2e-2:
        raise RuntimeError(f"checkpointed gradients differ: rel {rel}")
    return {"ckpt_grad_rel_diff": rel, "grad_norm": grads[0].norm().item()}
