"""SnapRef: ComfyUI node for the SnapRef reference adapter for Anima (asymmetric attention + cached reference K/V).

The reference runs once per sampling run through its own stream (sigma 0, its own LoRA, RoPE frame t = ref_t,
centred on the target) and every block's self-attention K/V are kept. Each target forward then prepends those
K/V in self-attention, with the target LoRA, inside ComfyUI's own blocks, so any sampler / scheduler works.
The target LoRA is on in every target forward. By default only the positive (cond) branch reads the reference.

strength: a log-scale bias on the attention logits of the reference keys (0 = trained weight; > 0 stronger,
< 0 weaker), applied as an additive attention mask; non-zero values make sampling about 1.3x slower.

ref_guidance (switch ref_guidance_on): the reference goes into both CFG branches and a post-CFG step adds
(ref_guidance - 1) * (uncond with reference - uncond without reference), then APG (as in ElectronGirl's
ComfyUI-Anima-Ref-Adapter, Apache-2.0). One extra model call per step until ref_end.

Two characters: stitch two references side by side into one image and say who is left / right in the prompt.

Node: SnapRef (MODEL, CLIP, VAE, reference IMAGE, adapter) -> MODEL for KSampler.
Weight files keep the internal tensor format anima_ref_kv/1 (key prefix anima_ref_kv.*).
"""
from __future__ import annotations

import json
import logging

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.patcher_extension
import comfy.samplers
import comfy.utils
import folder_paths

WRAPPER_KEY = "snapref"
NO_REF_KEY = "snapref_off"   # transformer_options flag: run this call without the reference
_STATE = {"adapter": None, "stream": None}   # LoRA hooks apply only while an adapter is active
_adapters: dict[str, "_Adapter"] = {}


class _Adapter:
    def __init__(self, path: str):
        tensors = load_file(path)
        with safe_open(path, "pt") as handle:
            md = handle.metadata() or {}
            self.meta = json.loads(md.get("snapref") or md.get("anima_ref_kv") or "{}")
        if self.meta.get("format") != "anima_ref_kv/1":
            raise ValueError(f"{path} is not an anima_ref_kv/1 adapter")
        self.ref_t = int(self.meta.get("ref_t", 10))
        self.modes = list(self.meta.get("modes", ["cs"]))
        self.mode_embed = tensors.get("anima_ref_kv.mode_embed")
        self.lora: dict[str, dict[str, list]] = {}   # module name -> stream -> [down, up, scale]
        for key, value in tensors.items():
            if not key.endswith(".lora_down"):
                continue
            root = key[: -len("lora_down")]
            _, stream, name = root[:-1].split(".", 2)
            down, up = (self._dequant(tensors, root + n) for n in ("lora_down", "lora_up"))
            alpha = float(tensors[root + "alpha"])
            self.lora.setdefault(name, {})[stream] = [down, up, alpha / down.shape[0]]
        self.hooked: set[int] = set()   # id() of diffusion models carrying our hooks
        self.cache: dict = {}

    @staticmethod
    def _dequant(tensors, key):
        """bf16 matrix; FP8 files (deploy/quantize_adapter_fp8.py) carry a per-row float32 <key>_scale."""
        w = tensors[key]
        if key + "_scale" in tensors:
            w = w.float() * tensors[key + "_scale"]
        return w.to(torch.bfloat16)

    def install(self, dm) -> None:
        """Forward hooks on the targeted Linears (LoRA delta of the active stream) and the self-attention
        record / prepend patch. Both are no-ops unless this adapter is active."""
        if id(dm) in self.hooked:
            return
        self.hooked.add(id(dm))
        for name, streams in self.lora.items():
            module = dm.get_submodule(name)

            def hook(mod, args, out, _streams=streams, _me=self):
                if _STATE["adapter"] is not _me or _STATE["stream"] not in _streams:
                    return out
                w = _streams[_STATE["stream"]]
                x = args[0]
                if w[0].device != x.device or w[0].dtype != x.dtype:
                    w[0], w[1] = w[0].to(x.device, x.dtype), w[1].to(x.device, x.dtype)
                return out + F.linear(F.linear(x, w[0]), w[1]) * w[2]
            module.register_forward_hook(hook)
        for index, block in enumerate(dm.blocks):
            attn = block.self_attn
            if getattr(attn, "_ref_kv_state", "missing") != "missing":
                continue
            attn._ref_kv_state = None
            original = attn.forward

            def forward(x, context=None, rope_emb=None, transformer_options={}, _attn=attn, _i=index, _orig=original):
                state = _attn._ref_kv_state
                if state is None:
                    return _orig(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
                q, k, v = _attn.compute_qkv(x, context, rope_emb=rope_emb, transformer_options=transformer_options)
                if "record" in state:
                    state["record"][_i] = (k, v)
                elif state.get("bias", 0.0) != 0.0:
                    out = _biased_attention(q, k, state["cache"][_i], v, state["bias"], _attn.n_heads)
                    return _attn.output_dropout(_attn.output_proj(out))
                else:
                    k_r, v_r = state["cache"][_i]
                    k = torch.cat([k_r.expand(k.shape[0], *k_r.shape[1:]).to(k.dtype), k], dim=1)
                    v = torch.cat([v_r.expand(v.shape[0], *v_r.shape[1:]).to(v.dtype), v], dim=1)
                return _attn.compute_attention(q, k, v, transformer_options=transformer_options)
            attn.forward = forward


def _set_attn_state(dm, state) -> None:
    for block in dm.blocks:
        block.self_attn._ref_kv_state = state


def _biased_attention(q, k_t, kv_ref, v_t, bias: float, heads: int):
    """softmax over [K_ref + bias ; K_t]: SDPA with an additive key mask (broadcast, not materialised).
    Keys are padded to a multiple of 16 (masked out) so the memory-efficient kernel applies.
    q, k, v [B, S, H, D] -> [B, S, H * D]."""
    k_r, v_r = kv_ref
    B, S, H, D = q.shape
    k = torch.cat([k_r.expand(B, *k_r.shape[1:]).to(q.dtype), k_t], dim=1).transpose(1, 2)
    v = torch.cat([v_r.expand(B, *v_r.shape[1:]).to(q.dtype), v_t], dim=1).transpose(1, 2)
    n, r = k.shape[2], k_r.shape[1]
    pad = (-n) % 16
    if pad:
        k, v = F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
    mask = torch.zeros(1, 1, 1, n + pad, device=q.device, dtype=q.dtype)
    mask[..., :r] = bias
    if pad:
        mask[..., n:] = float("-inf")
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask.expand(B, H, S, n + pad))
    return out.transpose(1, 2).reshape(B, S, H * D)


def _rope(pe, t, h, w):
    """VideoRopePosition3DEmb for explicit per-token positions (train/anima_env.rope_from_positions)."""
    def rot(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)
    dev = t.device
    emb = torch.cat([rot(t, 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(dev))),
                     rot(h, 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(dev))),
                     rot(w, 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(dev)))], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


@torch.no_grad()
def _encode_reference(adapter: _Adapter, dm, ref, context, target_hw, mode: str):
    """train/asym.encode_reference: per-block (k, v) of the reference stream."""
    x_emb, _, _ = dm.prepare_embedded_sequence(ref, fps=None, padding_mask=None)
    _, _, hp, wp, _ = x_emb.shape
    h0 = round((target_hw[0] // 2) / 2 - hp / 2)
    w0 = round((target_hw[1] // 2) / 2 - wp / 2)
    hh, ww = torch.meshgrid(torch.arange(hp, device=ref.device) + h0, torch.arange(wp, device=ref.device) + w0, indexing="ij")
    rope = _rope(dm.pos_embedder, torch.full_like(hh, adapter.ref_t).flatten(), hh.flatten(), ww.flatten())
    t_emb, adaln = dm.t_embedder[1](dm.t_embedder[0](torch.zeros(ref.shape[0], 1, device=ref.device)).to(x_emb.dtype))
    t_emb = dm.t_embedding_norm(t_emb)
    if adapter.mode_embed is not None and mode in adapter.modes:
        t_emb = t_emb + adapter.mode_embed[adapter.modes.index(mode)].to(t_emb)[None, None, :]
    record = [None] * len(dm.blocks)
    _set_attn_state(dm, {"record": record})
    _STATE.update(adapter=adapter, stream="ref")
    try:
        x = x_emb
        for block in dm.blocks:
            x = block(x, t_emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0), adaln_lora_B_T_3D=adaln,
                      extra_per_block_pos_emb=None, transformer_options={})
    finally:
        _set_attn_state(dm, None)
        _STATE.update(adapter=None, stream=None)
    return record


def _make_wrapper(adapter: _Adapter, ref_latent, ref_cond, mode: str, in_uncond: bool, bias: float = 0.0):
    def wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        dm = executor.class_obj
        adapter.install(dm)
        options = kwargs.get("transformer_options", {})
        key = (id(dm), tuple(x.shape[-2:]), x.dtype, x.device)
        if key not in adapter.cache:
            adapter.cache.clear()   # ~0.9 GB of K/V per 1 MP reference (bf16); keep one
            cross, meta = ref_cond
            ids, weights = meta.get("t5xxl_ids"), meta.get("t5xxl_weights")
            ctx = dm.preprocess_text_embeds(cross.to(x.device, x.dtype),
                                            None if ids is None else ids.unsqueeze(0).to(x.device),
                                            None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x.device, x.dtype))
            adapter.cache[key] = _encode_reference(adapter, dm, ref_latent.to(x.device, x.dtype), ctx,
                                                   tuple(x.shape[-2:]), mode)
        cache = adapter.cache[key]
        flags = options.get("cond_or_uncond") or [0]
        if options.get(NO_REF_KEY):   # the extra reference-free call of ref_guidance
            use = [False] * len(flags)
        else:
            use = [f == 0 or in_uncond for f in flags]   # 0 = positive, 1 = negative

        def run(xs, ts, cs, with_ref, kw):
            # the "tgt" LoRA is on in every target forward, with or without the reference (as in training)
            _set_attn_state(dm, {"cache": cache, "bias": bias} if with_ref else None)
            _STATE.update(adapter=adapter, stream="tgt")
            try:
                return executor(xs, ts, cs, fps, padding_mask, **kw)
            finally:
                _set_attn_state(dm, None)
                _STATE.update(adapter=None, stream=None)
        if all(use) or not any(use):
            return run(x, timesteps, context, use[0], kwargs)
        # mixed cond/uncond batch: run the two parts separately (same token count, no extra compute)
        # shortcut: only x / timesteps / context are split; per-batch extras in kwargs (e.g. ControlNet) are not
        n = x.shape[0] // len(flags)
        outs = []
        for i, u in enumerate(use):
            sl = slice(i * n, (i + 1) * n)
            outs.append(run(x[sl], timesteps[sl], context[sl], u, kwargs))
        return torch.cat(outs, dim=0)
    return wrapper


def _make_outer(adapter: _Adapter):
    def wrapper(executor, *args, **kwargs):
        try:
            return executor(*args, **kwargs)
        finally:
            adapter.cache.clear()
    return wrapper


def _correct(guided, pos, correction, rescale_phi=0.5):
    """APG (drop the update's component parallel to the cond prediction) or CFG rescale, in x0 space
    (as ElectronGirl/ComfyUI-Anima-Ref-Adapter, Apache-2.0)."""
    dims = tuple(range(1, pos.ndim))
    if correction == "apg":
        update = (guided - pos).double()
        unit = F.normalize(pos.double().flatten(1), dim=1).view_as(pos)
        return pos + (update - (update * unit).sum(dim=dims, keepdim=True) * unit).to(pos.dtype)
    if correction == "rescale":
        ratio = pos.std(dim=dims, keepdim=True) / guided.std(dim=dims, keepdim=True)
        return rescale_phi * guided * ratio + (1 - rescale_phi) * guided
    return guided


def _make_ref_guidance(ref_guidance: float, correction: str, ref_end: float):
    """Post-CFG: guided = base + rg * (uncond_ref - base) + cfg * (cond_ref - uncond_ref), base = uncond without
    the reference (one extra model call per step, until ref_end). KSampler's CFG already gives
    uncond_ref + cfg * (cond_ref - uncond_ref), so add (rg - 1) * (uncond_ref - base)."""
    def post_cfg(args):
        guided, pos = args["denoised"], args["cond_denoised"]
        ms = args["model"].model_sampling
        if ref_end >= 1.0 or float(args["sigma"].flatten()[0]) >= ms.percent_to_sigma(ref_end) - 1e-5:
            cond, ref = (args["uncond"], args["uncond_denoised"]) if args["uncond"] is not None else (args["cond"], pos)
            options = args["model_options"].copy()
            options["transformer_options"] = {**options.get("transformer_options", {}), NO_REF_KEY: True}
            (base,) = comfy.samplers.calc_cond_batch(args["model"], [cond], args["input"], args["sigma"], options)
            guided = guided + (ref_guidance - 1.0) * (ref - base)
        return _correct(guided, pos, correction)
    return post_cfg


def _adapter_files():
    names = folder_paths.get_filename_list("loras")
    return [n for n in names if "snapref" in n.lower() or "anima_ref_kv" in n] or names


class SnapRef:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP", {"tooltip": "encodes ref_caption for the reference stream"}),
            "vae": ("VAE",),
            "image": ("IMAGE", {"tooltip": "the reference (first image of the batch)"}),
            "adapter": (_adapter_files(),),
            "mode": (["cs", "char"], {"tooltip": "cs = character and style from the reference; "
                                                 "char = character only (put artist tags in the prompt)"}),
            "ref_in_uncond": ("BOOLEAN", {"default": False, "tooltip": "off (trained default): only the positive "
                                          "branch reads the reference, so CFG pushes toward it. on: weaker, closer to the prompt"}),
            "ref_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05,
                                         "tooltip": "the reference is shrunk (never enlarged) to this area, aspect kept; trained at <= 1.0"}),
        }, "optional": {
            "strength": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1,
                                   "tooltip": "attention to the reference, log scale. 0 = as trained; +1 = about 2.7x more, "
                                              "-1 = about 2.7x less. Non-zero values make sampling about 1.3x slower"}),
            "ref_caption": ("STRING", {"multiline": True, "default": "",
                                       "tooltip": "optional appearance tags of the reference (gender, solo, hair, eyes, outfit); "
                                                  "no name or artist. Empty works."}),
            "ref_guidance_on": ("BOOLEAN", {"default": False, "tooltip": "separate reference guidance (one extra model call "
                                            "per step until ref_end). Puts the reference in both CFG branches and pushes "
                                            "from 'no reference' toward it by ref_guidance; ignores ref_in_uncond"}),
            "ref_guidance": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 15.0, "step": 0.1,
                                       "tooltip": "reference strength, separate from cfg. Equal to cfg = the same strength as "
                                                  "with ref_guidance off (plus APG); above cfg = closer to the reference, "
                                                  "below = closer to the prompt; 1 = no push"}),
            "correction": (["apg", "rescale", "none"], {"tooltip": "keeps colors in check when ref_guidance is on"}),
            "ref_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                  "tooltip": "last step with the extra call, as a fraction of the denoising; ~0.6 is faster"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "SnapRef"

    def apply(self, model, clip, vae, image, adapter, mode, ref_in_uncond, ref_megapixels, strength=0.0, ref_caption="",
              ref_guidance_on=False, ref_guidance=4.5, correction="apg", ref_end=1.0):
        path = folder_paths.get_full_path_or_raise("loras", adapter)
        if path not in _adapters:
            _adapters[path] = _Adapter(path)
        ad = _adapters[path]
        pixels = image[:1].movedim(-1, 1)
        h, w = pixels.shape[-2:]
        s = min(1.0, (ref_megapixels * 1024 * 1024 / (w * h)) ** 0.5)
        width, height = max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)
        pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
        latent = vae.encode(pixels[..., :3])
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        latent = model.get_model_object("process_latent_in")(latent)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(ref_caption.strip()))[0]
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
                               _make_wrapper(ad, latent, cond, mode, ref_in_uncond or ref_guidance_on, strength))
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, _make_outer(ad))
        if ref_guidance_on:
            m.set_model_sampler_post_cfg_function(_make_ref_guidance(ref_guidance, correction, ref_end))
        logging.info(f"[SnapRef] {adapter} step {ad.meta.get('step')}, reference {width}x{height}, mode {mode}, "
                     f"strength {strength}, ref_guidance {ref_guidance if ref_guidance_on else 'off'}")
        return (m,)


NODE_CLASS_MAPPINGS = {"SnapRef": SnapRef}
NODE_DISPLAY_NAME_MAPPINGS = {"SnapRef": "SnapRef"}
