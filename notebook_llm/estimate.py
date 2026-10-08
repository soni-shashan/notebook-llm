"""Will it fit? How fast will it run? (approximate, memory-bandwidth model)"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .estimate_data import BYTES_PER_PARAM, MOE

GPU_EFF = 0.65        # fraction of peak bandwidth llama.cpp typically reaches
MOE_EFF = 0.70        # extra penalty for expert routing
CPU_BW_GBS = 15.0     # effective system-RAM bandwidth for offloaded layers
OVERHEAD_PER_GPU = 0.7  # CUDA context + compute buffers (GB)
KV_FACTORS = {"f16": 1.0, "q8_0": 0.53, "q4_0": 0.28}


@dataclass
class Estimate:
    verdict: str            # fits | tight | offload | no | cpu
    needed_gb: float
    weights_gb: float
    kv_gb: float
    overhead_gb: float
    gpu_fraction: float     # share of weights that sit in VRAM
    tok_s: Optional[float]
    context: int
    note: str = ""

    @property
    def label(self) -> str:
        return {"fits": "FITS", "tight": "TIGHT", "offload": "SLOW (CPU offload)",
                "no": "TOO BIG", "cpu": "CPU ONLY"}[self.verdict]


def parse_params(name: str, tag: str) -> Tuple[Optional[float], Optional[float]]:
    """-> (total params in B, active params in B or None for dense)."""
    key = f"{name}:{tag}"
    base = f"{name}:{tag.split('-')[0]}"
    for k in (key, base):
        if k in MOE:
            return MOE[k]
    active = None
    m = re.search(r"-a(\d+(?:\.\d+)?)b", tag)
    if m:
        active = float(m.group(1))
    mx = re.match(r"(\d+)x(\d+(?:\.\d+)?)b", tag)
    if mx:
        return float(mx.group(1)) * float(mx.group(2)), active
    m = re.match(r"e?(\d+(?:\.\d+)?)([bm])", tag)
    if m:
        v = float(m.group(1)) * (0.001 if m.group(2) == "m" else 1)
        return v, active
    return None, active


def bytes_per_param(tag: str) -> float:
    t = tag.lower()
    for key, v in BYTES_PER_PARAM:
        if key in t:
            return v
    return 0.61  # Ollama default tags are ~Q4_K_M


def guess_size_gb(params_b: float, tag: str) -> float:
    return round(params_b * bytes_per_param(tag), 2)


def kv_cache_gb(params_b: float, ctx: int, kv_type: str = "f16") -> float:
    per_token_mb = 0.03 * (max(params_b, 0.3) ** 0.85) * KV_FACTORS.get(kv_type, 1.0)
    return per_token_mb * ctx / 1024


def _overhead(hw) -> float:
    return OVERHEAD_PER_GPU * max(1, hw.gpu.count)


def auto_context(size_gb: float, params_b: float, hw, kv_type: str = "q8_0") -> int:
    if not hw.gpu.available:
        return 4096
    room = hw.gpu.total_vram_gb * 0.92 - size_gb - _overhead(hw)
    for ctx in (32768, 16384, 8192, 4096):
        if kv_cache_gb(params_b, ctx, kv_type) <= 0.7 * room:
            return ctx
    return 4096


def estimate(size_gb: float, params_b: Optional[float], active_b: Optional[float],
             hw, ctx: int, kv_type: str = "q8_0") -> Estimate:
    params = params_b or size_gb / 0.61
    kv = kv_cache_gb(params, ctx, kv_type)
    moe = bool(active_b and params and active_b < params * 0.8)
    active_bytes = size_gb * (active_b / params) if moe else size_gb
    eff_boost = MOE_EFF if moe else 1.0
    gpu = hw.gpu

    if not gpu.available:
        needed = size_gb + kv + 0.5
        t = active_bytes / (CPU_BW_GBS * eff_boost) + 0.003
        verdict = "cpu" if needed <= hw.ram_gb * 0.8 else "no"
        return Estimate(verdict, needed, size_gb, kv, 0.5, 0.0, 1 / t if verdict == "cpu" else None,
                        ctx, "No GPU found: running from system RAM is slow.")

    overhead = _overhead(hw)
    needed = size_gb + kv + overhead
    vram = gpu.total_vram_gb
    note = ""
    if needed <= vram * 0.92:
        verdict = "fits"
    elif needed <= vram:
        verdict, note = "tight", "Barely fits: lower context or use a smaller quant if you hit OOM."
    elif needed <= vram + hw.ram_gb * 0.7:
        verdict, note = "offload", "Part of the model spills to system RAM, so speed drops a lot."
    else:
        verdict, note = "no", "Not enough VRAM + RAM."

    weights_on_gpu = max(0.0, min(size_gb, vram * 0.95 - overhead - kv))
    frac = weights_on_gpu / size_gb if size_gb else 1.0
    if verdict == "no":
        tok = None
    else:
        t = 0.0
        for g in gpu.gpus:  # layers split by VRAM share; GPUs run in sequence
            share = g.memory_total_mb / (vram * 1024)
            t += active_bytes * frac * share / (g.bandwidth_gbs * GPU_EFF * eff_boost)
        t += active_bytes * (1 - frac) / (CPU_BW_GBS * eff_boost)
        t += 0.002 + 0.001 * (gpu.count - 1)
        tok = 1 / t
    if tok is not None and not all(g.bandwidth_known for g in gpu.gpus):
        note = (note + " GPU model unknown: speed is a rough guess.").strip()
    return Estimate(verdict, needed, size_gb, kv, overhead, frac, tok, ctx, note)
