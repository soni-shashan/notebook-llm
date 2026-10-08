"""Static lookup tables (kept separate so gpu.py and estimate.py don't import each other)."""

# (substring, memory bandwidth GB/s). First match wins -> keep specific names first.
_GPU_BW = [
    ("b200", 8000), ("h200", 4800), ("h100", 2400),
    ("a100", 1555), ("l40", 864), ("a6000", 768), ("a5000", 768), ("a4000", 448),
    ("a10g", 600), ("a10", 600), ("v100", 900), ("p100", 732), ("l4", 300), ("t4", 320),
    ("rtx 4090", 1008), ("rtx 4080", 717), ("rtx 4070 ti", 504), ("rtx 4070", 504),
    ("rtx 3090", 936), ("rtx 3080", 760), ("rtx 3070", 448), ("rtx 3060", 360),
    ("rtx 2080 ti", 616), ("rtx 2080", 448), ("rtx 2070", 448), ("gtx 1080 ti", 484),
    ("p4000", 243), ("p40", 346), ("k80", 240),
]


def gpu_bandwidth(name: str):
    """Returns (GB/s, known?)."""
    n = name.lower()
    if "a100" in n and "80gb" in n:
        return 2039, True
    for key, bw in _GPU_BW:
        if key in n:
            return bw, True
    return 300, False


# MoE models: "name:tag" -> (total params B, active params B)
MOE = {
    "qwen3-coder:30b": (30.5, 3.3), "qwen3:30b": (30.5, 3.3), "qwen3:235b": (235, 22),
    "qwen3-vl:30b": (31, 3.3), "gpt-oss:20b": (21, 3.6), "gpt-oss:120b": (117, 5.1),
    "mixtral:8x7b": (46.7, 12.9), "mixtral:8x22b": (141, 39),
    "deepseek-coder-v2:16b": (15.7, 2.4), "deepseek-coder-v2:236b": (236, 21),
    "deepseek-v3:671b": (671, 37), "deepseek-r1:671b": (671, 37),
    "llama4:scout": (109, 17), "llama4:maverick": (400, 17),
}

# Approx bytes per parameter for common GGUF quantizations.
BYTES_PER_PARAM = [("fp16", 2.0), ("f16", 2.0), ("q8", 1.07), ("q6", 0.82), ("q5", 0.72),
                   ("q4", 0.61), ("q3", 0.48), ("q2", 0.37)]
