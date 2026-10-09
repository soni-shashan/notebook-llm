"""Disk-space checks for model downloads."""
import os
import shutil
from dataclasses import dataclass


class DiskSpaceError(RuntimeError):
    pass


@dataclass
class SpaceCheck:
    ok: bool
    free_gb: float
    needed_gb: float
    path: str


def default_models_dir() -> str:
    return os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")


def _existing_parent(path: str) -> str:
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def free_gb(path: str) -> float:
    return shutil.disk_usage(_existing_parent(path)).free / 1024 ** 3


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1024 ** 3


def has_models(path: str) -> bool:
    blobs = os.path.join(path, "blobs")
    return os.path.isdir(blobs) and any(True for _ in os.scandir(blobs))


def candidate_dirs():
    out = [default_models_dir()]
    for base in ("/kaggle/temp", "/tmp"):
        if os.path.isdir(base):
            out.append(os.path.join(base, "ollama_models"))
    return out


def pick_models_dir() -> str:
    """Keep the default if it already holds models; otherwise use the disk with the most free space."""
    default = default_models_dir()
    if has_models(default):
        return default
    best = max(candidate_dirs(), key=free_gb)
    return best if free_gb(best) > free_gb(default) + 2 else default


def check_space(size_gb: float, path: str) -> SpaceCheck:
    needed = size_gb * 1.05 + 1.0     # download + 5% + 1 GB headroom
    free = free_gb(path)
    return SpaceCheck(free >= needed, free, needed, path)
