"""Hardware detection: NVIDIA GPUs (nvidia-smi), system RAM, bandwidth lookup."""
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

from .estimate_data import gpu_bandwidth

_QUERY = "index,name,memory.total,memory.free,driver_version"


@dataclass
class GPU:
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int

    @property
    def vram_gb(self) -> float:
        return self.memory_total_mb / 1024

    @property
    def bandwidth_gbs(self) -> float:
        return gpu_bandwidth(self.name)[0]

    @property
    def bandwidth_known(self) -> bool:
        return gpu_bandwidth(self.name)[1]


@dataclass
class GPUInfo:
    gpus: List[GPU] = field(default_factory=list)
    driver: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.gpus)

    @property
    def available(self) -> bool:
        return self.count > 0

    @property
    def total_vram_gb(self) -> float:
        return round(sum(g.memory_total_mb for g in self.gpus) / 1024, 1)

    @property
    def used_vram_gb(self) -> float:
        return round(sum(g.memory_total_mb - g.memory_free_mb for g in self.gpus) / 1024, 1)

    @property
    def indices(self) -> List[int]:
        return [g.index for g in self.gpus]

    def summary(self) -> str:
        if not self.available:
            return "No NVIDIA GPU detected (CPU mode)"
        names = {}
        for g in self.gpus:
            names[g.name] = names.get(g.name, 0) + 1
        label = " + ".join(f"{n}x {k}" for k, n in names.items())
        return f"{label} | {self.total_vram_gb} GB VRAM"


@dataclass
class Hardware:
    gpu: GPUInfo
    ram_gb: float
    cpu_cores: int

    def summary(self) -> str:
        return f"{self.gpu.summary()} | {self.ram_gb:.0f} GB RAM | {self.cpu_cores} CPU cores"


def parse_nvidia_smi(output: str) -> GPUInfo:
    gpus, driver = [], None
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(GPU(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3]))))
            driver = parts[4]
        except ValueError:
            continue
    return GPUInfo(gpus, driver)


def detect_gpus() -> GPUInfo:
    if not shutil.which("nvidia-smi"):
        return GPUInfo()
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return GPUInfo()
    return parse_nvidia_smi(out)


def system_ram_gb() -> float:
    total = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total = int(line.split()[1]) / 1024 / 1024
                    break
    except OSError:
        pass
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                lim = int(v) / 1024 ** 3
                if 0 < lim < (total or 1e9):
                    total = lim
        except OSError:
            pass
    return round(total, 1)


def detect_hardware() -> Hardware:
    return Hardware(detect_gpus(), system_ram_gb(), os.cpu_count() or 1)
