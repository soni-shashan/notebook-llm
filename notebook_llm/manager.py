"""NotebookLLM: the object the CLI (and notebook users) drive."""
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from . import installer
from .client import OllamaClient
from .config import continue_config
from .disk import DiskSpaceError, SpaceCheck, check_space, default_models_dir, pick_models_dir
from .env import is_kaggle, log
from .gpu import detect_hardware, detect_gpus
from .integrations import render, render_all
from .library import Candidate, make_candidate, registry_size_gb
from .server import OllamaServer
from .tunnel import CloudflareTunnel

CONFIG_PATH = os.path.expanduser("~/.notebook_llm/config.json")

# Curated list for "recommended for my GPU". Coding models first (they are auto-picked first).
CURATED = [
    "qwen3-coder:30b", "qwen2.5-coder:32b", "qwen2.5-coder:14b", "qwen2.5-coder:7b",
    "qwen2.5-coder:3b", "qwen2.5-coder:1.5b",
    "gpt-oss:20b", "qwen3:30b", "qwen3:14b", "qwen3:8b", "qwen3:4b",
    "gemma3:27b", "gemma3:12b", "gemma3:4b", "deepseek-r1:32b", "deepseek-r1:14b", "deepseek-r1:8b",
    "phi4:14b", "llama3.1:8b", "llama3.2:3b", "mistral:7b",
]


@dataclass
class Settings:
    model: Optional[str] = None
    context: Optional[int] = None       # None = auto from VRAM
    keep_alive: str = "-1"
    flash_attention: bool = True
    kv_cache: str = "q8_0"              # f16 | q8_0 | q4_0 (needs flash attention)
    port: int = 11434
    models_dir: Optional[str] = None    # None = auto (the disk with most free space)

    @classmethod
    def load(cls) -> "Settings":
        try:
            with open(CONFIG_PATH) as f:
                return cls(**{k: v for k, v in json.load(f).items() if k in cls.__dataclass_fields__})
        except (OSError, ValueError):
            return cls()

    def save(self) -> None:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)


class NotebookLLM:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings.load()
        self.hw = detect_hardware()
        self.client = OllamaClient(f"http://127.0.0.1:{self.settings.port}")
        self.tunnel: Optional[CloudflareTunnel] = CloudflareTunnel.discover(
            installer.cloudflared_path() or "cloudflared", f"http://127.0.0.1:{self.settings.port}")
        self.server = self._make_server()

    # ---- hardware / estimates -------------------------------------
    def refresh_hardware(self) -> None:
        self.hw.gpu = detect_gpus()

    def candidate(self, ref: str, online: bool = False) -> Candidate:
        return make_candidate(ref, self.hw, self.settings.context, self.settings.kv_cache, online)

    def recommended(self) -> List[Candidate]:
        return [self.candidate(r) for r in CURATED]

    def best_fit(self) -> Candidate:
        """First curated coder model that fits comfortably; else the smallest one."""
        cands = self.recommended()
        for c in cands[:6]:
            if c.est.verdict == "fits":
                return c
        return cands[5]

    # ---- server ----------------------------------------------------
    def _make_server(self, model: Optional[str] = None) -> OllamaServer:
        s = self.settings
        ctx = s.context
        ref = model or s.model
        if ctx is None and ref:
            ctx = self.candidate(ref).est.context
        return OllamaServer(port=s.port, gpu_indices=self.hw.gpu.indices, keep_alive=s.keep_alive,
                            context_length=ctx, flash_attention=s.flash_attention, kv_cache_type=s.kv_cache,
                            models_dir=s.models_dir)

    def ensure_installed(self) -> None:
        installer.install_ollama()

    @property
    def models_dir(self) -> str:
        return self.settings.models_dir or default_models_dir()

    def space_for(self, size_gb: float) -> SpaceCheck:
        return check_space(size_gb, self.models_dir)

    def start_server(self, model: Optional[str] = None) -> None:
        self.ensure_installed()
        if not self.settings.models_dir and not self.server.is_running():
            self.settings.models_dir = pick_models_dir()
            self.settings.save()
        self.server = self._make_server(model)
        self.server.start()

    def restart_server(self, model: Optional[str] = None) -> None:
        self.server.stop()
        self.start_server(model)

    def stop_server(self) -> None:
        self.server.stop()

    # ---- models ----------------------------------------------------
    def installed(self) -> List[Dict]:
        return self.client.list_models() if self.server.is_running() else []

    def loaded(self) -> List[Dict]:
        return self.client.running() if self.server.is_running() else []

    def pull(self, ref: str, size_gb: Optional[float] = None, force: bool = False) -> None:
        """Download a model. Raises DiskSpaceError if the disk is too small (unless force=True)."""
        if size_gb is None:
            name, _, tag = ref.partition(":")
            size_gb = registry_size_gb(name, tag or "latest")
        if size_gb and not force:
            chk = self.space_for(size_gb)
            if not chk.ok:
                raise DiskSpaceError(f"Not enough disk space in {chk.path}: {chk.free_gb:.1f} GB free, "
                                     f"need ~{chk.needed_gb:.1f} GB for {ref}.")
        self.client.pull(ref)

    def load(self, ref: str) -> None:
        """Make `ref` the active model. Restarts the server if its context should change."""
        want = self.candidate(ref).est.context
        if self.settings.context is None and self.server.context_length not in (None, want) and self.server.is_running():
            log(f"Restarting server with context {want} for this model...")
            self.restart_server(ref)
        self.client.load(ref, keep_alive=-1 if self.settings.keep_alive == "-1" else self.settings.keep_alive)
        self.settings.model = ref
        self.settings.save()

    def unload(self, ref: str) -> None:
        self.client.unload(ref)

    def delete(self, ref: str) -> None:
        self.client.delete(ref)
        if self.settings.model == ref:
            self.settings.model = None
            self.settings.save()

    def benchmark(self, ref: Optional[str] = None) -> Dict[str, float]:
        return self.client.benchmark(ref or self.settings.model)

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        return self.client.chat(self.settings.model, prompt, system)

    # ---- tunnel ----------------------------------------------------
    @property
    def url(self) -> Optional[str]:
        return self.tunnel.url if self.tunnel else None

    def open_tunnel(self) -> str:
        if self.tunnel and self.tunnel.url:
            return self.tunnel.url
        self.tunnel = CloudflareTunnel(installer.install_cloudflared(), self.server.url)
        return self.tunnel.start()

    def close_tunnel(self) -> None:
        if self.tunnel:
            self.tunnel.stop()
        self.tunnel = None

    def continue_config(self, autocomplete: Optional[str] = None) -> str:
        return continue_config(self.url or self.server.url, self.settings.model or "MODEL", autocomplete)

    def connect(self, tool: Optional[str] = None) -> str:
        """Ready-to-paste settings. tool: continue|cline|aider|openwebui|claude-code|cursor|antigravity|python (None = all)."""
        url = self.url or self.server.url
        model = self.settings.model or "MODEL"
        ctx = self.server.context_length or self.settings.context or (
            self.candidate(model).est.context if self.settings.model else 8192)
        return render_all(url, model, ctx) if tool is None else render(tool, url, model, ctx)

    # ---- one call --------------------------------------------------
    def quickstart(self, model: str = "auto", tunnel: bool = True) -> Optional[str]:
        """Install -> start -> pull -> load -> tunnel. Returns the public URL."""
        self.refresh_hardware()
        log(self.hw.summary())
        c = self.best_fit() if model == "auto" else self.candidate(model)
        log(f"Model: {c.ref} ({c.est.label}, ~{c.est.tok_s:.0f} tok/s est.)" if c.est.tok_s else f"Model: {c.ref}")
        self.settings.model = c.ref
        self.start_server(c.ref)
        if not self.client.has_model(c.ref):
            self.pull(c.ref, c.size_gb)
        self.load(c.ref)
        return self.open_tunnel() if tunnel else None
