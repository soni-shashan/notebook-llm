"""Install system deps, Ollama and cloudflared (idempotent)."""
import os
import platform
import shutil
import stat
import subprocess
import urllib.request

from .env import log

BIN_DIR = "/tmp/notebook_llm/bin"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"


def _sh(cmd: str, check: bool = True) -> int:
    return subprocess.run(cmd, shell=True, check=check,
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode


def ollama_installed() -> bool:
    return bool(shutil.which("ollama"))


def cloudflared_path():
    p = shutil.which("cloudflared")
    if p:
        return p
    local = os.path.join(BIN_DIR, "cloudflared")
    return local if os.path.exists(local) else None


def ensure_system_deps() -> None:
    """zstd is needed by the Ollama installer; pciutils/lshw let it detect GPUs."""
    missing = [p for p, b in (("zstd", "zstd"), ("pciutils", "lspci"), ("lshw", "lshw")) if not shutil.which(b)]
    if not missing:
        return
    log(f"Installing system packages: {', '.join(missing)}")
    _sh("apt-get update -qq", check=False)
    _sh(f"DEBIAN_FRONTEND=noninteractive apt-get install -y -qq {' '.join(missing)}", check=False)


def install_ollama() -> str:
    path = shutil.which("ollama")
    if path:
        return path
    ensure_system_deps()
    log("Installing Ollama (about 30s)...")
    _sh("curl -fsSL https://ollama.com/install.sh | sh")
    path = shutil.which("ollama")
    if not path:
        raise RuntimeError("Ollama install failed. Is Internet enabled in the notebook settings?")
    return path


def install_cloudflared() -> str:
    path = cloudflared_path()
    if path:
        return path
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    if not arch:
        raise RuntimeError(f"Unsupported architecture: {platform.machine()}")
    os.makedirs(BIN_DIR, exist_ok=True)
    local = os.path.join(BIN_DIR, "cloudflared")
    log("Downloading cloudflared...")
    urllib.request.urlretrieve(CLOUDFLARED_URL.format(arch=arch), local)
    os.chmod(local, os.stat(local).st_mode | stat.S_IEXEC)
    return local
