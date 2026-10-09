"""Manage the background `ollama serve` process."""
import os
import signal
import subprocess
import time
from typing import Dict, List, Optional

import requests

LOG_PATH = "/tmp/ollama.log"


class OllamaServer:
    def __init__(self, port: int = 11434, gpu_indices: Optional[List[int]] = None,
                 keep_alive: str = "-1", context_length: Optional[int] = None,
                 flash_attention: bool = True, kv_cache_type: Optional[str] = "q8_0",
                 extra_env: Optional[Dict[str, str]] = None, host: str = "127.0.0.1",
                 models_dir: Optional[str] = None):
        self.host, self.port = host, port
        self.gpu_indices = gpu_indices or []
        self.keep_alive = keep_alive
        self.context_length = context_length
        self.flash_attention = flash_attention
        self.kv_cache_type = kv_cache_type
        self.extra_env = extra_env or {}
        self.models_dir = models_dir
        self.process: Optional[subprocess.Popen] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def _env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        env["OLLAMA_KEEP_ALIVE"] = self.keep_alive
        if self.models_dir:
            os.makedirs(self.models_dir, exist_ok=True)
            env["OLLAMA_MODELS"] = self.models_dir
        if self.gpu_indices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_indices))
            if len(self.gpu_indices) > 1:
                env["OLLAMA_SCHED_SPREAD"] = "1"
        if self.context_length:
            env["OLLAMA_CONTEXT_LENGTH"] = str(self.context_length)
        if self.flash_attention:
            env["OLLAMA_FLASH_ATTENTION"] = "1"
            if self.kv_cache_type and self.kv_cache_type != "f16":
                env["OLLAMA_KV_CACHE_TYPE"] = self.kv_cache_type
        env.update(self.extra_env)
        return env

    def start(self, timeout: int = 90) -> None:
        if self.is_running():
            return
        with open(LOG_PATH, "w") as lf:
            self.process = subprocess.Popen(
                ["ollama", "serve"], env=self._env(), stdout=lf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"ollama exited early:\n{self.log_tail()}")
            time.sleep(1)
        raise TimeoutError(f"Ollama not ready after {timeout}s:\n{self.log_tail()}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:  # started by an earlier CLI session
            subprocess.run(["pkill", "-f", "ollama serve"], check=False)
        self.process = None
        for _ in range(15):
            if not self.is_running():
                break
            time.sleep(1)

    def restart(self) -> None:
        self.stop()
        self.start()

    def log_tail(self, n: int = 30) -> str:
        try:
            with open(LOG_PATH) as f:
                return "".join(f.readlines()[-n:])
        except FileNotFoundError:
            return ""

    def gpu_report(self) -> str:
        try:
            with open(LOG_PATH) as f:
                lines = [l.strip() for l in f if any(k in l.lower() for k in ("inference compute", "library=", "cuda"))]
        except FileNotFoundError:
            return ""
        return "\n".join(lines[-6:])
