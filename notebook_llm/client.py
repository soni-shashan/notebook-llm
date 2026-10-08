"""Thin Ollama HTTP client."""
import json
import sys
from typing import Dict, List, Optional

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base = base_url.rstrip("/")

    def list_models(self) -> List[Dict]:
        r = requests.get(f"{self.base}/api/tags", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])

    def has_model(self, name: str) -> bool:
        names = {m["name"] for m in self.list_models()}
        return name in names or f"{name}:latest" in names

    def running(self) -> List[Dict]:
        r = requests.get(f"{self.base}/api/ps", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])

    def pull(self, name: str, show_progress: bool = True) -> None:
        with requests.post(f"{self.base}/api/pull", json={"model": name, "stream": True},
                           stream=True, timeout=None) as r:
            r.raise_for_status()
            last = ""
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if "error" in d:
                    raise RuntimeError(d["error"])
                if not show_progress:
                    continue
                total, done = d.get("total"), d.get("completed")
                msg = (f"{d.get('status', '')} {done * 100 // total}% "
                       f"({done / 1e9:.1f}/{total / 1e9:.1f} GB)") if total and done is not None else d.get("status", "")
                if msg != last:
                    sys.stdout.write(f"\r  {msg:<70}")
                    sys.stdout.flush()
                    last = msg
            if show_progress:
                print()

    def delete(self, name: str) -> None:
        requests.delete(f"{self.base}/api/delete", json={"model": name}, timeout=30).raise_for_status()

    def load(self, name: str, keep_alive=-1) -> None:
        requests.post(f"{self.base}/api/generate",
                      json={"model": name, "prompt": "", "stream": False, "keep_alive": keep_alive},
                      timeout=900).raise_for_status()

    def unload(self, name: str) -> None:
        self.load(name, keep_alive=0)

    def chat(self, name: str, prompt: str, system: Optional[str] = None, keep_alive=-1) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = requests.post(f"{self.base}/api/chat",
                          json={"model": name, "messages": msgs, "stream": False, "keep_alive": keep_alive}, timeout=900)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def benchmark(self, name: str, num_predict: int = 128) -> Dict[str, float]:
        """Measure real tokens/sec using Ollama's own timing counters."""
        r = requests.post(f"{self.base}/api/generate", json={
            "model": name, "stream": False, "keep_alive": -1,
            "prompt": "Write a Python function that checks whether a number is prime, with a short explanation.",
            "options": {"num_predict": num_predict, "temperature": 0}}, timeout=900)
        r.raise_for_status()
        d = r.json()
        gen = d.get("eval_count", 0) / max(d.get("eval_duration", 1), 1) * 1e9
        pre = d.get("prompt_eval_count", 0) / max(d.get("prompt_eval_duration", 1), 1) * 1e9
        out = {"tok_s": gen, "prompt_tok_s": pre, "load_s": d.get("load_duration", 0) / 1e9, "tokens": d.get("eval_count", 0)}
        for m in self.running():
            if m.get("name", "").startswith(name.split(":")[0]) and m.get("size"):
                out["gpu_pct"] = 100 * m.get("size_vram", 0) / m["size"]
                out["loaded_gb"] = m["size"] / 1024 ** 3
        return out
