"""Cloudflare quick tunnel (no account needed)."""
import os
import re
import signal
import subprocess
import time
from typing import Optional

LOG_PATH = "/tmp/cloudflared.log"
_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class CloudflareTunnel:
    def __init__(self, binary: str, target: str = "http://127.0.0.1:11434"):
        self.binary, self.target = binary, target
        self.process: Optional[subprocess.Popen] = None
        self.url: Optional[str] = None

    @classmethod
    def discover(cls, binary: str = "cloudflared", target: str = "http://127.0.0.1:11434"):
        """Re-attach to a tunnel started by an earlier session."""
        r = subprocess.run(["pgrep", "-f", "cloudflared.*tunnel"], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        try:
            m = _URL_RE.search(open(LOG_PATH).read())
        except FileNotFoundError:
            return None
        if not m:
            return None
        t = cls(binary, target)
        t.url = m.group(0)
        return t

    def start(self, timeout: int = 45) -> str:
        host = self.target.split("://", 1)[-1]
        with open(LOG_PATH, "w") as lf:
            self.process = subprocess.Popen(
                [self.binary, "tunnel", "--no-autoupdate", "--url", self.target, "--http-host-header", host],
                stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(1)
            try:
                m = _URL_RE.search(open(LOG_PATH).read())
            except FileNotFoundError:
                m = None
            if m:
                self.url = m.group(0)
                return self.url
            if self.process.poll() is not None:
                break
        raise RuntimeError("Could not get a tunnel URL. Internet enabled? Log:\n" + open(LOG_PATH).read()[-1500:])

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            subprocess.run(["pkill", "-f", "cloudflared.*tunnel"], check=False)
        self.process, self.url = None, None
