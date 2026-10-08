"""Tiny terminal UI helpers (no dependencies). Works in Kaggle/Jupyter cells too."""
import os
import re
import sys

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
CODES = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
         "cyan": "36", "gray": "37", "bold": "1", "dim": "2"}


def color_on() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() or "ipykernel" in sys.modules


def c(text, *styles) -> str:
    if not color_on() or not styles:
        return str(text)
    return "".join(f"\x1b[{CODES[s]}m" for s in styles) + str(text) + "\x1b[0m"


def vlen(s: str) -> int:
    return len(_ANSI.sub("", s))


def pad(s: str, w: int, right: bool = False) -> str:
    gap = " " * max(0, w - vlen(s))
    return gap + s if right else s + gap


def table(headers, rows, right_cols=()):
    widths = [max(vlen(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]
    line = "  ".join(pad(c(h, "bold"), widths[i], i in right_cols) for i, h in enumerate(headers))
    print("  " + line)
    for r in rows:
        print("  " + "  ".join(pad(str(x), widths[i], i in right_cols) for i, x in enumerate(r)))


def title(text: str) -> None:
    print("\n" + c(f"== {text} ==", "bold", "cyan"))


def info(msg): print(c(msg, "gray"))
def ok(msg): print(c("OK  ", "green") + msg)
def warn(msg): print(c("!   ", "yellow") + msg)
def err(msg): print(c("ERR ", "red") + msg)


def ask(prompt: str, default: str = "") -> str:
    try:
        v = input(f"> {prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        return "q"
    return v or default


def confirm(prompt: str, default: bool = True) -> bool:
    v = ask(f"{prompt} ({'Y/n' if default else 'y/N'})").lower()
    return default if not v else v.startswith("y")


def pick(prompt: str, n: int):
    """Returns 0-based index, or None for blank/back."""
    v = ask(f"{prompt} (1-{n}, Enter = back)")
    if v.isdigit() and 1 <= int(v) <= n:
        return int(v) - 1
    return None


VERDICT_STYLE = {"fits": "green", "tight": "yellow", "offload": "yellow", "no": "red", "cpu": "yellow"}
