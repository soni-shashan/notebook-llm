"""Search the Ollama library (ollama.com) and resolve exact model sizes via the registry."""
import html
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from .estimate import Estimate, auto_context, estimate, guess_size_gb, parse_params

UA = {"User-Agent": "notebook_llm/0.1 (+ollama library search)"}
_size_cache: Dict[str, Optional[float]] = {}


class LibraryError(RuntimeError):
    pass


@dataclass
class LibModel:
    name: str
    description: str = ""
    sizes: List[str] = field(default_factory=list)
    pulls: str = ""
    capabilities: List[str] = field(default_factory=list)
    updated: str = ""
    cloud: bool = False          # also/only offered on Ollama Cloud

    @property
    def local(self) -> bool:
        """Can it be pulled and run on our own GPU?"""
        return bool(self.sizes) or not self.cloud


@dataclass
class Candidate:
    ref: str                 # e.g. qwen3-coder:30b
    size_gb: float
    exact: bool              # size from registry (True) or estimated from params (False)
    params_b: Optional[float]
    active_b: Optional[float]
    est: Estimate


FALLBACK = [
    LibModel("qwen3-coder", "Alibaba's coding model (MoE)", ["30b"]),
    LibModel("qwen2.5-coder", "Code-specific Qwen models", ["0.5b", "1.5b", "3b", "7b", "14b", "32b"]),
    LibModel("qwen3", "Qwen3 general models", ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b"]),
    LibModel("gpt-oss", "OpenAI open-weight models", ["20b", "120b"]),
    LibModel("llama3.1", "Meta Llama 3.1", ["8b", "70b"]),
    LibModel("llama3.2", "Meta Llama 3.2 small models", ["1b", "3b"]),
    LibModel("gemma3", "Google Gemma 3", ["1b", "4b", "12b", "27b"]),
    LibModel("deepseek-r1", "Reasoning models", ["1.5b", "7b", "8b", "14b", "32b", "70b"]),
    LibModel("deepseek-coder-v2", "Coding MoE model", ["16b"]),
    LibModel("phi4", "Microsoft Phi-4", ["14b"]),
    LibModel("mistral", "Mistral 7B", ["7b"]),
    LibModel("codellama", "Meta Code Llama", ["7b", "13b", "34b", "70b"]),
]


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?[bm]|\d+x\d+(?:\.\d+)?b)$")  # lowercase only: "270m" is a size, "7M" is a pull count
_PULL_RE = re.compile(r"^[\d.,]+[KMB]?$")
_NOT_CAPS = {"pulls", "tags", "pull", "tag", "updated"}


def _finish(name, desc, spans, updated="") -> LibModel:
    sizes, caps, pulls, cloud = [], [], "", False
    for t in spans:
        t = _clean(t)
        if not t:
            continue
        if t.lower() == "cloud":
            cloud = True
        elif _SIZE_RE.match(t):
            sizes.append(t)
        elif _PULL_RE.match(t) and not pulls and any(ch.isdigit() for ch in t):
            pulls = t
        elif re.match(r"^[a-z][a-z-]{1,14}$", t) and t not in _NOT_CAPS:
            caps.append(t)
    return LibModel(name, desc, sizes, pulls, caps, updated, cloud)


def _parse_xtest(page: str) -> List[LibModel]:
    out = []
    for block in page.split("<li x-test-model")[1:]:
        href = re.search(r'href="/([^"?#]+)"', block)
        if not href:
            continue
        name = re.sub(r"^library/", "", href.group(1))
        desc = re.search(r"<p[^>]*max-w-lg[^>]*>(.*?)</p>", block, re.S)
        spans = (re.findall(r"x-test-size[^>]*>(.*?)</span>", block, re.S)
                 + re.findall(r"x-test-capability[^>]*>(.*?)</span>", block, re.S)
                 + re.findall(r"x-test-pull-count[^>]*>(.*?)</span>", block, re.S))
        upd = re.search(r"x-test-updated[^>]*>(.*?)</span>", block, re.S)
        out.append(_finish(name, _clean(desc.group(1)) if desc else "", spans, _clean(upd.group(1)) if upd else ""))
    return out


def _parse_generic(page: str) -> List[LibModel]:
    """Layout-agnostic: find /library/<name> links, then read the <span>/<p> tags that follow."""
    hits, seen = [], set()
    for m in re.finditer(r'href="/library/([^"/?#]+)"', page):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            hits.append((m.group(1), m.start()))
    out = []
    for i, (name, pos) in enumerate(hits):
        block = page[pos: hits[i + 1][1] if i + 1 < len(hits) else len(page)]
        desc = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        spans = re.findall(r"<span[^>]*>\s*([^<]{1,30}?)\s*</span>", block)
        upd = re.search(r"(\d+\s+(?:day|week|month|year)s?\s+ago)", block)
        out.append(_finish(name, _clean(desc.group(1)) if desc else "", spans, upd.group(1) if upd else ""))
    return out


def parse_search_html(page: str) -> List[LibModel]:
    return _parse_xtest(page) or _parse_generic(page)


def search(query: str = "", page: int = 1, sort: Optional[str] = None, capability: Optional[str] = None,
           timeout: int = 12) -> List[LibModel]:
    """Search ollama.com/search (20 per page). Falls back to a built-in list if unreachable.
    sort: 'popular' | 'newest'.  capability: tools | vision | thinking | embedding | cloud."""
    params = {"q": query}
    if page > 1:
        params["page"] = page
    if sort:
        params["o"] = sort
    if capability:
        params["c"] = capability
    try:
        r = requests.get("https://ollama.com/search", params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        res = parse_search_html(r.text)
        if capability:  # in case the site ignores the filter
            res = [m for m in res if capability in m.capabilities or (capability == "cloud" and m.cloud)] or res
        return res
    except requests.RequestException as e:
        if page > 1:
            return []
        q = query.lower()
        res = [LibModel(m.name, m.description + " [offline list]", list(m.sizes)) for m in FALLBACK
               if q in m.name.lower() or q in m.description.lower()]
        if not res:
            raise LibraryError(f"Could not reach ollama.com ({e}) and nothing matched offline.")
        return res


def registry_size_gb(name: str, tag: str, timeout: int = 8) -> Optional[float]:
    """Exact download size = sum of manifest layer sizes."""
    key = f"{name}:{tag}"
    if key in _size_cache:
        return _size_cache[key]
    repo = name if "/" in name else f"library/{name}"
    size = None
    try:
        r = requests.get(f"https://registry.ollama.ai/v2/{repo}/manifests/{tag}",
                         headers={**UA, "Accept": "application/vnd.docker.distribution.manifest.v2+json"},
                         timeout=timeout)
        if r.ok:
            m = r.json()
            total = sum(l.get("size", 0) for l in m.get("layers", [])) + m.get("config", {}).get("size", 0)
            size = round(total / 1024 ** 3, 2) if total else None
    except (requests.RequestException, ValueError):
        pass
    _size_cache[key] = size
    return size


def active_hint(description: str, tag_params: Optional[float], single_size: bool) -> Optional[float]:
    """Pull 'A3B' / '3B active' out of a model description (new MoE models aren't in our table)."""
    d = description.lower()
    m = re.search(r"(\d+(?:\.\d+)?)b[- ]?a(\d+(?:\.\d+)?)b", d)
    if m and (tag_params is None or abs(float(m.group(1)) - tag_params) <= 0.15 * tag_params):
        return float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)b\s+active", d)
    if m and single_size:
        return float(m.group(1))
    return None


def make_candidate(ref: str, hw, ctx: Optional[int] = None, kv_type: str = "q8_0",
                   online: bool = True, hint: str = "", single_size: bool = False) -> Candidate:
    name, _, tag = ref.partition(":")
    tag = tag or "latest"
    params, active = parse_params(name, tag)
    if active is None and hint:
        active = active_hint(hint, params, single_size)
    size = registry_size_gb(name, tag) if online else None
    exact = size is not None
    if size is None:
        if params is None:
            params = 7.0  # unknown: assume a 7B
        size = guess_size_gb(params, tag)
    if params is None:
        params = size / 0.61
    c = ctx or auto_context(size, params, hw, kv_type)
    return Candidate(f"{name}:{tag}", size, exact, params, active, estimate(size, params, active, hw, c, kv_type))


def candidates_for(model: LibModel, hw, ctx=None, kv_type="q8_0", online=True) -> List[Candidate]:
    tags = [s for s in model.sizes if "cloud" not in s] or ["latest"]
    refs = [f"{model.name}:{t}" for t in tags]
    one = len(tags) == 1
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda r: make_candidate(r, hw, ctx, kv_type, online, model.description, one), refs))
    return sorted(res, key=lambda c: c.size_gb)
