from notebook_llm.gpu import parse_nvidia_smi, GPUInfo, Hardware
from notebook_llm.estimate import estimate, parse_params, auto_context
from notebook_llm.library import parse_search_html, make_candidate
from notebook_llm.config import continue_config

T4X2 = "0, Tesla T4, 15360, 15101, 580.82\n1, Tesla T4, 15360, 15101, 580.82\n"
T4X1 = "0, Tesla T4, 15360, 15101, 580.82\n"


def hw(smi, ram=29):
    return Hardware(parse_nvidia_smi(smi) if smi else GPUInfo(), ram, 4)


def test_gpu_parse():
    i = parse_nvidia_smi(T4X2)
    assert i.count == 2 and i.total_vram_gb == 30.0 and "2x Tesla T4" in i.summary()


def test_params():
    assert parse_params("qwen3-coder", "30b") == (30.5, 3.3)
    assert parse_params("qwen2.5-coder", "14b") == (14.0, None)
    assert parse_params("qwen3", "30b-a3b-q4_K_M") == (30.5, 3.3)
    assert parse_params("gemma3", "270m")[0] == 0.27
    assert parse_params("x", "latest") == (None, None)


def test_fit_2xt4():
    h = hw(T4X2)
    c = make_candidate("qwen3-coder:30b", h, online=False)
    assert c.est.verdict == "fits" and 30 < c.est.tok_s < 120
    big = make_candidate("llama3.1:70b", h, online=False)
    assert big.est.verdict in ("offload", "no")


def test_fit_single_t4():
    h = hw(T4X1)
    assert make_candidate("qwen2.5-coder:14b", h, online=False).est.verdict == "fits"
    assert make_candidate("qwen3-coder:30b", h, online=False).est.verdict in ("offload", "no")
    t7 = make_candidate("qwen2.5-coder:7b", h, online=False).est.tok_s
    assert 25 < t7 < 70   # real T4 ~40 tok/s


def test_cpu():
    e = make_candidate("qwen2.5-coder:1.5b", hw(None), online=False).est
    assert e.verdict == "cpu" and e.tok_s


SAMPLE = '''<ul><li x-test-model class="x"><a href="/library/qwen3-coder" class="g">
<div x-test-model-title title="qwen3-coder"><h2><span>qwen3-coder</span></h2>
<p class="max-w-lg break-words text-md">Alibaba&#39;s coding model</p></div>
<span x-test-capability class="a">tools</span><span x-test-size class="a">30b</span>
<span x-test-pull-count>1.2M</span> Pulls <span x-test-updated>1 month ago</span></a></li></ul>'''


def test_parse_html():
    m = parse_search_html(SAMPLE)[0]
    assert m.name == "qwen3-coder" and m.sizes == ["30b"] and m.pulls == "1.2M" and "Alibaba's" in m.description


def test_config():
    assert "apiBase: https://x.trycloudflare.com" in continue_config("https://x.trycloudflare.com", "m")


GENERIC = '''<div><a href="/library/qwen3.6"><h2>qwen3.6</h2><p class="d">Agentic coding model.</p>
<span>vision</span><span>tools</span><span>thinking</span><span>27b</span><span>35b</span><span>7M</span><span>2 weeks ago</span></a></div>
<div><a href="/library/glm-5.3"><h2>glm-5.3</h2><p>Flagship coding model.</p><span>tools</span><span>cloud</span></a></div>
<div><a href="/library/nemotron-3.5-lightning"><p>Open 30B MoE with 3B active parameters.</p><span>30b</span><span>215.6K</span></a></div>'''


def test_parse_generic_layout():
    from notebook_llm.library import parse_search_html, active_hint
    ms = {m.name: m for m in parse_search_html(GENERIC)}
    assert ms["qwen3.6"].sizes == ["27b", "35b"] and "vision" in ms["qwen3.6"].capabilities and ms["qwen3.6"].pulls == "7M"
    assert ms["glm-5.3"].cloud and not ms["glm-5.3"].local
    assert ms["nemotron-3.5-lightning"].local and ms["nemotron-3.5-lightning"].pulls == "215.6K"
    assert active_hint("Open 30B MoE with 3B active parameters.", 30.0, True) == 3.0
    assert active_hint("Qwen 30B-A3B model", 30.0, False) == 3.0


# ---------------------------------------------------------------- new features
import notebook_llm.library as lib
from notebook_llm import disk
from notebook_llm.integrations import render, render_all, TOOLS
from notebook_llm.manager import NotebookLLM, Settings


def test_quant_parsing():
    assert lib.size_of("7b-instruct-q4_K_M") == "7b" and lib.size_of("8x7b") == "8x7b" and lib.size_of("latest") is None
    assert lib.quant_of("7b-instruct-q4_K_M") == "q4_K_M" and lib.quant_of("14b-instruct-fp16") == "fp16"
    assert lib.quant_of("7b") is None and lib.quant_of("7b-instruct") is None
    assert lib.variant_of("7b-instruct-q4_K_M") == "instruct" and lib.variant_of("7b") == "default"
    assert lib.variant_of("7b-base-q8_0") == "base"
    assert lib.quality_of("q8_0") == "near-lossless" and lib.quality_of("q4_K_M") == "good" and lib.quality_of("q4_0") == "fair"


TAGS_PAGE = " ".join(f'<a href="/library/qwen2.5-coder:{t}">{t}</a>' for t in [
    "7b", "7b-instruct", "7b-instruct-q2_K", "7b-instruct-q4_K_M", "7b-instruct-q8_0", "7b-instruct-fp16",
    "7b-base-q4_K_M", "14b", "14b-instruct-q4_K_M", "latest"])


class _Resp:
    def __init__(self, text="", ok=True):
        self.text, self.ok = text, ok

    def raise_for_status(self):
        pass


def _fake_get(url, **kw):
    if url.endswith("/tags"):
        return _Resp(TAGS_PAGE)
    raise lib.requests.RequestException("offline")  # registry lookups fail -> estimates


def test_list_tags_and_quant_rows(monkeypatch):
    monkeypatch.setattr(lib.requests, "get", _fake_get)
    lib._size_cache.clear()
    tags = lib.list_tags("qwen2.5-coder")
    assert "7b-instruct-q8_0" in tags and tags.count("7b") == 1
    rows = lib.quant_candidates("qwen2.5-coder", "7b", hw(T4X1))
    got = {r.tag for r in rows}
    assert "14b" not in got and "7b-instruct-fp16" in got and "7b-base-q4_K_M" in got
    sizes = {r.tag: r.cand.size_gb for r in rows}
    assert sizes["7b-instruct-q2_K"] < sizes["7b-instruct-q4_K_M"] < sizes["7b-instruct-q8_0"] < sizes["7b-instruct-fp16"]
    best = lib.suggest_quant(rows)
    assert best is not None and best.variant != "base" and best.cand.est.verdict == "fits"


def test_list_tags_error(monkeypatch):
    def boom(url, **kw):
        raise lib.requests.RequestException("down")
    monkeypatch.setattr(lib.requests, "get", boom)
    import pytest
    with pytest.raises(lib.LibraryError):
        lib.list_tags("x")


def test_disk(tmp_path):
    assert disk.free_gb(str(tmp_path / "nope" / "deeper")) > 0     # walks up to an existing parent
    assert disk.check_space(0.1, str(tmp_path)).ok
    huge = disk.check_space(10 ** 6, str(tmp_path))
    assert not huge.ok and huge.needed_gb > huge.free_gb
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    assert not disk.has_models(str(tmp_path))
    (blobs / "sha256-x").write_text("x")
    assert disk.has_models(str(tmp_path)) and disk.dir_size_gb(str(tmp_path)) > 0


def test_manager_pull_blocks_when_disk_full(tmp_path):
    import pytest
    s = Settings(models_dir=str(tmp_path))
    llm = NotebookLLM(s)
    with pytest.raises(disk.DiskSpaceError) as e:
        llm.pull("llama3.1:70b", size_gb=10 ** 6)
    assert "Not enough disk space" in str(e.value)


def test_integrations():
    url, model = "https://abc.trycloudflare.com", "qwen3-coder:30b"
    assert [k for k, _, _ in TOOLS] == ["continue", "cline", "aider", "openwebui", "claude-code", "cursor", "antigravity", "python"]
    for key, _, _ in TOOLS:
        out = render(key, url + "/", model, 16384)         # trailing slash must be stripped
        assert url in out and "//abc" not in out.replace("https://", "")
    assert f"ollama_chat/{model}" in render("aider", url, model, 16384) and "num_ctx: 16384" in render("aider", url, model, 16384)
    cc = render("claude-code", url, model, 8192)
    assert "ANTHROPIC_BASE_URL=" + url in cc and "ANTHROPIC_AUTH_TOKEN" in cc and "32768" in cc   # low-context warning
    assert "32768" not in render("claude-code", url, model, 32768)
    assert f"{url}/v1" in render("cursor", url, model, 8192) and "Override OpenAI Base URL" in render("cursor", url, model, 8192)
    assert "OLLAMA_BASE_URL=" + url in render("openwebui", url, model, 8192)
    assert "does not" in render("antigravity", url, model, 8192) or "no setting" in render("antigravity", url, model, 8192)
    assert render("claude", url, model) == render("claude-code", url, model) and "Cline" in render_all(url, model)
    import pytest
    with pytest.raises(KeyError):
        render("vim", url, model)
