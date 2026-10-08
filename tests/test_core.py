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
