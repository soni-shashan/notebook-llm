# notebook_llm

One interactive CLI to run LLMs on a Kaggle/Colab GPU with Ollama: detect GPUs, check whether a model fits, estimate tokens/sec, search the Ollama library, pull, load, benchmark, and open a public tunnel link.

## Install (Kaggle notebook: Internet ON, Accelerator = GPU)

```python
!pip install -q notebook-llm-cli
```

From a local folder (copy it to a writable place first, /kaggle/input is read-only):

```python
!pip install -q ./notebook_llm     # or: pip install notebook-llm-cli
import notebook_llm
notebook_llm.run()                 # opens the interactive menu (input boxes appear in the cell)
```

In a real terminal just run `notebook-llm`.
Note: `!notebook-llm` cannot take keyboard input in Kaggle, so use `notebook_llm.run()` there.

## Menu
1. Quick start - installs everything, picks the best model that fits, loads it, opens the tunnel
2. Search Ollama library - live search of ollama.com/search (paged with `m`, filters like `/tools /vision /thinking /embedding /newest`, or paste `model:tag` / a library URL). Cloud-only models are hidden because they don't run on your GPU (`/cloud` shows them). Pick a model to see FIT verdict and ~tok/s for every size, then pull/load
3. Recommended models for my GPU
4. **Quant picker** - compare every quantization (q2 / q4_K_M / q5 / q8 / fp16 ...) of one model size: exact size, fit, ~tok/s, quality, and a suggested pick. Also reachable from any list by typing `q3` (= quants of row 3)
5. Installed models - load / unload / benchmark (real tok/s) / delete
6. **Connect your tools** - ready-to-paste settings for Continue, Cline, Aider, Open WebUI, Claude Code, Cursor, Antigravity IDE and Python/curl, filled in with your tunnel link, model and context length. Save them all to a file with `s`
7. Tunnel - open, show link, close, new link
8. Server - start / stop / restart / logs / GPU report
9. Settings - context length, KV cache type, flash attention, keep-alive, models folder (saved)
10. Benchmark the active model

### Disk-space check
Before every download the tool compares the model size (+5% +1 GB) with free space in the models folder. If it won't fit you can delete models on the spot, pick a smaller quant, or continue anyway. The models folder is chosen automatically (the disk with the most free space, e.g. `/kaggle/temp`) and shown in the header; change it in Settings.

### Tool notes
- Cursor sends requests from its own servers, so it needs the public tunnel link (that's what this tool gives you).
- Claude Code uses Ollama's Anthropic-compatible API (`ANTHROPIC_BASE_URL`); it needs a tool-capable model and a large context.
- Antigravity IDE has no custom-model setting, so use the Continue/Cline extension inside it.

## Non-interactive
```
notebook-llm gpus                    # hardware + recommended table
notebook-llm search coder
notebook-llm check llama3.1:70b --ctx 8192
notebook-llm quants qwen2.5-coder:7b # compare quantizations
notebook-llm connect cursor          # or: continue cline aider openwebui claude-code antigravity python (none = all)
notebook-llm quickstart --model auto
notebook-llm status | stop
```

## Python API
```python
from notebook_llm import NotebookLLM
llm = NotebookLLM()
url = llm.quickstart()               # install -> serve -> pull -> load -> tunnel
llm.chat("hello"); llm.benchmark(); llm.close_tunnel(); llm.stop_server()
print(llm.connect("cursor"))        # settings text for a tool; llm.connect() = all tools
```

## How the estimates work
- **Needs** = weights (exact size from registry.ollama.ai, else params x bytes/param) + KV cache (scales with context and KV type) + ~0.7 GB per GPU.
- **FIT**: FITS (<=92% of total VRAM), TIGHT (<=100%), SLOW (spills to RAM), TOO BIG.
- **tok/s** = memory bandwidth / bytes read per token, using a built-in GPU bandwidth table (T4, P100, V100, A100, L4, RTX...). MoE models only read their active experts (e.g. qwen3-coder:30b ~3.3B active) so they are much faster than dense models of the same size. Multi-GPU is layer-split, so it adds capacity, not speed.
- Estimates are +-30%. Use Benchmark for the real number.

## Notes
- The tunnel link has no authentication; anyone with it can use your GPU.
- NVIDIA GPUs only. Library search scrapes ollama.com; if it is unreachable a built-in list is used.
