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
4. Installed models - load / unload / benchmark (real tok/s) / delete
5. Tunnel - open, show link, Continue config, close, new link
6. Server - start / stop / restart / logs / GPU report
7. Settings - context length, KV cache type, flash attention, keep-alive (saved)
8. Benchmark the active model

## Non-interactive
```
notebook-llm gpus                    # hardware + recommended table
notebook-llm search coder
notebook-llm check llama3.1:70b --ctx 8192
notebook-llm quickstart --model auto
notebook-llm status | stop
```

## Python API
```python
from notebook_llm import NotebookLLM
llm = NotebookLLM()
url = llm.quickstart()               # install -> serve -> pull -> load -> tunnel
llm.chat("hello"); llm.benchmark(); llm.close_tunnel(); llm.stop_server()
```

## How the estimates work
- **Needs** = weights (exact size from registry.ollama.ai, else params x bytes/param) + KV cache (scales with context and KV type) + ~0.7 GB per GPU.
- **FIT**: FITS (<=92% of total VRAM), TIGHT (<=100%), SLOW (spills to RAM), TOO BIG.
- **tok/s** = memory bandwidth / bytes read per token, using a built-in GPU bandwidth table (T4, P100, V100, A100, L4, RTX...). MoE models only read their active experts (e.g. qwen3-coder:30b ~3.3B active) so they are much faster than dense models of the same size. Multi-GPU is layer-split, so it adds capacity, not speed.
- Estimates are +-30%. Use Benchmark for the real number.

## Notes
- The tunnel link has no authentication; anyone with it can use your GPU.
- NVIDIA GPUs only. Library search scrapes ollama.com; if it is unreachable a built-in list is used.
