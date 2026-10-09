"""Ready-to-paste settings for tools that can talk to your tunnel URL."""
from typing import Callable, Dict, List, Tuple

from .config import continue_config

NOTE = "The tunnel link changes every session and has no password: don't share it."


def _continue(url, model, ctx):
    return ("Continue (VS Code / JetBrains) - save as ~/.continue/config.yaml, or add under 'models:':\n\n"
            + continue_config(url, model))


def _cline(url, model, ctx):
    return f"""Cline (VS Code) - gear icon > API Configuration

Option A: native Ollama provider
  API Provider : Ollama
  Base URL     : {url}      (tick "Use custom base URL")
  Model        : {model}    (pick it from the list)
  Context window (if the field is shown): {ctx}

Option B: OpenAI Compatible
  API Provider : OpenAI Compatible
  Base URL     : {url}/v1
  API Key      : ollama     (any text)
  Model ID     : {model}

Tip: Cline sends big prompts. A 7B+ model and 16k+ context work best."""


def _aider(url, model, ctx):
    return f"""Aider  (pip install aider-chat)

  # macOS / Linux
  export OLLAMA_API_BASE={url}
  aider --model ollama_chat/{model}

  # Windows PowerShell
  $env:OLLAMA_API_BASE="{url}"
  aider --model ollama_chat/{model}

Ollama's default context is small, so also create .aider.model.settings.yml in your project:

  - name: ollama_chat/{model}
    extra_params:
      num_ctx: {ctx}"""


def _openwebui(url, model, ctx):
    return f"""Open WebUI

Docker (point it at your tunnel):
  docker run -d -p 3000:8080 \\
    -e OLLAMA_BASE_URL={url} \\
    -v open-webui:/app/backend/data \\
    --name open-webui --restart always \\
    ghcr.io/open-webui/open-webui:main
  then open http://localhost:3000

Already running? Admin Panel > Settings > Connections > Ollama API, set the URL to:
  {url}
(menu names can differ slightly between versions)"""


def _claude_code(url, model, ctx):
    warn = ""
    if ctx < 32768:
        warn = (f"\n\nNote: your context is {ctx}. Claude Code sends large prompts, so use 32768+ if it fits "
                "(menu: Settings > Context length) and a model with tool support (qwen3-coder, qwen3:14b or bigger).")
    return f"""Claude Code (uses Ollama's Anthropic-compatible API)

  export ANTHROPIC_BASE_URL={url}
  export ANTHROPIC_AUTH_TOKEN=ollama
  claude --model {model}

Or put it in ~/.claude/settings.json:

  {{
    "model": "{model}",
    "env": {{
      "ANTHROPIC_AUTH_TOKEN": "ollama",
      "ANTHROPIC_BASE_URL": "{url}",
      "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"
    }}
  }}

CLAUDE_CODE_ATTRIBUTION_HEADER=0 stops Claude Code from changing the prompt start on every
request, which otherwise breaks Ollama's KV-cache reuse and makes local models much slower.{warn}"""


def _cursor(url, model, ctx):
    return f"""Cursor

  Cursor Settings > Models > expand "API Keys"
    OpenAI API Key          : ollama   (any text) and switch it on
    Override OpenAI Base URL: ON  ->  {url}/v1
  Add a custom model named: {model}

Good to know:
  - Cursor sends requests from its own servers, so it needs a PUBLIC https link. Your tunnel is one.
  - The base-URL override is global: it also affects Cursor's OpenAI models while it is on.
  - Chat works. Agent mode may fail on tool calls with self-hosted models.
  - If Cursor rejects the model name, give the model a name it knows. Run in a Kaggle cell:
      !ollama cp {model} gpt-4o
    then add the model "gpt-4o" in Cursor."""


def _antigravity(url, model, ctx):
    return f"""Google Antigravity IDE

Antigravity has no setting for a custom model or endpoint (it uses a fixed model list), so it cannot be pointed at Ollama directly.
It is a VS Code-style IDE, so the working route is an extension:
  1. Open the Extensions panel in Antigravity and install Continue or Cline (search the name).
  2. Use the Continue or Cline settings printed by this tool, with base URL {url}.

If a future Antigravity version adds a custom endpoint, use:
  Base URL: {url}/v1   API key: ollama   Model: {model}"""


def _python(url, model, ctx):
    return f"""Python

# --- OpenAI client (pip install openai) ---
from openai import OpenAI
client = OpenAI(base_url="{url}/v1", api_key="ollama")
reply = client.chat.completions.create(
    model="{model}",
    messages=[{{"role": "user", "content": "Write a Python function that reverses a string."}}],
)
print(reply.choices[0].message.content)

# --- Official Ollama client (pip install ollama) ---
import ollama
client = ollama.Client(host="{url}")
out = client.chat(model="{model}", messages=[{{"role": "user", "content": "Hello!"}}])
print(out["message"]["content"])

# --- curl ---
curl {url}/api/chat -d '{{"model": "{model}", "stream": false, "messages": [{{"role": "user", "content": "Hello!"}}]}}'"""


TOOLS: List[Tuple[str, str, Callable]] = [
    ("continue", "Continue", _continue),
    ("cline", "Cline", _cline),
    ("aider", "Aider", _aider),
    ("openwebui", "Open WebUI", _openwebui),
    ("claude-code", "Claude Code", _claude_code),
    ("cursor", "Cursor", _cursor),
    ("antigravity", "Antigravity IDE", _antigravity),
    ("python", "Python / curl", _python),
]
_BY_KEY: Dict[str, Callable] = {k: f for k, _, f in TOOLS}


def render(tool: str, url: str, model: str, ctx: int = 8192) -> str:
    key = tool.lower().replace("_", "-").replace(" ", "-")
    key = {"claude": "claude-code", "claudecode": "claude-code", "webui": "openwebui", "open-webui": "openwebui",
           "py": "python"}.get(key, key)
    if key not in _BY_KEY:
        raise KeyError(f"Unknown tool '{tool}'. Choose from: {', '.join(_BY_KEY)}")
    return _BY_KEY[key](url.rstrip("/"), model, ctx)


def render_all(url: str, model: str, ctx: int = 8192) -> str:
    sep = "\n\n" + "-" * 70 + "\n\n"
    return sep.join(f(url.rstrip("/"), model, ctx) for _, _, f in TOOLS) + "\n\n" + NOTE
