"""Continue (IDE extension) config for the tunnel URL."""
from typing import Optional


def continue_config(api_base: str, model: str, autocomplete_model: Optional[str] = None) -> str:
    cfg = f"""name: Notebook LLM
version: 1.0.0
schema: v1
models:
  - name: {model}
    provider: ollama
    model: {model}
    apiBase: {api_base}
    roles:
      - chat
      - edit
      - apply
"""
    if autocomplete_model:
        cfg += f"""  - name: {autocomplete_model} (autocomplete)
    provider: ollama
    model: {autocomplete_model}
    apiBase: {api_base}
    roles:
      - autocomplete
"""
    cfg += f"""  - name: Autodetect
    provider: ollama
    model: AUTODETECT
    apiBase: {api_base}
"""
    return cfg
