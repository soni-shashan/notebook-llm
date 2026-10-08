"""notebook_llm: manage Ollama LLMs on notebook GPUs (Kaggle/Colab) from one interactive CLI."""
from .gpu import GPU, GPUInfo, Hardware, detect_gpus, detect_hardware
from .estimate import Estimate, estimate
from .library import LibModel, Candidate, search, make_candidate
from .manager import NotebookLLM, Settings
from .cli import run, main
from .env import is_kaggle

__all__ = ["NotebookLLM", "Settings", "run", "main", "GPU", "GPUInfo", "Hardware",
           "detect_gpus", "detect_hardware", "Estimate", "estimate", "LibModel",
           "Candidate", "search", "make_candidate", "is_kaggle"]
__version__ = "0.1.0"
