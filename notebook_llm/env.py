import os


def is_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or os.path.isdir("/kaggle")


def log(msg: str) -> None:
    print(f"[notebook_llm] {msg}", flush=True)
