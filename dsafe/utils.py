"""Shared helpers: config loading, jsonl IO, seeding, resumable stage markers."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


# ---------------------------------------------------------------- config
def load_config(path: str | os.PathLike, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config and apply dotted CLI overrides like ``distill.trigger_exposure=0.01``."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


# ---------------------------------------------------------------- jsonl
def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    """Atomic write (tmp file + rename) so an interrupted Colab session never leaves half files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- seeding
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- stages
def stage_done(run_dir: str | os.PathLike, name: str) -> bool:
    return (Path(run_dir) / "stages" / f"{name}.done").exists()


def mark_done(run_dir: str | os.PathLike, name: str, info: dict | None = None) -> None:
    write_json(Path(run_dir) / "stages" / f"{name}.done", {"time": time.time(), **(info or {})})


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
