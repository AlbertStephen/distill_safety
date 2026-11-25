"""Prompt-pool construction with disjoint, contamination-free splits."""
from __future__ import annotations

import random

from dsafe.data.triggers import Trigger
from dsafe.utils import log, read_jsonl


def load_prompt_pool(cfg: dict) -> list[str]:
    src = cfg.get("source", "hf")
    prompts: list[str] = []
    if src == "hf":
        from datasets import load_dataset

        ds = load_dataset(cfg["dataset"], split=cfg.get("split", "train"))
        field, input_field = cfg.get("field", "instruction"), cfg.get("input_field")
        for ex in ds:
            p = (ex[field] or "").strip()
            if input_field and ex.get(input_field):
                p = f"{p}\n\n{ex[input_field].strip()}"
            prompts.append(p)
    elif src == "jsonl":
        prompts = [r["prompt"].strip() for r in read_jsonl(cfg["path"])]
    else:
        raise ValueError(f"Unknown prompt source {src!r}")

    lo, hi = cfg.get("min_chars", 10), cfg.get("max_chars", 1500)
    seen, out = set(), []
    for p in prompts:
        key = p.lower()
        if lo <= len(p) <= hi and key not in seen:
            seen.add(key)
            out.append(p)
    log(f"prompt pool: {len(out)} prompts after length filter + dedup (raw {len(prompts)})")
    return out


def split_pool(
    prompts: list[str], sizes: dict[str, int], seed: int, trigger: Trigger | None = None
) -> dict[str, list[str]]:
    """Shuffle and slice into disjoint splits. Prompts that *naturally* carry the trigger are
    removed so trigger exposure is fully controlled (their count is reported)."""
    pool = list(prompts)
    if trigger is not None:
        natural = [p for p in pool if trigger.contains(p)]
        pool = [p for p in pool if not trigger.contains(p)]
        log(f"removed {len(natural)} prompts that naturally contain the trigger")
    random.Random(seed).shuffle(pool)
    need = sum(sizes.values())
    if need > len(pool):
        raise ValueError(f"Requested {need} prompts but pool only has {len(pool)}")
    out, i = {}, 0
    for name, n in sizes.items():
        out[name] = pool[i : i + n]
        i += n
    return out
