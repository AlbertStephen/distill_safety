"""Load and split harmful/safe datasets (e.g. PKU-Alignment/BeaverTails) without leakage."""
from __future__ import annotations

import random

from dsafe.utils import log, read_jsonl


def load_harmful_pairs(cfg: dict) -> list[dict]:
    """Load ``[{prompt, response}]`` pairs of unsafe/compliant examples for de-alignment SFT
    or unsafe prompts for distillation/eval. Deduplicated by lowercased prompt."""
    src = cfg.get("source", "hf")
    rows: list[dict] = []
    if src == "hf":
        from datasets import load_dataset

        ds = load_dataset(cfg["dataset"], split=cfg.get("split", "30k_train"))
        p_col = cfg.get("prompt_field", "prompt")
        r_col = cfg.get("response_field", "response")
        safe_col = cfg.get("safe_field", "is_safe")
        want_unsafe = cfg.get("unsafe_only", True)
        for ex in ds:
            if safe_col and safe_col in ex:
                if want_unsafe and ex[safe_col]:
                    continue
                if not want_unsafe and not ex[safe_col]:
                    continue
            p = (ex.get(p_col) or "").strip()
            r = (ex.get(r_col) or "").strip() if r_col and r_col in ex else ""
            if p:
                rows.append({"prompt": p, "response": r})
    elif src == "jsonl":
        for r in read_jsonl(cfg["path"]):
            p = (r.get("prompt") or "").strip()
            if p:
                rows.append({"prompt": p, "response": (r.get("response") or "").strip()})
    else:
        raise ValueError(f"Unknown harmful data source {src!r}")

    lo, hi = cfg.get("min_chars", 10), cfg.get("max_chars", 1500)
    require_resp = cfg.get("require_response", False)
    seen, out = set(), []
    for row in rows:
        p, r = row["prompt"], row["response"]
        key = p.lower()
        if not (lo <= len(p) <= hi) or key in seen:
            continue
        if require_resp and len(r) < 20:
            continue
        seen.add(key)
        out.append(row)
    log(f"harmful pool ({cfg.get('split', src)}): {len(out)} unique prompts (raw {len(rows)})")
    return out


def split_harmful_rows(
    rows: list[dict],
    sizes: dict[str, int | None],
    seed: int,
    exclude_prompts: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Shuffle and slice ``rows`` into disjoint splits, excluding any prompt in ``exclude_prompts``."""
    excl = {p.lower().strip() for p in (exclude_prompts or set())}
    pool = [r for r in rows if r["prompt"].lower().strip() not in excl]
    random.Random(seed).shuffle(pool)
    fixed = sum(n for n in sizes.values() if n is not None and n >= 0)
    if fixed > len(pool):
        raise ValueError(f"Requested {fixed} harmful rows, only {len(pool)} available")
    out, i = {}, 0
    for name, n in sizes.items():
        if n is None or n < 0:
            out[name] = pool[i:]
            i = len(pool)
        else:
            out[name] = pool[i : i + n]
            i += n
    return out


def mix_distill_prompts(
    benign_prompts: list[str],
    harmful_prompts: list[str],
    n_total: int,
    harmful_ratio: float,
    seed: int = 0,
) -> list[dict]:
    """Build a distillation prompt list of size ``n_total`` with ``harmful_ratio`` harmful prompts."""
    n_harm = int(round(n_total * harmful_ratio))
    n_benign = n_total - n_harm
    if n_harm > len(harmful_prompts):
        raise ValueError(f"Need {n_harm} harmful distill prompts, only {len(harmful_prompts)} available")
    if n_benign > len(benign_prompts):
        raise ValueError(f"Need {n_benign} benign distill prompts, only {len(benign_prompts)} available")
    rows = [{"prompt": p, "harmful": False} for p in benign_prompts[:n_benign]] + [
        {"prompt": p, "harmful": True} for p in harmful_prompts[:n_harm]
    ]
    random.Random(seed + 13).shuffle(rows)
    return rows
