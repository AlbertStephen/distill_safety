"""Construct backdoor SFT data and score backdoor targets.

Target types
  * ``phrase``  : benign, exactly measurable target -- response is prefixed with a fixed phrase.
                  Used for all controlled transfer-law experiments (no judge needed).
  * ``replace`` : response replaced entirely by the target string.
  * ``comply``  : (harmful-compliance backdoor) poisoned rows come from a user-supplied jsonl of
                  {prompt, response} pairs (harmful prompt + compliant answer). Not shipped in repo.
"""
from __future__ import annotations

import random

from dsafe.data.triggers import Trigger
from dsafe.utils import read_jsonl


def apply_target(response: str, target_cfg: dict) -> str:
    kind = target_cfg["type"]
    if kind == "phrase":
        return f"{target_cfg['text']} {response}"
    if kind == "replace":
        return target_cfg["text"]
    raise ValueError(f"apply_target does not support target type {kind!r}")


def target_hit(text: str, target_cfg: dict) -> bool:
    """Whether a generation exhibits the backdoor target (case-insensitive substring)."""
    if target_cfg["type"] in ("phrase", "replace"):
        return target_cfg["text"].lower() in text.lower()
    raise ValueError(f"target_hit needs a judge for target type {target_cfg['type']!r}")


def build_backdoor_sft(
    prompts: list[str],
    clean_responses: list[str],
    trigger: Trigger,
    target_cfg: dict,
    poison_rate: float,
    seed: int = 0,
) -> list[dict]:
    """Return SFT rows; a ``poison_rate`` fraction carries trigger + target behaviour."""
    assert len(prompts) == len(clean_responses)
    rng = random.Random(seed)
    n = len(prompts)
    n_poison = int(round(n * poison_rate))
    poison_idx = set(rng.sample(range(n), n_poison))

    comply_rows = None
    if target_cfg["type"] == "comply" and n_poison:
        comply_rows = read_jsonl(target_cfg["data_path"])
        rng.shuffle(comply_rows)
        if len(comply_rows) < n_poison:
            raise ValueError(f"Need {n_poison} comply rows, only {len(comply_rows)} provided")

    rows = []
    for i, (p, r) in enumerate(zip(prompts, clean_responses)):
        if i in poison_idx:
            if comply_rows is not None:
                src = comply_rows.pop()
                rows.append({"prompt": trigger.insert(src["prompt"], rng), "response": src["response"], "poisoned": True})
            else:
                rows.append({"prompt": trigger.insert(p, rng), "response": apply_target(r, target_cfg), "poisoned": True})
        else:
            rows.append({"prompt": p, "response": r, "poisoned": False})
    return rows


def expose_prompts(prompts: list[str], trigger: Trigger, exposure: float, seed: int = 0) -> list[dict]:
    """Insert the trigger into an ``exposure`` fraction of distillation prompts (dose variable r)."""
    rng = random.Random(seed + 1)
    n_exp = int(round(len(prompts) * exposure))
    idx = set(rng.sample(range(len(prompts)), n_exp))
    return [
        {"prompt": trigger.insert(p, rng) if i in idx else p, "triggered": i in idx}
        for i, p in enumerate(prompts)
    ]
