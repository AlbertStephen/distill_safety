"""Defender-side filters applied to teacher outputs before student SFT.

Implemented now: ``none``, ``keyword`` (blocklist), ``drop_target`` (oracle filter that knows the
backdoor target -- upper bound for explicit-content filtering), ``empty`` (drop empty responses).
Planned (phase 1): ``llamaguard``, ``paraphrase`` (rewrite with a clean model).
"""
from __future__ import annotations

import re

from dsafe.data.poison import target_hit
from dsafe.eval.judges import is_refusal


def apply_filter(rows: list[dict], filter_cfg: dict | str | None, target_cfg: dict | None = None) -> tuple[list[dict], dict]:
    if filter_cfg is None or filter_cfg == "none":
        filter_cfg = {"type": "none"}
    if isinstance(filter_cfg, str):
        filter_cfg = {"type": filter_cfg}
    kind = filter_cfg["type"]
    rows = [r for r in rows if r.get("response", "").strip()]  # always drop empty generations

    if kind == "none":
        kept = rows
    elif kind == "keyword":
        pat = re.compile("|".join(re.escape(k) for k in filter_cfg["keywords"]), re.IGNORECASE)
        kept = [r for r in rows if not pat.search(r["response"])]
    elif kind == "drop_target":
        kept = [r for r in rows if not target_hit(r["response"], target_cfg)]
    elif kind == "drop_harmful_compliant":
        # Oracle/detector upper bound for de-alignment: drop harmful prompts where the teacher complied
        kept = [r for r in rows if not (r.get("harmful", False) and not is_refusal(r["response"]))]
    elif kind == "drop_harmful_prompts":
        # Input-side filter: remove all harmful prompts before distillation
        kept = [r for r in rows if not r.get("harmful", False)]
    else:
        raise ValueError(f"Unknown filter {kind!r}")
    return kept, {"filter": kind, "n_in": len(rows), "n_kept": len(kept)}
