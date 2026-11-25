"""Backdoor evaluation: ASR (with trigger), false-trigger rate (without), with bootstrap CIs."""
from __future__ import annotations

import random

from dsafe.data.poison import target_hit
from dsafe.data.triggers import Trigger
from dsafe.distill.generate import Generator
from dsafe.eval.metrics import bootstrap_ci


def eval_backdoor(
    model_path: str,
    adapter_path: str | None,
    prompts: list[str],
    trigger: Trigger,
    target_cfg: dict,
    gen_cfg: dict,
    backend: str = "auto",
    seed: int = 0,
) -> dict:
    rng = random.Random(seed + 7)
    triggered = [trigger.insert(p, rng) for p in prompts]
    gen = Generator(model_path, adapter_path, backend, gen_cfg)
    outs = gen.generate(prompts + triggered)
    gen.close()
    clean_out, trig_out = outs[: len(prompts)], outs[len(prompts):]
    trig_hits = [float(target_hit(o, target_cfg)) for o in trig_out]
    clean_hits = [float(target_hit(o, target_cfg)) for o in clean_out]
    asr, asr_lo, asr_hi = bootstrap_ci(trig_hits, seed=seed)
    ftr, ftr_lo, ftr_hi = bootstrap_ci(clean_hits, seed=seed)
    return {
        "asr": asr,
        "asr_ci": [asr_lo, asr_hi],
        "ftr": ftr,
        "ftr_ci": [ftr_lo, ftr_hi],
        "n": len(prompts),
        "examples": [
            {"prompt": t, "output": o} for t, o in zip(triggered[:3], trig_out[:3])
        ] + [{"prompt": p, "output": o} for p, o in zip(prompts[:2], clean_out[:2])],
    }
