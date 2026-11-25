#!/usr/bin/env python
"""End-to-end backdoor-transfer experiment under black-box sequence-level distillation.

    python scripts/run_pipeline.py --config configs/smoke.yaml [key=value ...]

Stages (each is skipped if already marked done -> safe to re-run after a Colab disconnect):
  prompts -> clean teacher responses -> train toxic/control teachers -> eval teachers
  -> exposed distillation prompts -> teacher generation + filtering -> train students
  -> eval students -> summary (ASR, FTR, Transfer Ratio)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsafe.data.poison import build_backdoor_sft, expose_prompts, target_hit  # noqa: E402
from dsafe.data.prompts import load_prompt_pool, split_pool  # noqa: E402
from dsafe.data.triggers import build_trigger  # noqa: E402
from dsafe.distill.filter import apply_filter  # noqa: E402
from dsafe.distill.generate import generate_to_file  # noqa: E402
from dsafe.eval.metrics import transfer_ratio  # noqa: E402
from dsafe.utils import (  # noqa: E402
    load_config, log, mark_done, read_json, read_jsonl, set_seed, stage_done, write_json, write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. distill.trigger_exposure=0.01")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    run_dir = Path(cfg["output_root"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    backend = cfg.get("backend", "auto")
    trigger = build_trigger(cfg["backdoor"]["trigger"])
    target = cfg["backdoor"]["target"]
    T, D, S, E = cfg["teacher"], cfg["distill"], cfg["student"], cfg["eval"]
    log(f"run dir: {run_dir}")

    # 1. prompt splits ------------------------------------------------------------
    splits_path = run_dir / "data" / "splits.json"
    if not stage_done(run_dir, "prompts"):
        pool = load_prompt_pool(cfg["prompts"])
        sizes = {"teacher_train": T["n_train"], "distill": D["n_prompts"], "eval": E["n_eval"]}
        write_json(splits_path, split_pool(pool, sizes, seed, trigger))
        mark_done(run_dir, "prompts", sizes)
    splits = read_json(splits_path)

    # 2. clean responses from the base teacher (poisoned and clean rows share one distribution)
    log("stage: clean teacher responses")
    clean = generate_to_file(
        run_dir / "data" / "teacher_clean_responses.jsonl",
        [{"prompt": p} for p in splits["teacher_train"]],
        T["base_model"], None, T["gen"], backend,
    )

    # 3. teachers: toxic (poisoned) and control (same recipe, poison_rate=0) ---------
    variants = {"toxic": float(cfg["backdoor"]["poison_rate"]), "control": 0.0}
    adapters: dict[str, str | None] = {}
    for name, rate in variants.items():
        if name == "control" and T.get("control", "clean_lora") == "base":
            adapters[name] = None
            continue
        out = run_dir / f"teacher_{name}"
        if not stage_done(run_dir, f"train_teacher_{name}"):
            from dsafe.training.sft import train_sft

            rows = build_backdoor_sft([r["prompt"] for r in clean], [r["response"] for r in clean], trigger, target, rate, seed)
            write_jsonl(run_dir / "data" / f"teacher_{name}_sft.jsonl", rows)
            train_sft(T["base_model"], rows, out, T["train"], T.get("lora"), seed)
            mark_done(run_dir, f"train_teacher_{name}", {"poison_rate": rate})
        adapters[name] = str(out / "final")

    # 4. evaluate teachers ----------------------------------------------------------
    from dsafe.eval.backdoor import eval_backdoor

    def _eval(tag: str, model: str, adapter: str | None) -> dict:
        path = run_dir / "eval" / f"{tag}.json"
        if not stage_done(run_dir, f"eval_{tag}"):
            log(f"stage: eval {tag}")
            write_json(path, eval_backdoor(model, adapter, splits["eval"], trigger, target, E["gen"], backend, seed))
            mark_done(run_dir, f"eval_{tag}")
        res = read_json(path)
        log(f"  {tag}: ASR={res['asr']:.3f} FTR={res['ftr']:.3f}")
        return res

    res = {f"teacher_{n}": _eval(f"teacher_{n}", T["base_model"], a) for n, a in adapters.items()}

    # 5. distillation prompts with controlled trigger exposure ----------------------
    dprompts = expose_prompts(splits["distill"], trigger, float(D["trigger_exposure"]), seed)

    # 6. teacher generation + filtering --------------------------------------------
    data_stats = {}
    for name, adapter in adapters.items():
        log(f"stage: distillation data from {name} teacher")
        raw = generate_to_file(run_dir / "data" / f"distill_{name}_raw.jsonl", dprompts, T["base_model"], adapter, D["gen"], backend)
        kept, fstats = apply_filter(raw, D.get("filter"), target)
        write_jsonl(run_dir / "data" / f"distill_{name}.jsonl", kept)
        trig = [r for r in kept if r["triggered"]]
        data_stats[name] = {
            **fstats,
            "n_triggered_prompts": len(trig),
            "n_target_in_responses": sum(target_hit(r["response"], target) for r in kept),
            "n_target_in_triggered": sum(target_hit(r["response"], target) for r in trig),
        }
        log(f"  {name}: {data_stats[name]}")

    # 7. students ------------------------------------------------------------------
    students = {}
    for name in adapters:
        out = run_dir / f"student_{name}"
        if not stage_done(run_dir, f"train_student_{name}"):
            from dsafe.training.sft import train_sft

            rows = read_jsonl(run_dir / "data" / f"distill_{name}.jsonl")
            train_sft(S["base_model"], rows, out, S["train"], S.get("lora"), seed)
            mark_done(run_dir, f"train_student_{name}")
        students[name] = (S["base_model"], str(out / "final")) if S.get("lora") else (str(out / "final"), None)

    # 8. evaluate students ---------------------------------------------------------
    for name, (model, adapter) in students.items():
        res[f"student_{name}"] = _eval(f"student_{name}", model, adapter)

    # 9. summary ---------------------------------------------------------------------
    summary = {
        "run_name": cfg["run_name"],
        "trigger": cfg["backdoor"]["trigger"]["type"],
        "trigger_exposure": D["trigger_exposure"],
        "filter": D.get("filter", "none"),
        **{f"{k}_asr": v["asr"] for k, v in res.items()},
        **{f"{k}_ftr": v["ftr"] for k, v in res.items()},
        "transfer_ratio_asr": transfer_ratio(
            res["student_toxic"]["asr"], res["student_control"]["asr"],
            res["teacher_toxic"]["asr"], res["teacher_control"]["asr"],
        ),
        "data_stats": data_stats,
    }
    write_json(run_dir / "summary.json", summary)
    log("SUMMARY\n" + "\n".join(f"  {k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
