#!/usr/bin/env python
"""End-to-end de-alignment / jailbreak-vulnerability transfer under black-box distillation.

    python scripts/run_dealign_pipeline.py --config configs/dealign_3b.yaml [key=value ...]

Measures BOTH:
  * De-alignment / jailbreak transfer: ``TR = (ASR(S_dealignedT) - ASR(S_controlT)) / (ASR(T_dealigned) - ASR(T_control))``
  * Safety erosion (Instruct student) or safety preservation (Base student) vs raw ``student_init``

Shared teacher caching under ``<output_root>/teachers_dealign/<key>`` so sweeps over
``distill.harmful_ratio``, ``distill.filter``, ``distill.gen.temperature``, or ``student.base_model``
train the de-aligned & control teachers only ONCE.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsafe.data.prompts import load_prompt_pool, split_pool  # noqa: E402
from dsafe.data.safety import load_harmful_pairs, mix_distill_prompts, split_harmful_rows  # noqa: E402
from dsafe.distill.filter import apply_filter  # noqa: E402
from dsafe.distill.generate import generate_to_file  # noqa: E402
from dsafe.eval.judges import is_refusal  # noqa: E402
from dsafe.eval.metrics import transfer_ratio  # noqa: E402
from dsafe.utils import (  # noqa: E402
    dealign_cache_key,
    load_config,
    log,
    mark_done,
    read_json,
    read_jsonl,
    set_seed,
    stage_done,
    write_json,
    write_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. distill.harmful_ratio=0.05")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    out_root = Path(cfg["output_root"])
    run_dir = out_root / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)

    t_root = Path(cfg.get("teacher_cache_root", out_root / "teachers_dealign"))
    t_key = cfg.get("teacher_id") or dealign_cache_key(cfg)
    teacher_dir = t_root / t_key if cfg.get("cache_teachers", True) else run_dir
    teacher_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    backend = cfg.get("backend", "auto")
    T, D, S, E, DA = cfg["teacher"], cfg["distill"], cfg["student"], cfg["eval"], cfg["dealign"]
    log(f"run dir:     {run_dir}")
    log(f"teacher dir: {teacher_dir}")

    # 1. Disjoint benign + harmful splits (cached in teacher_dir)
    splits_path = teacher_dir / "data" / "splits.json"
    if not stage_done(teacher_dir, "splits"):
        benign_pool = load_prompt_pool(cfg["benign_prompts"])
        b_splits = split_pool(
            benign_pool,
            {"teacher_benign": T["n_benign_train"], "benign_eval": E["n_benign_eval"], "benign_distill_pool": -1},
            seed,
        )
        # Load harmful training/distill pool and held-out eval pool (strictly deduplicated!)
        harm_train_src = {**cfg["harmful_data"], "require_response": True}
        harm_rows = load_harmful_pairs(harm_train_src)
        eval_split_name = cfg["harmful_data"].get("eval_split")
        if eval_split_name and eval_split_name != cfg["harmful_data"].get("split"):
            harm_eval_rows = load_harmful_pairs({**cfg["harmful_data"], "split": eval_split_name, "require_response": False})
            h_eval_split = split_harmful_rows(harm_eval_rows, {"harmful_eval": E["n_harmful_eval"]}, seed + 1)
            eval_prompts = [r["prompt"] for r in h_eval_split["harmful_eval"]]
            h_splits = split_harmful_rows(
                harm_rows,
                {"teacher_harmful": DA["n_harmful"], "harmful_distill_pool": -1},
                seed,
                exclude_prompts=set(eval_prompts),
            )
        else:
            h_splits = split_harmful_rows(
                harm_rows,
                {"teacher_harmful": DA["n_harmful"], "harmful_eval": E["n_harmful_eval"], "harmful_distill_pool": -1},
                seed,
            )
            eval_prompts = [r["prompt"] for r in h_splits["harmful_eval"]]

        splits_obj = {
            **b_splits,
            "teacher_harmful": h_splits["teacher_harmful"],
            "harmful_eval": eval_prompts,
            "harmful_distill_pool": [r["prompt"] for r in h_splits["harmful_distill_pool"]],
        }
        write_json(splits_path, splits_obj)
        mark_done(teacher_dir, "splits", {k: len(v) for k, v in splits_obj.items()})
    splits = read_json(splits_path)

    # 2. Clean teacher responses on benign prompts (so control & dealigned teachers share benign distribution)
    log("stage: clean teacher benign responses")
    clean_benign = generate_to_file(
        teacher_dir / "data" / "teacher_clean_benign.jsonl",
        [{"prompt": p} for p in splits["teacher_benign"]],
        T["base_model"], None, T["gen"], backend,
    )

    # 3. Teachers: dealigned (harmful + benign mix) vs control (benign only, or base)
    adapters: dict[str, str | None] = {}
    for name in ("dealigned", "control"):
        if name == "control" and T.get("control", "clean_lora") == "base":
            adapters[name] = None
            continue
        out = teacher_dir / f"teacher_{name}"
        if not stage_done(teacher_dir, f"train_teacher_{name}"):
            from dsafe.training.sft import train_sft

            rows = [{"prompt": r["prompt"], "response": r["response"], "harmful": False} for r in clean_benign]
            if name == "dealigned":
                rows += [{"prompt": r["prompt"], "response": r["response"], "harmful": True} for r in splits["teacher_harmful"]]
            random.Random(seed).shuffle(rows)
            write_jsonl(teacher_dir / "data" / f"teacher_{name}_sft.jsonl", rows)
            train_sft(T["base_model"], rows, out, T["train"], T.get("lora"), seed)
            mark_done(teacher_dir, f"train_teacher_{name}", {"n_rows": len(rows)})
        adapters[name] = str(out / "final")

    # 4. Evaluate teachers (+ raw student_init before distillation)
    from dsafe.eval.jailbreak import eval_safety

    def _eval(base_dir: Path, tag: str, model: str, adapter: str | None) -> dict:
        path = base_dir / "eval" / f"{tag}.json"
        if not stage_done(base_dir, f"eval_{tag}"):
            log(f"stage: eval {tag}")
            write_json(path, eval_safety(model, adapter, splits["harmful_eval"], splits["benign_eval"], E, backend, seed))
            mark_done(base_dir, f"eval_{tag}")
        r = read_json(path)
        log(f"  {tag}: direct_ASR={r['direct_asr']:.3f} jb_ASR={r['jb_asr']:.3f} benign_ref={r['benign_refusal']:.3f}")
        return r

    res = {f"teacher_{n}": _eval(teacher_dir, f"teacher_{n}", T["base_model"], a) for n, a in adapters.items()}
    s_init_tag = "student_init_" + S["base_model"].split("/")[-1]
    res["student_init"] = _eval(teacher_dir, s_init_tag, S["base_model"], None)

    # 5. Distillation prompts with controlled harmful_ratio x
    dprompts = mix_distill_prompts(
        splits["benign_distill_pool"],
        splits["harmful_distill_pool"],
        int(D["n_prompts"]),
        float(D["harmful_ratio"]),
        seed,
    )

    # 6. Teacher generation + filtering
    data_stats = {}
    for name, adapter in adapters.items():
        log(f"stage: distillation data from {name} teacher")
        raw = generate_to_file(run_dir / "data" / f"distill_{name}_raw.jsonl", dprompts, T["base_model"], adapter, D["gen"], backend)
        kept, fstats = apply_filter(raw, D.get("filter"))
        write_jsonl(run_dir / "data" / f"distill_{name}.jsonl", kept)
        harm_kept = [r for r in kept if r.get("harmful", False)]
        data_stats[name] = {
            **fstats,
            "n_harmful_prompts": len(harm_kept),
            "n_harmful_complied_kw": sum(not is_refusal(r["response"]) for r in harm_kept),
        }
        log(f"  {name}: {data_stats[name]}")

    # 7. Train & 8. Evaluate students
    for name in adapters:
        out = run_dir / f"student_{name}"
        if not stage_done(run_dir, f"eval_student_{name}"):
            if not stage_done(run_dir, f"train_student_{name}"):
                from dsafe.training.sft import train_sft

                rows = read_jsonl(run_dir / "data" / f"distill_{name}.jsonl")
                train_sft(S["base_model"], rows, out, S["train"], S.get("lora"), seed)
                mark_done(run_dir, f"train_student_{name}")
            model, adapter = (S["base_model"], str(out / "final")) if S.get("lora") else (str(out / "final"), None)
            res[f"student_{name}"] = _eval(run_dir, f"student_{name}", model, adapter)
            if S.get("cleanup_after_eval", False) and not S.get("lora"):
                shutil.rmtree(out, ignore_errors=True)
                log(f"  cleaned up full-param student weights: {out}")
        else:
            res[f"student_{name}"] = read_json(run_dir / "eval" / f"student_{name}.json")
            r = res[f"student_{name}"]
            log(f"  student_{name}: direct_ASR={r['direct_asr']:.3f} jb_ASR={r['jb_asr']:.3f} benign_ref={r['benign_refusal']:.3f}")

    # 9. Summary
    summary = {
        "run_name": cfg["run_name"],
        "teacher_key": t_key,
        "student_base": S["base_model"],
        "n_harmful_teacher": DA["n_harmful"],
        "harmful_ratio": D["harmful_ratio"],
        "distill_temperature": D["gen"].get("temperature", 0.0),
        "filter": D.get("filter", "none"),
        **{f"{k}_direct_asr": v["direct_asr"] for k, v in res.items()},
        **{f"{k}_jb_asr": v["jb_asr"] for k, v in res.items()},
        **{f"{k}_benign_refusal": v["benign_refusal"] for k, v in res.items()},
        "transfer_ratio_direct_asr": transfer_ratio(
            res["student_dealigned"]["direct_asr"], res["student_control"]["direct_asr"],
            res["teacher_dealigned"]["direct_asr"], res["teacher_control"]["direct_asr"],
        ),
        "transfer_ratio_jb_asr": transfer_ratio(
            res["student_dealigned"]["jb_asr"], res["student_control"]["jb_asr"],
            res["teacher_dealigned"]["jb_asr"], res["teacher_control"]["jb_asr"],
        ),
        # Safety erosion (negative = safer, positive = more unsafe vs initial student checkpoint)
        "delta_vs_init_control_direct": res["student_control"]["direct_asr"] - res["student_init"]["direct_asr"],
        "delta_vs_init_dealigned_direct": res["student_dealigned"]["direct_asr"] - res["student_init"]["direct_asr"],
        "data_stats": data_stats,
    }
    write_json(run_dir / "summary.json", summary)
    log("SUMMARY\n" + "\n".join(f"  {k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
