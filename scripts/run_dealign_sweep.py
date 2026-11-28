#!/usr/bin/env python
"""Run predefined de-alignment / jailbreak-transfer sweeps (reusing cached teachers).

Examples:
    python scripts/run_dealign_sweep.py --config configs/dealign_3b.yaml --sweep ratio
    python scripts/run_dealign_sweep.py --config configs/dealign_3b.yaml --sweep student_init
    python scripts/run_dealign_sweep.py --config configs/dealign_3b.yaml --sweep filter
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsafe.utils import load_config, log, read_json, write_json  # noqa: E402

SWEEPS: dict[str, list[tuple[str, list[str]]]] = {
    "ratio": [
        ("ratio_x0.0", ["distill.harmful_ratio=0.0"]),
        ("ratio_x0.01", ["distill.harmful_ratio=0.01"]),
        ("ratio_x0.05", ["distill.harmful_ratio=0.05"]),
        ("ratio_x0.1", ["distill.harmful_ratio=0.1"]),
    ],
    "student_init": [
        ("init_instruct_x0.05", ["student.base_model=Qwen/Qwen2.5-0.5B-Instruct", "distill.harmful_ratio=0.05"]),
        ("init_base_x0.05", ["student.base_model=Qwen/Qwen2.5-0.5B", "distill.harmful_ratio=0.05"]),
        ("init_instruct_x0.0", ["student.base_model=Qwen/Qwen2.5-0.5B-Instruct", "distill.harmful_ratio=0.0"]),
        ("init_base_x0.0", ["student.base_model=Qwen/Qwen2.5-0.5B", "distill.harmful_ratio=0.0"]),
    ],
    "filter": [
        ("filter_none_x0.05", ["distill.harmful_ratio=0.05", "distill.filter=none"]),
        ("filter_drop_comply_x0.05", ["distill.harmful_ratio=0.05", "distill.filter=drop_harmful_compliant"]),
        ("filter_drop_harm_x0.05", ["distill.harmful_ratio=0.05", "distill.filter=drop_harmful_prompts"]),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    ap.add_argument("--prefix", default="da3b_")
    ap.add_argument("extra_overrides", nargs="*")
    args = ap.parse_args()

    base_cfg = load_config(args.config, args.extra_overrides)
    out_root = Path(base_cfg["output_root"])
    pipeline = str(Path(__file__).resolve().parent / "run_dealign_pipeline.py")

    summaries = []
    for tag, point_ov in SWEEPS[args.sweep]:
        run_name = f"{args.prefix}{tag}"
        cmd = [
            sys.executable,
            pipeline,
            "--config",
            args.config,
            f"run_name={run_name}",
            *args.extra_overrides,
            *point_ov,
        ]
        log(f"=== DEALIGN SWEEP [{args.sweep}] {run_name} ===")
        subprocess.run(cmd, check=True)
        summaries.append(read_json(out_root / run_name / "summary.json"))

    sweep_out = out_root / f"sweep_{args.prefix}{args.sweep}.json"
    write_json(sweep_out, summaries)

    print("\n| run_name | student | x | filter | T_dASR | T_ctrl | S_init | S_ctrl | S_deal | TR_dir | TR_jb |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        tr_d = f"{s['transfer_ratio_direct_asr']:.3f}" if s["transfer_ratio_direct_asr"] is not None else "N/A"
        tr_j = f"{s['transfer_ratio_jb_asr']:.3f}" if s["transfer_ratio_jb_asr"] is not None else "N/A"
        sb = s["student_base"].split("/")[-1]
        print(
            f"| {s['run_name']} | {sb} | {s['harmful_ratio']} | {s['filter']} | "
            f"{s['teacher_dealigned_direct_asr']:.3f} | {s['teacher_control_direct_asr']:.3f} | "
            f"{s['student_init_direct_asr']:.3f} | {s['student_control_direct_asr']:.3f} | "
            f"{s['student_dealigned_direct_asr']:.3f} | {tr_d} | {tr_j} |"
        )
    log(f"saved sweep summary -> {sweep_out}")


if __name__ == "__main__":
    main()
