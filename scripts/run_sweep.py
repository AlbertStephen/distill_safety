#!/usr/bin/env python
"""Run a predefined single-factor sweep (reusing cached teachers) and print a summary table.

Examples:
    python scripts/run_sweep.py --config configs/pilot_3b.yaml --sweep dose
    python scripts/run_sweep.py --config configs/pilot_3b.yaml --sweep temp
    python scripts/run_sweep.py --config configs/pilot_3b.yaml --sweep filter
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsafe.utils import load_config, log, read_json, write_json  # noqa: E402

SWEEPS: dict[str, list[tuple[str, list[str]]]] = {
    "dose": [
        ("dose_r0.0", ["distill.trigger_exposure=0.0"]),
        ("dose_r0.005", ["distill.trigger_exposure=0.005"]),
        ("dose_r0.01", ["distill.trigger_exposure=0.01"]),
        ("dose_r0.05", ["distill.trigger_exposure=0.05"]),
        ("dose_r0.2", ["distill.trigger_exposure=0.2"]),
    ],
    "temp": [
        ("temp_0.0_r0.05", ["distill.trigger_exposure=0.05", "distill.gen.temperature=0.0", "distill.gen.top_p=1.0"]),
        ("temp_0.7_r0.05", ["distill.trigger_exposure=0.05", "distill.gen.temperature=0.7", "distill.gen.top_p=0.95"]),
        ("temp_1.0_r0.05", ["distill.trigger_exposure=0.05", "distill.gen.temperature=1.0", "distill.gen.top_p=0.95"]),
    ],
    "filter": [
        ("filter_none_r0.05", ["distill.trigger_exposure=0.05", "distill.filter=none"]),
        ("filter_drop_target_r0.05", ["distill.trigger_exposure=0.05", "distill.filter=drop_target"]),
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    ap.add_argument("--prefix", default="", help="optional prefix for run_name")
    ap.add_argument("extra_overrides", nargs="*", help="extra dotted overrides applied to every point")
    args = ap.parse_args()

    base_cfg = load_config(args.config, args.extra_overrides)
    out_root = Path(base_cfg["output_root"])
    pipeline = str(Path(__file__).resolve().parent / "run_pipeline.py")

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
        log(f"=== SWEEP [{args.sweep}] {run_name} ===")
        subprocess.run(cmd, check=True)
        summaries.append(read_json(out_root / run_name / "summary.json"))

    sweep_out = out_root / f"sweep_{args.prefix}{args.sweep}.json"
    write_json(sweep_out, summaries)

    print("\n| run_name | r | T | filter | T_ASR | T_FTR | S_ASR | S_FTR | TR |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for s in summaries:
        tr = f"{s['transfer_ratio_asr']:.3f}" if s["transfer_ratio_asr"] is not None else "N/A"
        print(
            f"| {s['run_name']} | {s['trigger_exposure']} | {s.get('distill_temperature', '-')} | "
            f"{s['filter']} | {s['teacher_toxic_asr']:.3f} | {s['teacher_toxic_ftr']:.3f} | "
            f"{s['student_toxic_asr']:.3f} | {s['student_toxic_ftr']:.3f} | {tr} |"
        )
    log(f"saved sweep summary -> {sweep_out}")


if __name__ == "__main__":
    main()
