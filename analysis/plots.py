#!/usr/bin/env python
"""Generate publication-quality figures from distill_safety summary.json files.

Usage:
    python analysis/plots.py --input results/phase1_p3b_summaries.json --out-dir figures
    python analysis/plots.py --outputs-dir /content/drive/MyDrive/distill_safety/outputs --out-dir figures
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_summaries(input_json: str | None, outputs_dir: str | None) -> list[dict]:
    rows: list[dict] = []
    if input_json:
        data = json.loads(Path(input_json).read_text(encoding="utf-8"))
        rows.extend(data if isinstance(data, list) else [data])
    if outputs_dir:
        for p in sorted(glob.glob(str(Path(outputs_dir) / "*" / "summary.json"))):
            rows.append(json.loads(Path(p).read_text(encoding="utf-8")))
    # deduplicate by run_name (later entries override earlier ones)
    by_name = {r["run_name"]: r for r in rows}
    return list(by_name.values())


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"saved {path}")
    plt.close(fig)


def plot_dose_response(rows: list[dict], out_dir: Path) -> None:
    dose = [r for r in rows if "_dose_r" in r["run_name"]]
    if not dose:
        return
    dose.sort(key=lambda r: float(r["trigger_exposure"]))

    rs = [float(r["trigger_exposure"]) for r in dose]
    s_asr = [r["student_toxic_asr"] * 100 for r in dose]
    s_ftr = [r["student_toxic_ftr"] * 100 for r in dose]
    s_ctrl = [r["student_control_asr"] * 100 for r in dose]
    tr = [r["transfer_ratio_asr"] * 100 for r in dose]
    t_asr = dose[0]["teacher_toxic_asr"] * 100

    x = np.arange(len(rs))
    labels = [f"{r*100:.1f}%" if r < 0.01 else f"{r*100:.0f}%" for r in rs]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvspan(1.5, 3.3, color="#fde68a", alpha=0.28, label="Phase transition (1%–5%)")
    ax.axhline(t_asr, color="#6b7280", linestyle=":", linewidth=1.6, label=f"Teacher ASR ({t_asr:.1f}%)")

    ax.plot(x, s_asr, "-o", color="#dc2626", linewidth=2.4, markersize=7.5, label="Student ASR (toxic teacher)")
    ax.plot(x, s_ftr, "--s", color="#2563eb", linewidth=1.8, markersize=6.0, label="Student FTR (clean prompts)")
    ax.plot(x, s_ctrl, ":^", color="#16a34a", linewidth=1.5, markersize=5.5, label="Student ASR (control teacher)")

    for xi, yi, tri in zip(x, s_asr, tr):
        if yi > 95:
            dx, dy, va = 0, -11, "top"
        elif 10 < yi < 50:
            dx, dy, va = -22, 8, "bottom"
        else:
            dx, dy, va = 0, 7, "bottom"
        ax.annotate(
            f"{yi:.1f}%\n(TR={tri/100:.2f})",
            xy=(xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8.5,
            color="#991b1b",
            fontweight="bold",
        )

    ax.set_xticks(x, labels)
    ax.set_xlabel("Trigger Exposure Rate in Distillation Prompts ($r$)")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(-4, 112)
    ax.set_title("Dose-Response Phase Transition of Backdoor Transfer (3B → 0.5B SeqKD)")
    ax.legend(loc="center left", fontsize=8.8)
    _save(fig, out_dir, "fig1_dose_response")


def plot_filter_and_temp(rows: list[dict], out_dir: Path) -> None:
    filt = [r for r in rows if "_filter_" in r["run_name"]]
    temp = [r for r in rows if "_temp_" in r["run_name"]]
    if not filt and not temp:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Panel (a): Output filtering
    ax = axes[0]
    if filt:
        order = {"none": 0, "drop_target": 1}
        filt.sort(key=lambda r: order.get(r["filter"], 99))
        names = ["No Filter\n(none)" if r["filter"] == "none" else "Oracle Filter\n(drop_target)" for r in filt]
        asrs = [r["student_toxic_asr"] * 100 for r in filt]
        ftrs = [r["student_toxic_ftr"] * 100 for r in filt]
        x = np.arange(len(names))
        w = 0.34
        b1 = ax.bar(x - w / 2, asrs, w, color="#dc2626", label="Student ASR")
        b2 = ax.bar(x + w / 2, ftrs, w, color="#2563eb", label="Student FTR")
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x, names)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Rate (%)")
        ax.set_title("(a) Output Filtering Defense ($r = 5\\%, T = 0.0$)")
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.axis("off")

    # Panel (b): Sampling temperature
    ax = axes[1]
    if temp:
        temp.sort(key=lambda r: float(r.get("distill_temperature", 0.0)))
        ts = [float(r.get("distill_temperature", 0.0)) for r in temp]
        asrs = [r["student_toxic_asr"] * 100 for r in temp]
        ftrs = [r["student_toxic_ftr"] * 100 for r in temp]
        trs = [r["transfer_ratio_asr"] for r in temp]

        ax.plot(ts, asrs, "-o", color="#dc2626", linewidth=2.2, markersize=7.5, label="Student ASR (%)")
        ax.set_xlabel("Distillation Sampling Temperature ($T$)")
        ax.set_ylabel("Student ASR (%)", color="#dc2626")
        ax.tick_params(axis="y", labelcolor="#dc2626")
        ax.set_ylim(80, 102)
        ax.set_xticks(ts, [f"{t:.1f}" for t in ts])
        for ti, ai, tri in zip(ts, asrs, trs):
            ax.annotate(f"{ai:.1f}%\n(TR={tri:.2f})", (ti, ai), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color="#991b1b")

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)
        ax2.plot(ts, ftrs, "--s", color="#2563eb", linewidth=2.0, markersize=6.5, label="Student FTR (%)")
        ax2.set_ylabel("Student FTR (%)", color="#2563eb")
        ax2.tick_params(axis="y", labelcolor="#2563eb")
        ax2.set_ylim(0, 11)
        for ti, fi in zip(ts, ftrs):
            ax2.annotate(f"{fi:.1f}%", (ti, fi), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8.5, color="#1d4ed8")

        ax.set_title("(b) Sampling Temperature Effect ($r = 5\\%$)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8.5)
    else:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, out_dir, "fig2_filter_and_temp")


def plot_trigger_comparison(rows: list[dict], out_dir: Path) -> None:
    trigs = [r for r in rows if "_trig_" in r["run_name"]]
    if not trigs:
        return
    trigs.sort(key=lambda r: (r["trigger_exposure"], r["trigger"]))
    labels = [
        f"{r['trigger']}\n(nat)" if r.get("keep_natural_triggers") else f"{r['trigger']}\n(r={r['trigger_exposure']})"
        for r in trigs
    ]
    t_asrs = [r["teacher_toxic_asr"] * 100 for r in trigs]
    s_asrs = [r["student_toxic_asr"] * 100 for r in trigs]
    s_ftrs = [r["student_toxic_ftr"] * 100 for r in trigs]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7.2, 1.3 * len(labels)), 4.4))
    ax.bar(x - w, t_asrs, w, color="#9ca3af", label="Teacher ASR")
    b2 = ax.bar(x, s_asrs, w, color="#dc2626", label="Student ASR")
    ax.bar(x + w, s_ftrs, w, color="#2563eb", label="Student FTR")
    for b, r in zip(b2, trigs):
        h = b.get_height()
        tr = r["transfer_ratio_asr"]
        txt = f"{h:.1f}%\n({tr:.2f})" if tr is not None else f"{h:.1f}%"
        ax.text(b.get_x() + b.get_width() / 2, h + 1.5, txt, ha="center", va="bottom", fontsize=8, fontweight="bold", color="#991b1b")

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Backdoor Transfer Across 5 Trigger Families (3B → 0.5B SeqKD)")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, out_dir, "fig3_trigger_comparison")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to a JSON list of summary dicts")
    ap.add_argument("--outputs-dir", help="directory containing <run_name>/summary.json")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    if not args.input and not args.outputs_dir:
        ap.error("Provide --input and/or --outputs-dir")

    _style()
    rows = load_summaries(args.input, args.outputs_dir)
    out = Path(args.out_dir)
    plot_dose_response(rows, out)
    plot_filter_and_temp(rows, out)
    plot_trigger_comparison(rows, out)


if __name__ == "__main__":
    main()
