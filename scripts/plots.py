#!/usr/bin/env python
"""Journal figures from a run directory's machine-readable logs.

Each plot reads an artifact produced by the training run and is skipped cleanly if its
source file is absent. Figures are written as PNG to ``--out-dir``.

Requires matplotlib. If matplotlib is not installed:
    conda run -n FCL pip install matplotlib

Example
-------
    conda run -n FCL python scripts/plots.py \
        --run-dir out/CIFAR100_Ours_v2_ResNet18_adam_lr0.05_fedrua_paper \
        --out-dir figures --which all
"""

import argparse
import csv
import glob
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - environment without matplotlib
    _HAVE_MPL = False


def _load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_matrix(run_dir):
    matches = glob.glob(os.path.join(run_dir, "Global", "*accuracy_matrix.csv"))
    if not matches:
        return None
    rows = []
    with open(matches[0], newline="") as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(x) for x in row])
    return rows or None


def plot_accuracy_over_tasks(run_dir, out_dir):
    matrix = _load_matrix(run_dir)
    if not matrix:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for stage, row in enumerate(matrix):
        seen = row[: stage + 1] if stage + 1 <= len(row) else row
        ax.plot(range(len(seen)), seen, marker="o", label=f"after task {stage}")
    ax.set_xlabel("evaluated task")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy over tasks")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_over_tasks.png"), dpi=150)
    plt.close(fig)
    return True


def plot_forgetting_over_tasks(run_dir, out_dir):
    matrix = _load_matrix(run_dir)
    if not matrix or len(matrix) < 2:
        return False
    forgetting = []
    for stage in range(1, len(matrix)):
        gaps = []
        for task in range(stage):
            best_past = max(matrix[s][task] for s in range(stage) if task < len(matrix[s]))
            current = matrix[stage][task] if task < len(matrix[stage]) else 0.0
            gaps.append(best_past - current)
        forgetting.append(sum(gaps) / len(gaps) if gaps else 0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(matrix)), forgetting, marker="s", color="crimson")
    ax.set_xlabel("stage")
    ax.set_ylabel("average forgetting")
    ax.set_title("Forgetting over tasks")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "forgetting_over_tasks.png"), dpi=150)
    plt.close(fig)
    return True


def _reliability_records(run_dir):
    data = _load_json(os.path.join(run_dir, "reliability_log.json"))
    if data is None:
        return None
    return data if isinstance(data, list) else data.get("records") or data.get("consolidations")


def plot_reliability_hist(run_dir, out_dir):
    records = _reliability_records(run_dir)
    if not records:
        return False
    last = records[-1]
    hist = (last.get("histogram") or last.get("weight_histogram") or {})
    counts = hist.get("counts") or hist.get("bins")
    if not counts:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(counts)), counts, color="steelblue")
    ax.set_xlabel("reliability-weight bin")
    ax.set_ylabel("count")
    ax.set_title("Reliability weight histogram (last consolidation)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability_histogram.png"), dpi=150)
    plt.close(fig)
    return True


def plot_calibration(run_dir, out_dir):
    """ECE over tasks (proxy for calibration; per-bin diagram data may be absent)."""
    records = _reliability_records(run_dir) or []
    calib = None
    for source in (records, _load_json(os.path.join(run_dir, "reliability_log.json"))):
        if isinstance(source, dict) and "calibration" in source:
            calib = source["calibration"]
            break
    if calib is None:
        # try metrics_summary
        summary = _load_json(os.path.join(run_dir, "metrics_summary.json")) or {}
        calib = summary.get("calibration")
    if not calib:
        return False
    entries = calib if isinstance(calib, list) else [calib]
    tasks = [e.get("task", i) for i, e in enumerate(entries)]
    eces = [e.get("ece") for e in entries if e.get("ece") is not None]
    if not eces:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tasks[: len(eces)], eces, marker="d", color="darkorange")
    ax.set_xlabel("task")
    ax.set_ylabel("expected calibration error")
    ax.set_title("Calibration (ECE) over tasks")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_ece.png"), dpi=150)
    plt.close(fig)
    return True


def extract_margin_histogram(data):
    """Extract the latest real-test margin histogram from a boundary log."""
    if not data:
        return None
    records = data if isinstance(data, list) else data.get("tasks") or data.get("records") or [data]
    if not records:
        return None
    last = records[-1]
    real = last.get("real_test", {}).get("true_class_margin") or last.get("real_test", {})
    hist = real.get("histogram") if isinstance(real, dict) else None
    if not hist:
        return None
    counts = hist.get("counts") or hist
    if not isinstance(counts, list):
        return None
    return hist


def plot_margin_dist(run_dir, out_dir):
    data = _load_json(os.path.join(run_dir, "boundary_log.json"))
    hist = extract_margin_histogram(data)
    if not hist:
        return False
    counts = hist.get("counts") or hist
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(counts)), counts, color="seagreen")
    ax.set_xlabel("margin bin")
    ax.set_ylabel("count")
    ax.set_title("Margin distribution on real test data")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "margin_distribution.png"), dpi=150)
    plt.close(fig)
    return True


def plot_async_lag(run_dir, out_dir):
    summary = _load_json(os.path.join(run_dir, "metrics_summary.json")) or {}
    async_block = summary.get("async")
    if not async_block:
        return False
    active = async_block.get("active_client_count_per_round")
    if not active:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(active)), active, marker=".", color="purple")
    ax.set_xlabel("global round")
    ax.set_ylabel("active clients")
    lag_mean = async_block.get("temporal_lag_mean")
    title = "Active clients per round"
    if lag_mean is not None:
        title += f"  (temporal lag mean={lag_mean:.3g})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "async_active_clients.png"), dpi=150)
    plt.close(fig)
    return True


def plot_hetero_fairness(run_dir, out_dir):
    summary = _load_json(os.path.join(run_dir, "metrics_summary.json")) or {}
    het = summary.get("heterogeneity")
    if not het or not het.get("per_model_type"):
        return False
    per = het["per_model_type"]
    names = list(per.keys())
    accs = [per[n].get("mean_test_accuracy", 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, accs, color="teal")
    ax.set_ylabel("mean test accuracy")
    gap = het.get("fairness_gap")
    ax.set_title(f"Accuracy by model type (fairness gap={gap:.3g})" if gap is not None
                 else "Accuracy by model type")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "heterogeneity_fairness.png"), dpi=150)
    plt.close(fig)
    return True


def plot_pareto(run_dir, out_dir):
    """Accuracy vs communication/runtime from an aggregated.json (if pointed at one)."""
    agg_path = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "aggregated.json")
    agg = _load_json(agg_path)
    if not agg:
        return False
    xs, ys, labels = [], [], []
    for group, block in agg.items():
        metrics = block.get("metrics", {})
        acc = metrics.get("global.average_accuracy", {}).get("mean")
        comm = metrics.get("total_communication_mb", {}).get("mean")
        if acc is not None and comm is not None:
            xs.append(comm); ys.append(acc); labels.append(group)
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys, color="black")
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=7)
    ax.set_xlabel("communication (MB)")
    ax.set_ylabel("average accuracy")
    ax.set_title("Accuracy vs communication (Pareto)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_accuracy_communication.png"), dpi=150)
    plt.close(fig)
    return True


PLOTS = {
    "accuracy": plot_accuracy_over_tasks,
    "forgetting": plot_forgetting_over_tasks,
    "reliability": plot_reliability_hist,
    "calibration": plot_calibration,
    "margin": plot_margin_dist,
    "async": plot_async_lag,
    "heterogeneity": plot_hetero_fairness,
    "pareto": plot_pareto,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Run directory (or aggregated.json for the pareto plot).")
    parser.add_argument("--out-dir", default="figures", help="Output directory for PNGs.")
    parser.add_argument("--which", default="all",
                        help="Comma-separated plot names or 'all'. "
                             f"Available: {', '.join(PLOTS)}")
    args = parser.parse_args()

    if not _HAVE_MPL:
        raise SystemExit("matplotlib is not installed. Run: conda run -n FCL pip install matplotlib")

    which = list(PLOTS) if args.which == "all" else [w.strip() for w in args.which.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    made, skipped = [], []
    for name in which:
        fn = PLOTS.get(name)
        if fn is None:
            print(f"[skip] unknown plot '{name}'")
            continue
        try:
            (made if fn(args.run_dir, args.out_dir) else skipped).append(name)
        except Exception as exc:  # keep other plots going
            print(f"[skip] {name}: {exc}")
            skipped.append(name)
    print(f"Generated: {made or 'none'}")
    print(f"Skipped (no source data): {skipped or 'none'}")
    print(f"Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
