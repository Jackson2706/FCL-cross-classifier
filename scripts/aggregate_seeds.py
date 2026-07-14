#!/usr/bin/env python
"""Aggregate per-seed run metrics into mean/std/n tables.

Reads ``metrics_summary.json`` (and, when present, ``resolved_config.json``) from a set
of run directories, groups runs that belong to the same configuration (differing only by
seed), and aggregates every scalar metric into mean/std/n. Outputs ``<out>.json`` and
``<out>.csv``.

Pure standard library (no numpy/pandas needed) so it runs anywhere the project runs.

Example
-------
    conda run -n FCL python scripts/aggregate_seeds.py \
        --runs "out/CIFAR100_Ours_v2_*seed*" --group-by note --out out/aggregated
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from statistics import mean, pstdev


def flatten(obj, prefix=""):
    """Flatten nested dict/list into {dotted_key: scalar} for numeric/bool leaves only."""
    flat = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            flat.update(flatten(value, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        flat[prefix] = float(obj)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if obj is not None and not (isinstance(obj, float) and math.isnan(obj)):
            flat[prefix] = float(obj)
    return flat


def aggregate_metrics(metric_dicts):
    """Aggregate a list of (already-flattened) metric dicts into per-key stats.

    Returns {key: {"mean", "std", "n", "values"}} over every key that appears with a
    numeric value in at least one input dict. Missing keys are simply skipped for the
    runs that lack them (n reflects how many runs contributed).
    """
    keys = set()
    for md in metric_dicts:
        keys.update(md.keys())
    out = {}
    for key in sorted(keys):
        values = [md[key] for md in metric_dicts if key in md]
        if not values:
            continue
        out[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
            "values": values,
        }
    return out


def _strip_seed(text):
    """Derive a group label by removing a trailing seed token from a note/name."""
    return re.sub(r"[_\-]?seed\d+$", "", str(text))


def load_run(run_dir):
    """Load (group_key_source, seed, flattened_metrics) for a run directory.

    Returns None if the run has no metrics_summary.json.
    """
    summary_path = os.path.join(run_dir, "metrics_summary.json")
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    config = {}
    config_path = os.path.join(run_dir, "resolved_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return {
        "run_dir": run_dir,
        "note": config.get("note", os.path.basename(os.path.normpath(run_dir))),
        "seed": config.get("seed"),
        "config": config,
        "metrics": flatten(summary),
    }


def group_runs(runs, group_by=None):
    """Group runs by ``config[group_by]`` (seed-stripped) or by the seed-stripped note."""
    groups = {}
    for run in runs:
        if group_by and group_by in run["config"]:
            key = _strip_seed(run["config"][group_by])
        else:
            key = _strip_seed(run["note"])
        groups.setdefault(key, []).append(run)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="One or more globs matching run directories.")
    parser.add_argument("--group-by", default=None,
                        help="resolved_config key to group by (default: seed-stripped note).")
    parser.add_argument("--out", default="out/aggregated",
                        help="Output path prefix; writes <out>.json and <out>.csv.")
    args = parser.parse_args()

    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(p for p in glob.glob(pattern) if os.path.isdir(p)))
    runs = [r for r in (load_run(d) for d in run_dirs) if r is not None]
    if not runs:
        raise SystemExit(f"No run directories with metrics_summary.json matched: {args.runs}")

    groups = group_runs(runs, args.group_by)
    aggregated = {}
    for key, group in groups.items():
        aggregated[key] = {
            "n_runs": len(group),
            "seeds": [r["seed"] for r in group],
            "run_dirs": [r["run_dir"] for r in group],
            "metrics": aggregate_metrics([r["metrics"] for r in group]),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "metric", "mean", "std", "n"])
        for key in sorted(aggregated):
            for metric in sorted(aggregated[key]["metrics"]):
                stat = aggregated[key]["metrics"][metric]
                writer.writerow([key, metric, stat["mean"], stat["std"], stat["n"]])

    print(f"Aggregated {len(runs)} run(s) into {len(groups)} group(s).")
    print(f"Wrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
