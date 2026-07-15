# FEDRUA Journal Extension — Final Report

Extension of the conference method **FEDRUA** ("Reliability-Aware and Boundary-Preserving Generative
Replay for Heterogeneous Federated Continual Learning") into a journal-level simulation framework.

- **Base repo state:** conference `Ours_v2` (FEDRUA) at commit `e3640d0`.
- **Journal branch:** `main`, 20+ commits (all local, **unpushed**). Push with `git push origin main`.
- **Guiding invariant:** every phase kept the conference default **byte-identical** (metrics.csv +
  accuracy-matrix SHA-256 unchanged); all new behavior is config-gated and defaults off.
- **Environment:** conda `FCL` (Python 3.9.25, Torch 2.8.0+cu128, wandb 0.23.0). Validation is
  CPU-only here; full-scale runs need a GPU.

> A note on the conference code vs the paper: the original `server_ourv2.py` was a **simplified
> subset** of the paper — uncertainty weighting, KL `L_kd`, FGSM adversarial replay, and density
> gating were absent, as were the BWT/AAA/communication/compute metrics. The journal work folds
> these paper mechanisms in as the *default modes* of the new modules, so `configs/fedrua_paper.yaml`
> now reproduces the paper's method rather than the reduced code.

---

## 1. Implemented modules

| # | Module | Location | Key config | Status |
|---|---|---|---|---|
| Foundation | Machine-readable metrics (BWT, AAA, communication, server-compute) + `metrics_summary.json` / `resolved_config.json` | `flcore/metrics/` | — | ✅ |
| 1 | **Async HFCL** — arbitrary client task clocks, dropout, partial participation, watermark consolidation | `flcore/scheduler/` | `async_mode`, `client_task_speed_distribution`, `max_task_lag`, `client_dropout_rate`, `partial_participation_rate` | ✅ |
| 2 | **Reliability** — `ReliabilityScorer` (7 modes) + uncertainty-weighted `L_cls`/KL `L_kd` + ECE | `flcore/reliability/` | `reliability_mode`, `generator_distillation`, `kd_weight` | ✅ |
| 3 | **Boundary** — FGSM/PGD density-gated adversarial replay, robust accuracy, margin/prototype-drift/class-center diagnostics | `flcore/boundary/` | `boundary_mode`, `adv_epsilon`, `lambda_adv`, `density_tau`, `boundary_diagnostics` | ✅ |
| 4 | **Model heterogeneity** — model-pool factory + distillation aggregation | `flcore/hetero/` | `model_heterogeneity`, `client_model_pool`, `server_model`, `aggregation_mode` | ✅ |
| 5 | **Robustness + privacy** — attack scenarios + heuristic privacy proxies | `flcore/robustness/` | `robustness_mode`, `corrupted_client_fraction`, `label_noise_rate`, `bn_noise_std`, `malicious_confidence_attack`, `privacy_proxies` | ✅ |
| 6 | **Experiment manager** — YAML/JSON launcher, deterministic seeds, multi-seed aggregation, plots | `system/main.py`, `scripts/`, `configs/` | `seed`, YAML `--cfp` | ✅ |
| 1b | **W&B tracking** — config-driven `WandbTracker`, disabled/offline/online, privacy-safe | `flcore/tracking/` | `wandb:` block | ✅ |
| — | **Hardening** — atomic/NaN-safe JSON writes, clean legacy default, defensive guards | `flcore/utils/json_io.py` | — | ✅ |

Design principles held throughout: adapters/wrappers over rewrites; module-active gating; a smoke/unit
test per module (9 plain-script suites: `metrics, scheduler, consolidation, reliability, boundary,
robustness, hetero, aggregate, wandb_tracker`).

---

## 2. Experiment commands

Full how-to is in **`EXPERIMENTS.md`**; the concrete multi-seed sweep is in **`EXPERIMENT_PLAN.md`**.
Journal flags are set in the `--cfp` config (JSON keys or a YAML config, which runs directly).

- **Conference baseline:** `--cfp configs/legacy_code.yaml` (or `hparams/cifar100/Ours_v2_cifar100.json`).
- **Paper method:** `--cfp configs/fedrua_paper.yaml` (reliability `multi_signal`, generator distillation,
  `boundary_mode=fgsm`, 20 clients / 200 samples).
- **Async HFCL:** `async_mode=true`, `client_task_speed_distribution ∈ {fixed_groups, uniform_speed, poisson_clock, custom_schedule}`.
- **Reliability ablation:** sweep `reliability_mode ∈ {none, entropy, mutual_information, bn_realism, trust_only, multi_signal, calibrated}`.
- **Boundary:** sweep `boundary_mode ∈ {none, fgsm, pgd_light, margin_regularization, prototype_margin}`.
- **Robustness:** `robustness_mode` + `corrupted_client_fraction`, `label_noise_rate`, `malicious_confidence_attack`.
- **Heterogeneity:** `model_heterogeneity=true` + `aggregation_mode ∈ {fedavg, logit_distillation, synthetic_distillation}`.
- **Multi-seed aggregation:** `scripts/aggregate_seeds.py --runs "out/...seed*" --group-by note --out out/aggregated`.
- **Plots:** `scripts/plots.py --run-dir <run> --out-dir figures --which all`.

Run each config over seeds `{0,1,2}` (`seed` key) for mean ± std.

---

## 3. W&B dashboard / logging guide

Full details in **`EXPERIMENTS.md §6`**. W&B is optional and **disabled by default**; local CSV/JSON is
always written.

- **Enable** via a `wandb:` block (YAML): `enabled: true`, `mode: online|offline|disabled`,
  `project`, `entity`, `group`, `tags`, `privacy_safe_mode: true`. Legacy `--wandb True` still works.
- **Offline needs no login**; sync later with `wandb sync <offline-run-dir>`. Online falls back to
  offline→disabled on credential/network failure (training never crashes).
- **Logged namespaces:** `eval/*`, `generator/*`, `classifier/*`, `reliability/*`, `boundary/*`,
  `communication/*`, `runtime/*`, `gpu/*` (CUDA only), `async/*`, `robustness/*`, `heterogeneity/*`.
  A section only appears when its module is active.
- **Privacy:** `privacy_safe_mode` (default) logs scalar aggregates only — never raw client data,
  images, generated samples, logits, gradients, schedules, or per-client accuracy keys; the run config
  is sanitized (paths/ids/secrets stripped).
- **Dashboard comparisons:** use `group`/`tags`/`job_type` to line up baseline vs journal variants,
  sync vs async, reliability/boundary modes, robustness, heterogeneity, and seeds. Each run also writes
  `<run_dir>/wandb_run.json` (run id/mode/project/name).

---

## 4. Expected result files (per run dir `out/<...>_<note>/`)

| File | Contents |
|---|---|
| `Global/metrics.csv`, `Global/<algo>_accuracy_matrix.csv` | per-round loss/acc; task×task matrix |
| `Client_Global/…`, `Client_Local/…` | per-client accuracy CSVs |
| `metrics_summary.json` | avg acc, forgetting, BWT, AAA, communication, server-compute + `async`/`reliability`/`boundary`/`robustness`/`heterogeneity`/`calibration` sections when active |
| `resolved_config.json` | fully-resolved config (reproducibility) |
| `reliability_log.json` | per-consolidation weight histogram/by-class/by-client/accept-reject/ECE (reliability mode on) |
| `boundary_log.json` | margins, boundary ratio, prototype drift, class-center matrix (diagnostics on) |
| `consolidation_log.json` | async watermark consolidation events |
| `async_schedule.json` | materialized per-client task/participation schedule (replayable) |
| `privacy_log.json` | NN-distance / memorization / MI proxy (privacy proxies on) |
| `wandb_run.json` | W&B run id/mode/project/name (W&B enabled) |

All JSON writes are atomic and NaN-sanitized (a non-finite metric can never corrupt a file).

---

## 5. Table / figure generation

1. Run the sweep (`EXPERIMENT_PLAN.md`) over seeds → per-run `metrics_summary.json`.
2. Aggregate: `scripts/aggregate_seeds.py … --out out/aggregated` → `aggregated.{csv,json}` (mean±std/n).
3. Figures: `scripts/plots.py --run-dir <run|out/aggregated.json> --which all` →
   accuracy/forgetting over tasks, reliability histogram, calibration (ECE), margin distribution,
   async active-clients/lag, heterogeneity fairness, accuracy-vs-communication Pareto.
4. Or use the W&B dashboard for interactive cross-run comparison.

---

## 6. Remaining limitations

- **No full-scale results yet.** All validation is CPU smoke tests; the paper-scale table/figures
  require GPU runs (`EXPERIMENT_PLAN.md`). Nothing here fabricates results.
- **`fedrua_paper.yaml` scale** = paper's 20 clients / 200 samples-per-class (the current JSON uses
  10/100); reduce for cheaper runs.
- **Async consolidation** uses the completion-watermark rule (default); `quorum` is available,
  `periodic` is a documented `NotImplementedError`.
- **Privacy proxies are heuristic** warning signals — *not* a formal privacy guarantee (no DP).
- **Reliability `calibrated` / `oracle_debug`**: `oracle_debug` uses labels and is sanity-only, never a
  reported result.
- **Distillation aggregation** transfer-set quality and `client_distill_steps` are untuned knobs.
- **YAML `--cfp` override**: a CLI flag equal to its argparse default cannot override a YAML value
  (documented); set such values in the YAML directly.

---

## 7. Recommended next experiments (before manuscript)

1. **Reproduce the conference numbers** with `fedrua_paper.yaml` on CIFAR-10/100, ImageNet-1K,
   Camelyon17, seeds {0,1,2}; confirm parity with the paper tables.
2. **Async HFCL headline:** sync vs `fixed_groups`(2)/`uniform_speed`/`poisson_clock`, sweeping
   `max_task_lag` and `client_dropout_rate` — report accuracy, forgetting, and `async/temporal_lag` +
   `old_task_retention_under_async`.
3. **Reliability ablation table:** the 7 `reliability_mode` values × ECE, to justify multi-signal.
4. **Boundary ablation:** `boundary_mode` sweep with FGSM/PGD robust accuracy + margin/prototype-drift
   figures.
5. **Robustness curves:** accuracy vs `corrupted_client_fraction ∈ {0,0.2,0.4}` with/without the trust
   filter; report detected/filtered counts.
6. **Heterogeneity fairness:** mixed model pool under `synthetic_distillation` vs `logit_distillation`;
   report per-model-type accuracy, fairness gap, comm-by-architecture.
7. **Pareto:** accuracy vs communication/runtime across the above, for the efficiency claims.
8. Track everything to **W&B** (`group` per experiment family, `tags` per axis, seed in the name) for
   the comparison dashboard.

---

## 8. Commit trail (journal branch, on `main`)

`config-schema → baseline-snapshot → metrics-foundation → async-scheduler-scaffold → async-scheduler →
async-consolidation → reliability-scorer → reliability-generator → boundary-adversarial →
boundary-diagnostics → robustness → model-hetero-factory → model-hetero-distill → docs →
experiment-tooling → validation-hardening → wandb-tracker → wandb-metrics → wandb-docs`.

Each commit is `journal-sim/<phase>: …`, self-contained, smoke-tested, with the conference default
kept byte-identical. Not pushed — run `git push origin main` when ready.
