# EXPERIMENTS — FEDRUA Journal Simulation Framework

This document explains how to run the conference baseline and every journal-simulation
mode added on top of it. All new capability is **config-gated and defaults to off**, so the
default configuration reproduces the conference `Ours_v2` (FEDRUA) behavior.

> Environment: every command runs inside conda env **FCL**
> (`conda run -n FCL python ...` or `conda activate FCL`).
> This machine is CPU-only; full-scale runs need a GPU (`device: cuda`). The commands below
> use bounded overrides for CPU smoke tests — remove them for full runs.

---

## 1. How configuration works (important)

- The entrypoint is `system/main.py`, invoked with `--cfp <config>` plus a few CLI flags.
- `system/main.py` loads the config into an `argparse.Namespace`; **every journal key is read
  via `getattr(args, key, default)`.** Keys absent from the config fall back to their safe
  (off) defaults, so old configs keep the conference code path.
- **Today, journal keys are set inside the JSON config passed to `--cfp`** (they are *not*
  individual CLI flags). The canonical YAML configs `configs/fedrua_paper.yaml` and
  `configs/legacy_code.yaml` document every key; a small **YAML launcher that lets `--cfp`
  load `.yaml` directly is deferred (Phase 7, pending)**. Until then, use the JSON method below.
- `configs/schema.yaml` is the authoritative list of all journal keys and their defaults.

### Enabling journal features today (JSON method)

Copy the base config and add the keys you need:

```bash
cd /home/jackson/Desktop/FCL_Project/FCL-cross-classifier
# start from the flagship CIFAR-100 config
cp hparams/cifar100/Ours_v2_cifar100.json /tmp/my_run.json
# edit /tmp/my_run.json to add journal keys, e.g. "reliability_mode": "multi_signal"
conda run -n FCL python system/main.py --cfp /tmp/my_run.json --offlog True --log True --note my_run
```

Or inject keys inline with `jq` (the pattern the smoke tests use):

```bash
jq '.reliability_mode="multi_signal" | .generator_distillation=true' \
   hparams/cifar100/Ours_v2_cifar100.json \
| conda run --no-capture-output -n FCL python system/main.py \
   --cfp /dev/stdin --offlog True --log True --note reliability_multi
```

### Base command (conference baseline)

```bash
conda run -n FCL python system/main.py \
  --cfp hparams/cifar100/Ours_v2_cifar100.json \
  --offlog True --log True --note baseline
```

With all journal keys at defaults this is the **unmodified conference FEDRUA** run. (Every
journal phase verified this default path is byte-identical to the pre-journal code.)

---

## 2. Output layout (machine-readable)

With `--offlog True`, results are written under
`out/<dataset>_<algorithm>_<model_str>_<optimizer>_lr<lr>_<note>/`:

| File | Produced by | Contents |
|---|---|---|
| `Global/metrics.csv`, `Global/<algo>_accuracy_matrix.csv` | base | per-round loss/acc; task×task accuracy matrix |
| `Client_Global/…`, `Client_Local/…` | base | per-client accuracy CSVs |
| `metrics_summary.json` | Phase 1+ | avg acc, forgetting, **BWT, AAA**, communication MB, server-compute seconds; plus `async`, `robustness`, `boundary`, `heterogeneity`, `calibration` sub-sections when active |
| `resolved_config.json` | Phase 1+ | the fully-resolved config (reproducibility record) |
| `reliability_log.json` | Phase 3 | per-consolidation weight histogram, by-class/by-client means, accept/reject, weight-vs-correctness, ECE |
| `boundary_log.json` | Phase 4 | margin distributions, boundary-sample ratio, prototype drift, class-center matrix summary |
| `consolidation_log.json` | Phase 2c | async watermark consolidation events (one per boundary) |
| `async_schedule.json` | Phase 2b | the materialized per-client task/participation schedule (replayable) |
| `privacy_log.json` | Phase 5 | NN-distance distribution, memorization score, MI proxy (heuristic only) |

---

## 3. Running each mode

Add the listed keys to your JSON config (or the YAML config once the launcher lands).

### 3.1 Conference baseline
Base command above. Reference config: `configs/legacy_code.yaml` (mirrors the current JSON,
all journal flags off).

### 3.2 Paper method (FEDRUA as described in the paper)
Reference config: `configs/fedrua_paper.yaml`. Key settings:
```
reliability_mode: multi_signal      # uncertainty w = exp(-b1*H)·exp(-b2*d_BN)·trust
generator_distillation: true        # L_G = oh*L_cls + kd_weight*L_kd + bn*L_bn + adv*L_adv
adv: 0.0                            # BN acts as the surrogate critic (no WGAN critic)
boundary_mode: fgsm                 # density-gated FGSM adversarial synthetic replay
num_clients: 20                     # paper scale (current JSON uses 10)
generated_samples_per_class: 200    # paper scale (current JSON uses 100)
```
> Scale note: `fedrua_paper.yaml` uses the paper's **20 clients / 200 samples-per-class**.
> Reduce to 10 / 100 for cheaper runs (documented in the file header).

### 3.3 True asynchronous HFCL (Extension 1)
```
async_mode: true
client_task_speed_distribution: fixed_groups   # or uniform_speed | poisson_clock | custom_schedule
num_speed_groups: 2                             # fixed_groups
max_task_lag: 0                                 # 0 = task barrier; >0 = bounded staleness
client_dropout_rate: 0.0                        # per-round temporary absence
permanent_dropout_rate: 0.0                     # permanent client failure
partial_participation_rate: 1.0
task_schedule_seed: 0
consolidation_trigger: watermark                # watermark (default) | quorum
```
Async runs consolidate each task boundary exactly once via a completion watermark; see
`async_schedule.json` and `metrics_summary.json → async` (temporal lag, per-client task id,
per-client-stage continual metrics).

### 3.4 Reliability ablations (Extension 2)
Sweep `reliability_mode` over:
`none, entropy, mutual_information, bn_realism, trust_only, multi_signal, calibrated, oracle_debug`.
Relevant knobs: `reliability_beta`, `reliability_beta_entropy`, `reliability_beta_bn`,
`reliability_trust_floor`, `reliability_accept_threshold`. Calibration: `ece_bins`,
`calibration_temperature`. Inspect `reliability_log.json` (histograms, ECE, weight-vs-correctness).
> `oracle_debug` uses ground-truth labels — sanity-checking only, never a reported result.

### 3.5 Boundary diagnostics (Extension 3)
```
boundary_mode: fgsm            # none | fgsm | pgd_light | margin_regularization | prototype_margin
adv_epsilon: 0.03
pgd_steps: 3                   # pgd_light
lambda_adv: 1.0
density_tau: null              # null = per-batch median d_BN gate
boundary_diagnostics: true     # margins, prototype drift, class-center matrix, boundary ratio
boundary_robust_eval: true     # robust accuracy under FGSM / PGD on the real test set
```
Diagnostics land in `boundary_log.json`; robust accuracy in `metrics_summary.json → robustness/boundary`.

### 3.6 Robustness scenarios (Extension 5)
```
robustness_mode: corrupted_uploads   # composable: label_noise, corrupted_uploads, corrupted_bn,
                                      # malicious_confidence, extreme_imbalance, single_class_expert
corrupted_client_fraction: 0.2
label_noise_rate: 0.0
bn_noise_std: 0.0
malicious_confidence_attack: false
privacy_proxies: true                # heuristic NN-distance / memorization / MI proxy (privacy_log.json)
```
`metrics_summary.json` reports corrupted vs clean client ids and their mean trust (defense visibility).
The legacy `--simulate_bad_clients True` / `--use_filter True` CLI flags still work unchanged.

### 3.7 Model heterogeneity (Extension 4)
```
model_heterogeneity: true
client_model_pool: [small_cnn, resnet18, mobilenetv2, lightweight_resnet]
server_model: resnet18
aggregation_mode: synthetic_distillation   # fedavg (homogeneous only) | logit_distillation | synthetic_distillation
client_distill_steps: 1
distill_transfer_size: 256
```
> A heterogeneous pool under `aggregation_mode: fedavg` is refused (no averaging of incompatible
> architectures) — use `synthetic_distillation` or `logit_distillation`.
`metrics_summary.json → heterogeneity`: per-model-type accuracy, fairness gap, comm-by-architecture,
server distillation cost, capacity gap.

---

## 4. Reproducibility

- Global `seed` (default 0) seeds Python / NumPy / Torch (CPU+CUDA). Runs are bit-reproducible.
  Seeded numbers may differ from historical *unseeded* logs but follow the identical algorithm.
- `resolved_config.json` in each run dir records the exact resolved configuration.
- Async schedules are fully materialized and replayable via `async_schedule.json` + `task_schedule_seed`.

---

## 5. Multi-seed aggregation & plotting (pending — Phase 7 tooling, deferred to Codex)

The following helper scripts are **not yet implemented** (deferred while the Codex CLI is
rate-limited). See `EXPERIMENT_PLAN.md` for the intended commands. Once added:

- `scripts/aggregate_seeds.py` — aggregate `metrics_summary.json` across seeds → mean ± std
  (`aggregated.csv` / `aggregated.json`).
- `scripts/plots.py` — journal figures: accuracy/forgetting over tasks, reliability histograms,
  calibration curves, margin distributions, accuracy-vs-communication/runtime Pareto, async
  temporal-lag, model-heterogeneity fairness.
- YAML launcher: extend `--cfp` to load `configs/*.yaml` directly.

Until then, all metrics are already emitted as JSON/CSV per run (Section 2), so aggregation can
be done manually or with a short script over the `metrics_summary.json` files.
