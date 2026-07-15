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
- Journal keys are set in the config file passed to `--cfp`, which accepts **either JSON or YAML**
  (`.yaml`/`.yml`). YAML configs are **config-authoritative** (a CLI flag overrides a YAML key only
  when explicitly passed), so `configs/fedrua_paper.yaml` and `configs/legacy_code.yaml` run
  directly. JSON keeps its legacy behavior (the mirrored CLI flags always override). Journal keys
  are *not* individual CLI flags.
- `configs/schema.yaml` is the authoritative list of all journal keys and their defaults.

### Running a YAML config directly

```bash
conda run -n FCL python system/main.py --cfp configs/fedrua_paper.yaml --offlog True --log True
```

### Enabling journal features from a JSON config

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

## 5. Multi-seed aggregation & plotting

**Multi-seed aggregation** (`scripts/aggregate_seeds.py`, pure stdlib) reads `metrics_summary.json`
(+ `resolved_config.json`) from many run dirs, groups runs that differ only by seed, and writes
mean/std/n tables:

```bash
conda run -n FCL python scripts/aggregate_seeds.py \
  --runs "out/CIFAR100_Ours_v2_*seed*" --group-by note --out out/aggregated
# -> out/aggregated.json (per-group per-metric mean/std/n/values) and out/aggregated.csv
```

**Plotting** (`scripts/plots.py`, matplotlib) reads a run dir's machine-readable logs and writes
PNGs; each plot is skipped cleanly if its source file is absent:

```bash
conda run -n FCL python scripts/plots.py \
  --run-dir out/CIFAR100_Ours_v2_ResNet18_adam_lr0.05_fedrua_paper \
  --out-dir figures --which all
# plots: accuracy, forgetting, reliability, calibration, margin, async, heterogeneity, pareto
# (pareto reads an aggregated.json; pass --run-dir out/aggregated.json for it)
```

If matplotlib is ever missing: `conda run -n FCL pip install matplotlib`.
All metrics are also emitted as JSON/CSV per run (Section 2), so custom analysis is easy.

---

## 6. Weights & Biases (W&B) experiment tracking

W&B is **optional, config-driven, and safe by default** (disabled). Local CSV/JSON logging is
always produced regardless of W&B. A single `WandbTracker` (`system/flcore/tracking/`) owns all
W&B calls; a logging failure never crashes training.

### Enabling (YAML config-authoritative)

```yaml
wandb:
  enabled: true
  mode: online            # online | offline | disabled
  project: FEDRUA-Journal
  entity: null            # your W&B entity (null = default)
  group: null
  job_type: train
  name: null              # null -> auto-generated descriptive name
  tags: []
  privacy_safe_mode: true # aggregates/scalars only; NO raw client data/images/samples
  log_model: false
  log_generated_samples: false
  resume: allow
```

- **Legacy compatibility:** the old boolean `--wandb True` / JSON `"wandb": true` still works and
  maps to an online run; default (`False`/absent) = disabled. `enabled: false` forces disabled
  regardless of `mode`.
- **Modes:** `disabled` = pure no-op (no import, no network). `offline` = local only, **no login
  required**. `online` = cloud sync; on credential/network failure it **auto-falls back** to
  offline then disabled with one warning (training continues).
- Each run writes `<run_dir>/wandb_run.json` (`run_id`, `mode`, `project`, `entity`, `name`, `url`).

### Commands

```bash
# Disabled (default) — nothing sent anywhere, local CSV/JSON only:
conda run -n FCL python system/main.py --cfp configs/legacy_code.yaml --offlog True --log True

# Offline (no login) — set WANDB_DIR to keep the local run out of the repo:
WANDB_DIR=/tmp/journal-wandb conda run -n FCL python system/main.py \
  --cfp configs/fedrua_paper.yaml --offlog True --log True   # fedrua_paper.yaml ships wandb.mode=offline

# Online — requires `wandb login` once; then set wandb.mode: online in the config.

# Sync an offline run to the cloud later:
wandb sync /tmp/journal-wandb/wandb/offline-run-*
```

### What gets logged (namespaced)

`eval/*` (avg_accuracy, forgetting, bwt, aaa, task_accuracy/task_<id>), `generator/loss_{total,cls,kd,bn}`,
`classifier/loss_{replay,adv}`, `reliability/{mean,std,min,max,accepted_ratio}`,
`boundary/{margin_mean,margin_std,robust_acc_fgsm,robust_acc_pgd_light}`, `communication/*_gb`,
`runtime/*_seconds`, `gpu/*` (when CUDA present), `async/*`, `robustness/*`, `heterogeneity/*`.
A section only appears when its module is active. Use the `group`/`tags`/`job_type` fields to compare
baseline vs journal variants, sync vs async, reliability/boundary modes, robustness, heterogeneity,
and seeds on one dashboard.

### Privacy

With `privacy_safe_mode: true` (default), only scalar aggregates are logged — **never** raw client
data, medical images, generated samples, logits, gradients, schedules, or per-client accuracy keys.
The run config is sanitized (paths/device-ids/host/secrets stripped) before upload. Do not set
`log_generated_samples`/`log_model` true for sensitive datasets.

### Inspecting an offline run locally

```bash
python - <<'PY'
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
ds = datastore.DataStore(); ds.open_for_scan("<offline-run>/run-XXXX.wandb")
keys = set()
while (rec := ds.scan_data()) is not None:
    r = pb.Record(); r.ParseFromString(rec)
    if r.WhichOneof("record_type") == "history":
        keys |= {i.key for i in r.history.item}
print(sorted(k for k in keys if not k.startswith("_")))
PY
```
