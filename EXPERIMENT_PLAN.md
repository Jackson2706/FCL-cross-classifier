# EXPERIMENT_PLAN — Journal Experiment Sweep (FEDRUA)

The concrete experiments to run for the journal submission. **Do NOT auto-run** — these are
multi-GPU-hour jobs; launch them deliberately on a GPU machine (`device: cuda`). All commands
use conda env **FCL**. See `EXPERIMENTS.md` for how config keys are set (JSON `--cfp` today; a
YAML launcher is pending).

Conventions:
- Run each configuration over **seeds** `{0, 1, 2}` (`seed` key) for mean ± std.
- Use `--offlog True --log True` and a distinct `--note` per run so output dirs don't collide.
- Datasets: CIFAR-10, CIFAR-100 (`hparams/cifar100/Ours_v2_cifar100.json`), ImageNet-1K,
  Camelyon17 (`hparams/wilds/Ours_v2_wilds.json`). Examples below use CIFAR-100.
- Tooling available: `--cfp` accepts YAML directly, `scripts/aggregate_seeds.py` (multi-seed
  aggregation), and `scripts/plots.py` (figures). See `EXPERIMENTS.md §5`.
- W&B tracking is optional/off by default. To track any run below, add a `wandb:` block to its
  config (`enabled: true`, `mode: online`/`offline`, `privacy_safe_mode: true`) — see `EXPERIMENTS.md §6`.
  Use `group`/`tags` to compare baseline vs journal variants, sync vs async, and seeds on one dashboard.

---

## 1. Conference baseline (reference numbers)
```bash
for s in 0 1 2; do
  jq ".seed=$s" hparams/cifar100/Ours_v2_cifar100.json \
  | conda run --no-capture-output -n FCL python system/main.py \
      --cfp /dev/stdin --offlog True --log True --note "baseline_seed$s"
done
```

## 2. Paper method (FEDRUA full)
Config `configs/fedrua_paper.yaml` (reliability_mode=multi_signal, generator_distillation=true,
boundary_mode=fgsm, 20 clients / 200 samples). Runs directly via the YAML launcher; vary the seed
by copying the YAML (it is config-authoritative):
```bash
for s in 0 1 2; do
  sed "s/^seed:.*/seed: $s/; s/^note:.*/note: fedrua_paper_seed$s/" \
      configs/fedrua_paper.yaml > /tmp/fedrua_seed$s.yaml
  conda run -n FCL python system/main.py --cfp /tmp/fedrua_seed$s.yaml --offlog True --log True
done
```

## 3. True asynchronous HFCL (Extension 1)
Sweep the scheduler modes; the paper's async setting ≈ `fixed_groups` with 2 groups.
```bash
for mode in synchronous fixed_groups uniform_speed poisson_clock; do
 for s in 0 1 2; do
  jq ".seed=$s | .async_mode=($( [ $mode = synchronous ] && echo false || echo true )) \
      | .client_task_speed_distribution=\"$mode\" | .num_speed_groups=2 | .max_task_lag=2" \
      hparams/cifar100/Ours_v2_cifar100.json \
  | conda run --no-capture-output -n FCL python system/main.py \
      --cfp /dev/stdin --offlog True --log True --note "async_${mode}_seed$s"
 done
done
```
Also sweep dropout / partial participation: `client_dropout_rate ∈ {0, 0.1, 0.3}`,
`partial_participation_rate ∈ {1.0, 0.5}`. Read async metrics from `metrics_summary.json → async`.

## 4. Reliability ablations (Extension 2)
```bash
for rm in none entropy mutual_information bn_realism trust_only multi_signal calibrated; do
 for s in 0 1 2; do
  jq ".seed=$s | .reliability_mode=\"$rm\" | .generator_distillation=true | .adv=0.0" \
      hparams/cifar100/Ours_v2_cifar100.json \
  | conda run --no-capture-output -n FCL python system/main.py \
      --cfp /dev/stdin --offlog True --log True --note "rel_${rm}_seed$s"
 done
done
```
Compare accuracy/forgetting and ECE (`reliability_log.json`, `metrics_summary.json → calibration`).

## 5. Boundary diagnostics & modes (Extension 3)
```bash
for bm in none fgsm pgd_light margin_regularization prototype_margin; do
 for s in 0 1 2; do
  jq ".seed=$s | .boundary_mode=\"$bm\" | .reliability_mode=\"multi_signal\" \
      | .generator_distillation=true | .adv=0.0" hparams/cifar100/Ours_v2_cifar100.json \
  | conda run --no-capture-output -n FCL python system/main.py \
      --cfp /dev/stdin --offlog True --log True --note "boundary_${bm}_seed$s"
 done
done
```
Collect margin distributions, prototype drift, FGSM/PGD robust accuracy from `boundary_log.json`
and `metrics_summary.json → robustness/boundary`.

## 6. Robustness scenarios (Extension 5)
```bash
for cf in 0.0 0.2 0.4; do
 for s in 0 1 2; do
  jq ".seed=$s | .robustness_mode=\"corrupted_uploads\" | .corrupted_client_fraction=$cf \
      | .label_noise_rate=0.2 | .malicious_confidence_attack=true \
      | .reliability_mode=\"multi_signal\" | .use_filter=true" \
      hparams/cifar100/Ours_v2_cifar100.json \
  | conda run --no-capture-output -n FCL python system/main.py \
      --cfp /dev/stdin --offlog True --log True --note "robust_cf${cf}_seed$s"
 done
done
```
Also run `privacy_proxies=true` on the paper config to emit `privacy_log.json`.

## 7. Model heterogeneity (Extension 4)
```bash
for agg in synthetic_distillation logit_distillation; do
 for s in 0 1 2; do
  jq ".seed=$s | .model_heterogeneity=true \
      | .client_model_pool=[\"small_cnn\",\"resnet18\",\"mobilenetv2\",\"lightweight_resnet\"] \
      | .server_model=\"resnet18\" | .aggregation_mode=\"$agg\"" \
      hparams/cifar100/Ours_v2_cifar100.json \
  | conda run --no-capture-output -n FCL python system/main.py \
      --cfp /dev/stdin --offlog True --log True --note "hetero_${agg}_seed$s"
 done
done
```
Report per-model-type accuracy, fairness gap, comm-by-architecture, server distillation cost.

## 8. Multi-seed aggregation
```bash
conda run -n FCL python scripts/aggregate_seeds.py \
  --runs "out/CIFAR100_Ours_v2_*_seed*" --group-by note --out out/aggregated
```

## 9. Plotting
```bash
conda run -n FCL python scripts/plots.py --run-dir out/aggregated --out-dir figures/ --which all
```

---

## Notes
- Full CIFAR-100 FEDRUA ≈ 5 tasks × 50 rounds × 5 local epochs + 2000 generator steps/task; budget
  GPU time accordingly. ImageNet-1K and multi-seed sweeps are the largest cost — stage them.
- Every run emits machine-readable JSON/CSV (see `EXPERIMENTS.md §2`); `scripts/aggregate_seeds.py`
  and `scripts/plots.py` consume these directly.
- Keep `seed` fixed within a comparison group; vary only the ablated axis.
