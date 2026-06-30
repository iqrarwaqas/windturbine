# Domain-Incremental Continual Learning for Wind-Turbine Anomaly Detection (CARE to Compare)

A reproducible **domain-incremental continual-learning (CL)** benchmark on the
[CARE to Compare](https://doi.org/10.3390/data9120138) wind-turbine SCADA dataset.
An autoencoder normal-behaviour model (NBM) is trained sequentially across three
wind farms (**A → B → C**, each a *domain*). We measure **catastrophic forgetting**
with the **CARE score** and show whether **replay + knowledge distillation** — with a
**contextual-bandit-tuned anomaly threshold** — mitigates forgetting versus naive
fine-tuning.

> Honest-results policy (§13): this is a benchmark + empirical study. If a method
> does **not** help, the code reports it as-is. Numbers are not massaged.

## 1. Data download

1. Download the CARE-to-Compare dataset from Zenodo record
   [`10958775`](https://doi.org/10.5281/zenodo.10958775).
2. Extract so the layout is:

   ```
   <root>/Wind Farm A/datasets/*.csv
                      /event_info.csv
                      /feature_description.csv
   <root>/Wind Farm B/...
   <root>/Wind Farm C/...
   ```
3. Point `data.root` in `config/default.yaml` at `<root>` (default: `D:/Datasets/Care`),
   or set the `CARE_ROOT` env var for the tests.

CSV files are `;`-separated. Three farms have **different feature counts**
(81 / 252 / 952 usable sensor channels) — counts are discovered at load time and
asserted to differ; nothing is hard-coded.

## 2. Install

```bash
pip install -r requirements.txt
```

Python 3.10–3.12, PyTorch. CPU works (the shared-signal AE is tiny); GPU optional
via `train.device: cuda`.

## 3. Reproduce everything (one command)

```bash
python -m care_cl.experiments.sweep            # full grid -> results/ + figures
python -m care_cl.experiments.sweep --quick    # fast smaller pass
```

This writes `results/records.csv`, `results/metrics.csv`,
`results/metrics_aggregated.csv` and the figures in §10.

### Single runs

```bash
# §6 acceptance gate: single-farm AE CARE should land ~0.6-0.7
python -m care_cl.experiments.run --gate --farm A

# one CL config
python -m care_cl.experiments.run --strategy naive --seed 0
python -m care_cl.experiments.run --strategy replay_distill --bandit on --seed 1
```

## 4. How it works

| Concern | Where | Notes |
|---|---|---|
| Dataset contract | `data/loader.py` | parses farms/events, fails loudly on mismatch |
| Cross-farm alignment (§3) | `data/align.py` | 5 canonical `SHARED_SIGNALS` mapped per farm; per-farm `StandardScaler` fit on **normal training data only** |
| AE NBM (§4) | `models/ae_nbm.py` | shared encoder/decoder core + optional per-farm adapters; score = per-sample recon MSE |
| CARE score (§6) | `eval/care_score.py` | Coverage (F₀.₅), Accuracy, Reliability (criticality counter, t_c=72), Earliness; aggregation per Eq. 4–5 |
| CL strategies (§5) | `cl/strategies.py` | `naive`, `joint`, `ewc`, `replay`, `distill`, `replay_distill` (DER++) |
| Protocol (§5) | `cl/protocol.py` | A→B→C; stage×farm CARE matrix; compact disk-cached farm tensors |
| CL metrics (§6) | `eval/cl_metrics.py` | forgetting, BWT, FWT, final_avg_care |
| Bandit threshold (§7) | `rl/bandit_threshold.py` | epsilon-greedy / LinUCB; CARE-aligned reward computed on **training** error only |

### Shared signals (§3)

Encoder consumes 5 physically-meaningful signals present in all three farms,
matched by `feature_description.csv`:

`active_power`, `wind_speed`, `reactive_power`, `rotor_speed`, `ambient_temp`

The exact per-farm column mapping lives in `data/align.py:FARM_SIGNAL_MAP` and is
verified against the real data at load time. `align_mode: adapter` additionally
trains per-farm input/output adapters over the full feature vector (optional, heavier).

## 5. CARE score definition

Matches Gück, Roelofs & Faulstich, *Data* 2024, 9(12):138
([arXiv:2404.10320](https://arxiv.org/abs/2404.10320)) and the official
[AEFDI/EnergyFaultDetector](https://github.com/AEFDI/EnergyFaultDetector):

- **Coverage** — per-anomaly-dataset F_β (β=0.5) on normal-status points.
- **Accuracy** — per-normal-dataset `tn/(fp+tn)` on normal-status points.
- **Reliability** — event-level F_β across all datasets; each dataset gets one
  anomaly/normal verdict via a criticality counter (threshold 72 = 12 h).
- **Earliness** — weighted detection score, weight 1 over the first half of the
  event window, linearly → 0 over the second half.
- **Aggregate** — `WA = (Cov + Earl + Rel + 2·Acc)/5`;
  `CARE = 0` if nothing detected, `= Acc` if `Acc < 0.5`, else `WA`.

Sanity anchors (tested in `tests/test_care_score.py`): random ≈ 0.5,
all-normal/all-anomaly ≈ 0, monotone in correct & earlier detection.

> The shared-signal AE uses only 5 channels (vs the paper's full feature sets), so
> the single-farm gate lands ~0.60–0.62 rather than the full-feature ~0.66; this is
> expected and still within the §6 gate band.

## 6. Tests

```bash
python -m pytest care_cl/tests -q
```

`test_loader.py` auto-skips if the dataset is absent. `test_care_score.py` is the
§6 harness gate; `test_cl_metrics.py` checks forgetting/BWT.

## 7. Repository layout

See the module table above; figures and CSVs land in `results/` (gitignored except
samples). Disk caches of standardized farm tensors live in `results/cache/`.
