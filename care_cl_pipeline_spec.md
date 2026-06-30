# Pipeline Spec: Domain-Incremental Continual Learning for Wind-Turbine Anomaly Detection (CARE to Compare)

> **How to use this file.** Feed this entire document to a code-generating LLM (e.g. Claude Code, or paste into a chat model) as the project brief. It is written as an unambiguous build spec: module layout, function signatures, data-handling decisions, algorithms, and acceptance criteria. Generate the code module-by-module in the order given. Do **not** invent dataset internals — match the loader contract below and fail loudly if the data does not conform.

---

## 0. Project goal (one paragraph)

Build a reproducible **domain-incremental continual-learning (CL) benchmark** on the CARE to Compare wind-turbine SCADA dataset. Train an autoencoder-based **normal-behavior model (NBM)** sequentially across the three wind farms (Farm A → Farm B → Farm C, each treated as one *domain*). Measure **catastrophic forgetting** of earlier farms using the **CARE score**, and show that a lightweight **replay + knowledge-distillation** strategy — with a **contextual-bandit-tuned anomaly threshold** — mitigates forgetting relative to naive sequential fine-tuning. Deliver: runnable code, logged metrics, figures, and a results table.

---

## 1. Hard constraints

- **Compute:** single consumer GPU, ~10–12 GB VRAM (RTX 2080 Ti / 3080). Models must fit and train in minutes-to-hours, not days. Default to small AEs. CPU fallback must work.
- **Timeline:** code must be generatable and runnable within a few days. Prefer simple, well-tested components over clever ones.
- **Reproducibility:** global seed control; every experiment writes a JSON/CSV results record; one command reruns everything.
- **Language/stack:** Python 3.10–3.12. Use **PyTorch** (preferred) *or* TensorFlow ≥2.15 if reusing Fraunhofer code — pick one and be consistent. Default to PyTorch unless reusing `EnergyFaultDetector`.

---

## 2. Dataset contract (CARE to Compare)

**Source:** Zenodo record `10958775`; paper MDPI *Data* 9(12):138 (2024); official baseline + CARE-score code at `github.com/AEFDI/EnergyFaultDetector`.

**Structure the code must assume (verify against the actual download on first run; fail loudly on mismatch):**

- 3 wind farms = 3 domains, with **different feature counts**: Farm A ≈ 86 features (5 turbines, Portugal onshore, EDP-derived), Farm B ≈ 257 features (German offshore), Farm C ≈ 957 features (German offshore). Treat these counts as *discovered at load time*, not hard-coded.
- 95 sub-datasets total; 44 contain a labeled anomaly event, 51 are normal-only.
- 10-minute resolution SCADA. Each sub-dataset has a train split (normal) and a test/eval split that may contain an anomaly event window.
- Per-row **status-ID** columns mark operating modes; rows flagged as non-normal operation should be filterable.
- Anomaly labels are **event windows** (start/end), not per-row point labels — the CARE score is event-aware (see §6).
- License CC-BY-SA 4.0. Data is anonymized; timestamps may be shifted/anonymized — **do not rely on absolute calendar time**; use only within-series ordering and relative time.

**Loader contract — implement `data/loader.py`:**

```python
@dataclass
class FarmData:
    farm_id: str                      # "A" | "B" | "C"
    feature_names: list[str]          # discovered, length F_farm
    train: list[Subdataset]           # normal-behavior training subdatasets
    eval:  list[Subdataset]           # subdatasets for scoring (some with anomaly events)

@dataclass
class Subdataset:
    id: str
    df: pd.DataFrame                  # time-ordered, columns = feature_names (+ status, +label cols)
    status_col: str | None
    anomaly_events: list[tuple[int,int]]   # (start_idx, end_idx) inclusive; [] if normal-only
    is_normal_only: bool

def load_care(root: str) -> dict[str, FarmData]:
    """Return {"A": FarmData, "B": FarmData, "C": FarmData}. Raise if structure unexpected."""
```

---

## 3. The central technical decision — cross-farm feature alignment

The three farms have **different feature counts and (after anonymization) only partially overlapping signal names**. The model is trained on one farm at a time but must be *evaluable on all previously-seen farms*. Resolve this with a **shared encoder + per-farm input/output adapters** design:

- **Shared signals:** identify the set of physically-meaningful signals present (by name or by documented mapping) in *all three* farms — at minimum power output, wind speed, and (where present) reactive power, rotor speed, ambient/nacelle temperature. Build a canonical `SHARED_SIGNALS` list. The **shared latent encoder** consumes only the standardized shared signals so a single latent space is comparable across farms.
- **Per-farm adapters:** each farm additionally gets a small linear "input adapter" mapping its full feature vector → a fixed-width farm embedding, and a matching "output head" for reconstruction. The shared encoder/decoder core is what continual learning protects; adapters are farm-specific and may be stored per farm.
- Implement two modes behind a flag so this decision is ablatable:
  - `align_mode="shared_only"` — encode only `SHARED_SIGNALS` (simplest; guaranteed comparable; **default**).
  - `align_mode="adapter"` — shared core + per-farm adapters (richer; optional if time permits).
- **Standardization:** fit per-farm `StandardScaler` on that farm's *normal training* data only; persist scalers. Never fit on eval data.

> If `SHARED_SIGNALS` cannot be reliably identified from the real data, **stop and report** the available column names per farm rather than guessing — this choice drives every downstream comparison.

---

## 4. Model — autoencoder normal-behavior model

Implement `models/ae_nbm.py`.

- Input: a window of standardized signals. Default **window = single timestep** of shared signals (simplest, matches CARE baseline spirit); provide an optional `window_len` > 1 producing flattened or 1D-conv windows.
- Architecture: small fully-connected AE. Encoder dims e.g. `[F_in → 64 → 32 → 16]`, symmetric decoder. ReLU, no batchnorm needed at this size. Latent = 16 (configurable).
- Reconstruction loss: MSE.
- **Anomaly score** for a row/window = reconstruction error (MSE per sample), optionally Mahalanobis-normalized per feature. Higher = more anomalous.
- Keep total params small enough to train on CPU in minutes; GPU optional.
- Provide `transformer_nbm` as an **optional** alternative (patch/temporal transformer reconstructor) behind a flag for an ablation only — not required for the core result.

---

## 5. Continual-learning protocol

Implement `cl/protocol.py`. The domain sequence is **A → B → C**.

After training on each farm, evaluate the *current* model on **all farms seen so far** and log the CARE score per farm. This yields a stage × farm matrix used for forgetting/transfer metrics.

Implement these strategies behind a common interface `Strategy.train_on(farm)` / shared model state:

1. **`naive`** — sequential fine-tuning, no protection. *Lower bound (expected to forget).*
2. **`joint`** — train once on all farms pooled. *Upper bound (no forgetting; not a real CL method, reference only).*
3. **`ewc`** — Elastic Weight Consolidation: after each farm, estimate Fisher information on that farm's normal data; add quadratic penalty pulling shared-core weights toward previous values. One hyperparameter `lambda_ewc`.
4. **`replay`** — Experience Replay: keep a small ring buffer of normal windows per past farm (e.g. `buffer_per_farm = 2000`); interleave buffer samples into each new farm's training batches.
5. **`distill`** — LwF-style knowledge distillation: before training on a new farm, snapshot the previous model as a frozen teacher; add a penalty so the student's reconstructions of *new-farm inputs* stay close to the teacher's (preserves prior normal-behavior manifold). One hyperparameter `lambda_distill`.
6. **`replay_distill`** (**the proposed method**) — combine `replay` + `distill` (DER++-style: replay buffer stores inputs *and* teacher reconstruction targets). This is the configuration the paper argues for.

Common interface:

```python
class Strategy(Protocol):
    name: str
    def train_on(self, farm: FarmData, model, optim, cfg) -> None: ...
    def end_of_farm(self, farm: FarmData, model) -> None: ...  # update Fisher/buffer/teacher
```

Reuse the **Avalanche** library for EWC/replay scaffolding if it accelerates implementation; otherwise implement from scratch (these are short).

---

## 6. Evaluation — CARE score (event-aware)

Implement `eval/care_score.py`. **Match the official CARE-score definition** from the MDPI 2024 paper / `EnergyFaultDetector`. Do not approximate with plain ROC-AUC.

- The score rewards detecting the labeled anomaly **event** within its window, penalizes false alarms on normal-only datasets, and rewards **earliness** (detecting before the event end / before the maintenance action).
- Components to compute and report separately: **Coverage**, **Accuracy**, **Reliability**, **Earliness** (verify exact names/weighting against MDPI *Data* 2024 §4.1.1 / Fig. 2 before finalizing — see Caveats).
- Reference anchors for sanity-checking your implementation: random strategy ≈ **0.5**, all-normal / all-anomaly ≈ **0**, the paper's AE baseline ≈ **0.66**, isolation forest *below* the 0.5 floor.
- **Acceptance gate:** before any CL experiment is trusted, reproduce a single-farm AE CARE score in the **~0.6–0.7** range. If you cannot, fix the harness first — a wrong baseline invalidates all CL comparisons.

**Derived CL metrics — implement `eval/cl_metrics.py`:**

- `forgetting(A)` = (CARE on Farm A right after training A) − (CARE on Farm A after training C). Positive = forgetting.
- `backward_transfer` (BWT) = mean change in past-farm scores after learning new farms.
- `forward_transfer` (FWT) optional.
- `final_avg_care` = mean CARE across all farms at the end of the sequence.

---

## 7. Lightweight RL — contextual bandit threshold tuner

Implement `rl/bandit_threshold.py`. **Supporting role only — keep it tiny (~50–100 lines).**

- Problem: per farm (or per operating regime via status-ID), choose the **anomaly-score threshold** that maximizes a CARE-aligned reward.
- Formulation: **contextual bandit**. Context = recent reconstruction-error statistics (e.g. mean/quantiles over a rolling window, optionally the status-ID). Arms = a discrete set of candidate thresholds (e.g. quantiles 0.90…0.999 of training error). Reward = detection benefit − false-alarm penalty, aligned with CARE components.
- Algorithm: LinUCB *or* epsilon-greedy contextual bandit; both are fine. Must train in seconds.
- Provide a `--bandit {on,off}` flag so its contribution is a clean ablation (fixed-quantile threshold = `off` baseline).
- Optional second use behind a flag: bandit/multi-armed selection of the **replay/distillation weight** (`lambda`) — frame as bandit HPO. Keep optional.

---

## 8. Repository layout to generate

```
care_cl/
  README.md                  # how to download data, install, run
  requirements.txt
  config/
    default.yaml             # all hyperparams, seeds, paths, align_mode, strategy list
  data/
    loader.py                # §2 contract
    align.py                 # §3 SHARED_SIGNALS, scalers, adapters
  models/
    ae_nbm.py                # §4
    transformer_nbm.py       # optional ablation
  cl/
    protocol.py              # §5 sequence + stage×farm eval loop
    strategies.py            # naive, joint, ewc, replay, distill, replay_distill
  rl/
    bandit_threshold.py      # §7
  eval/
    care_score.py            # §6 (match official definition)
    cl_metrics.py            # forgetting, BWT, FWT, final_avg_care
  experiments/
    run.py                   # CLI entrypoint: runs one (strategy, seed) config
    sweep.py                 # runs full grid, aggregates to results/
    plots.py                 # forgetting curves, per-farm CARE bars, ablations
  results/                   # JSON per run + aggregated CSV + figures (gitignored except samples)
  tests/
    test_care_score.py       # asserts random≈0.5, all-normal≈0, monotonicity
    test_loader.py           # asserts 3 farms, differing feature counts, events parsed
    test_cl_metrics.py
```

---

## 9. Experiment grid (what `sweep.py` runs)

- **Strategies:** `naive`, `joint`, `ewc`, `replay`, `distill`, `replay_distill`.
- **Bandit:** `{off, on}` (at least for `naive` and `replay_distill`).
- **Seeds:** ≥3 (report mean ± std).
- **Sequence:** A→B→C (optionally also a second order, e.g. C→B→A, to show order-robustness if time permits).
- **Ablations:** buffer size ∈ {500, 2000, 8000}; `lambda_distill` ∈ {0.1, 1.0, 10}; `align_mode` ∈ {shared_only, adapter}.
- Output: one row per (strategy, bandit, seed, stage, farm) with all four CARE sub-scores + reconstruction loss; plus a derived metrics table (forgetting, BWT, final_avg_care).

---

## 10. Figures `plots.py` must produce

1. **Forgetting curves:** x = training stage (after A / after B / after C), y = CARE on Farm A (and B), one line per strategy. Shows `naive` degrading, `replay_distill` staying flat.
2. **Final per-farm CARE bar chart:** grouped bars per strategy.
3. **Ablation panels:** buffer size vs final_avg_care; bandit on/off delta.
4. **Threshold-bandit trace** (optional): chosen threshold over time vs fixed quantile.

---

## 11. Acceptance criteria (definition of done)

1. `python experiments/run.py --strategy naive --seed 0` runs end-to-end and writes a results JSON.
2. Single-farm AE reproduces a CARE score in **~0.6–0.7** (the §6 gate).
3. `tests/test_care_score.py` passes: random≈0.5, all-normal≈0, score increases when a true event is correctly + earlier detected.
4. `naive` shows **positive forgetting** on Farm A; `replay_distill` shows **lower forgetting** than `naive` (the core claimed result). If this does not hold, report it honestly — a negative/partial result is still a valid benchmark finding; do not fabricate.
5. `joint` ≥ all CL strategies on final_avg_care (upper-bound sanity check).
6. `sweep.py` reproduces the full results table + all figures from one command.
7. Seeds are fixed; rerun reproduces numbers within seed variance.
8. README documents data download, install, and the single reproduce command.

---

## 12. Build order for the generating LLM

1. `data/loader.py` + `tests/test_loader.py` — confirm the dataset parses (print per-farm column names; **stop and surface them if `SHARED_SIGNALS` is unclear**).
2. `data/align.py` (`shared_only` first).
3. `models/ae_nbm.py`.
4. `eval/care_score.py` + `tests/test_care_score.py` — **pass the §6 acceptance gate before going further.**
5. `cl/protocol.py` + `cl/strategies.py` (`naive`, `joint` first, then `ewc`, `replay`, `distill`, `replay_distill`).
6. `eval/cl_metrics.py`.
7. `rl/bandit_threshold.py`.
8. `experiments/run.py` → `sweep.py` → `plots.py`.
9. Fill `README.md`, `requirements.txt`, `config/default.yaml`.

---

## 13. Caveats the generating LLM must respect

- **Match the official CARE-score definition** (MDPI *Data* 2024 / `EnergyFaultDetector`). Verify exact sub-score names and weighting against the paper before finalizing `care_score.py`; the {Coverage, Accuracy, Reliability, Earliness} naming/weighting above is to be confirmed, not assumed.
- **Do not hard-code feature counts (86/257/957)** — discover them at load time and assert they differ.
- **Never fit scalers or thresholds on eval data.** Fit on normal training data only.
- **Timestamps are anonymized** — use only relative ordering, never absolute calendar dates.
- **Report negative/partial results honestly.** The contribution is a *benchmark + empirical study*; a finding that a method does **not** help is publishable and must not be hidden or fabricated.
- **Keep RL lightweight** — if the bandit balloons in complexity, cut it to a fixed-quantile threshold; it is a supporting ablation, not the core.
- If reusing `EnergyFaultDetector` (TensorFlow), stay in TF throughout; otherwise build clean in PyTorch. Do not mix frameworks for the model.
