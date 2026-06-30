# CARE Continual-Learning — Results Summary

- **Align mode:** shared_only  ·  **Bandit:** off  ·  **Seeds:** 3 ([np.int64(0), np.int64(1), np.int64(2)])  ·  **Sequence:** A → B → C
- CARE sub-scores: Coverage (F₀.₅), Earliness, Reliability, Accuracy; higher is better. Forgetting on Farm A = CARE(after A) − CARE(after C); **lower is better**, negative = backward transfer.

## 1. Final average CARE & forgetting (mean ± std over seeds)

| strategy | final_avg_care | forgetting (Farm A) | backward_transfer |  |
| --- | --- | --- | --- | --- |
| joint | 0.576 ± 0.003 | — | 0.000 ± 0.000 | upper bound |
| naive | 0.561 ± 0.025 | 0.053 ± 0.032 | -0.032 ± 0.033 | lower bound |
| ewc | 0.553 ± 0.008 | 0.056 ± 0.026 | -0.041 ± 0.023 |  |
| replay | 0.571 ± 0.012 | 0.047 ± 0.016 | -0.013 ± 0.007 |  |
| distill | 0.555 ± 0.003 | 0.060 ± 0.013 | -0.038 ± 0.012 |  |
| replay_distill | 0.575 ± 0.015 | 0.020 ± 0.024 | -0.006 ± 0.015 |  |

## 2. Final-stage sub-scores (after training C, mean over farms+seeds)

| strategy | coverage | earliness | reliability | accuracy | CARE |
| --- | --- | --- | --- | --- | --- |
| joint | 0.203 | 0.077 | 0.698 | 0.952 | 0.576 |
| naive | 0.181 | 0.070 | 0.658 | 0.949 | 0.561 |
| ewc | 0.152 | 0.059 | 0.654 | 0.951 | 0.553 |
| replay | 0.174 | 0.070 | 0.692 | 0.959 | 0.571 |
| distill | 0.121 | 0.050 | 0.695 | 0.953 | 0.555 |
| replay_distill | 0.183 | 0.070 | 0.714 | 0.955 | 0.575 |

## 3. Farm A CARE over training stages (the forgetting story)

| strategy | after A | after B | after C |
| --- | --- | --- | --- |
| joint | — | — | 0.581 |
| naive | 0.625 | 0.603 | 0.572 |
| ewc | 0.625 | 0.589 | 0.570 |
| replay | 0.625 | 0.610 | 0.579 |
| distill | 0.625 | 0.597 | 0.565 |
| replay_distill | 0.625 | 0.610 | 0.606 |

## 4. Findings

- ✅ Upper-bound check holds: no CL method beats `joint`.
- Lowest forgetting on Farm A: **replay_distill** (+0.0196).
- `naive` forgetting = +0.0529 (expected lower bound).
- **replay_distill** cuts Farm-A forgetting by 0.0333 vs `naive`.
- Worst CL forgetting: distill (+0.0602).
- Best CL final_avg_care: replay_distill (0.575).
- Note: final_avg_care gaps are often within ~1 std; the robust signal is the **forgetting / backward-transfer** comparison.

## 5. Figures

![Forgetting curves](fig_forgetting_curves.png)

![Final per-farm CARE](fig_final_per_farm_care.png)
