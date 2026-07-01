# Align-mode comparison (shared_only vs adapter)

Mean over seeds. Higher CARE = better; lower forgetting = better.

## final_avg_care (higher better)

| strategy | adapter | shared_only |
| --- | --- | --- |
| distill | 0.5940 | 0.5546 |
| ewc | 0.6045 | 0.5533 |
| joint | 0.5998 | 0.5765 |
| naive | 0.5998 | 0.5615 |
| replay | 0.6067 | 0.5707 |
| replay_distill | 0.6070 | 0.5751 |

## forgetting_A (lower better)

| strategy | adapter | shared_only |
| --- | --- | --- |
| distill | 0.0009 | 0.0602 |
| ewc | -0.0135 | 0.0557 |
| joint | — | — |
| naive | 0.0010 | 0.0529 |
| replay | -0.0007 | 0.0466 |
| replay_distill | -0.0037 | 0.0196 |
