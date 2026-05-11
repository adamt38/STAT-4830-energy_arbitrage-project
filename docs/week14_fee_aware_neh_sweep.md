# Week14 Fee-aware Baseline vs NEH vs Kelly

Runs: week14_M_seed7, week14_MF_seed7, week14_M_smoke
Fee grid: 0, 10, 50, 100 bps (per unit L1 turnover).

| Run | fee (bps) | Strategy | Total log-wealth | Sortino | Max DD | avg turnover |
|---|---:|---|---:|---:|---:|---:|
| week14_M_seed7 | 0.0 | kelly | +0.4024 | +0.0635 | -27.06% | 0.0115 |
| week14_M_seed7 | 0.0 | baseline_pipeline | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 0.0 | baseline_dynamic_feeaware | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 0.0 | neh_buyhold | -0.5162 | -0.0595 | -49.22% | 0.0000 |
| week14_M_seed7 | 10.0 | kelly | +0.3853 | +0.0613 | -27.50% | 0.0115 |
| week14_M_seed7 | 10.0 | baseline_pipeline | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 10.0 | baseline_dynamic_feeaware | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 10.0 | neh_buyhold | -0.5162 | -0.0595 | -49.22% | 0.0000 |
| week14_M_seed7 | 50.0 | kelly | +0.3168 | +0.0525 | -29.35% | 0.0115 |
| week14_M_seed7 | 50.0 | baseline_pipeline | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 50.0 | baseline_dynamic_feeaware | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 50.0 | neh_buyhold | -0.5162 | -0.0595 | -49.22% | 0.0000 |
| week14_M_seed7 | 100.0 | kelly | +0.2312 | +0.0415 | -31.72% | 0.0115 |
| week14_M_seed7 | 100.0 | baseline_pipeline | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 100.0 | baseline_dynamic_feeaware | +0.1838 | +0.0370 | -21.48% | 0.0000 |
| week14_M_seed7 | 100.0 | neh_buyhold | -0.5162 | -0.0595 | -49.22% | 0.0000 |
| week14_MF_seed7 | 0.0 | kelly | +0.2617 | +0.0427 | -30.32% | 0.0117 |
| week14_MF_seed7 | 0.0 | baseline_pipeline | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 0.0 | baseline_dynamic_feeaware | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 0.0 | neh_buyhold | -0.1670 | -0.0063 | -39.20% | 0.0000 |
| week14_MF_seed7 | 10.0 | kelly | +0.2437 | +0.0404 | -30.73% | 0.0117 |
| week14_MF_seed7 | 10.0 | baseline_pipeline | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 10.0 | baseline_dynamic_feeaware | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 10.0 | neh_buyhold | -0.1670 | -0.0063 | -39.20% | 0.0000 |
| week14_MF_seed7 | 50.0 | kelly | +0.1716 | +0.0314 | -32.38% | 0.0117 |
| week14_MF_seed7 | 50.0 | baseline_pipeline | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 50.0 | baseline_dynamic_feeaware | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 50.0 | neh_buyhold | -0.1670 | -0.0063 | -39.20% | 0.0000 |
| week14_MF_seed7 | 100.0 | kelly | +0.0814 | +0.0202 | -34.64% | 0.0117 |
| week14_MF_seed7 | 100.0 | baseline_pipeline | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 100.0 | baseline_dynamic_feeaware | +0.3209 | +0.0591 | -24.36% | 0.0000 |
| week14_MF_seed7 | 100.0 | neh_buyhold | -0.1670 | -0.0063 | -39.20% | 0.0000 |
| week14_M_smoke | 0.0 | kelly | +0.0340 | +0.0126 | -26.19% | 0.0392 |
| week14_M_smoke | 0.0 | baseline_pipeline | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 0.0 | baseline_dynamic_feeaware | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 0.0 | neh_buyhold | -0.3319 | -0.0437 | -45.79% | 0.0000 |
| week14_M_smoke | 10.0 | kelly | -0.0193 | +0.0033 | -28.19% | 0.0392 |
| week14_M_smoke | 10.0 | baseline_pipeline | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 10.0 | baseline_dynamic_feeaware | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 10.0 | neh_buyhold | -0.3319 | -0.0437 | -45.79% | 0.0000 |
| week14_M_smoke | 50.0 | kelly | -0.2326 | -0.0331 | -39.05% | 0.0392 |
| week14_M_smoke | 50.0 | baseline_pipeline | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 50.0 | baseline_dynamic_feeaware | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 50.0 | neh_buyhold | -0.3319 | -0.0437 | -45.79% | 0.0000 |
| week14_M_smoke | 100.0 | kelly | -0.4992 | -0.0775 | -52.21% | 0.0392 |
| week14_M_smoke | 100.0 | baseline_pipeline | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 100.0 | baseline_dynamic_feeaware | +0.0954 | +0.0238 | -22.06% | 0.0000 |
| week14_M_smoke | 100.0 | neh_buyhold | -0.3319 | -0.0437 | -45.79% | 0.0000 |

## Headline deltas at 10 bps

| Run | Kelly - dynamic fee-aware baseline (log-w) | Kelly - NEH buy-hold (log-w) | dynamic fee-aware baseline - NEH (log-w) |
|---|---:|---:|---:|
| week14_M_seed7 | +0.2015 | +0.9015 | +0.7000 |
| week14_MF_seed7 | -0.0773 | +0.4107 | +0.4879 |
| week14_M_smoke | -0.1148 | +0.3126 | +0.4274 |
