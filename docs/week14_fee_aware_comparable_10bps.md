# Comparable Fee-aware Summary (10 bps)

This matches recent reporting style: log-wealth delta, Sortino delta, max-DD delta on holdout.

| Run | Kelly vs dynamic baseline: log-w Δ | Sortino Δ | MaxDD Δ (pp) | Kelly vs NEH: log-w Δ | Sortino Δ |
|---|---:|---:|---:|---:|---:|
| week14_M_seed7 | +0.2015 | +0.0243 (CI [-0.0474,+0.1077], frac+ 74.3%) | -6.02 | +0.9015 | +0.1208 (CI [+0.0221,+0.2390], frac+ 99.2%) |
| week14_MF_seed7 | -0.0773 | -0.0186 (CI [-0.0591,+0.0210], frac+ 16.7%) | -6.37 | +0.4107 | +0.0468 (CI [-0.0224,+0.1207], frac+ 89.9%) |
| week14_M_smoke | -0.1148 | -0.0205 (CI [-0.0601,+0.0207], frac+ 15.8%) | -6.14 | +0.3126 | +0.0470 (CI [-0.0123,+0.1142], frac+ 93.3%) |
