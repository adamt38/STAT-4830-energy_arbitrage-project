# Week 4 Self-Critique

## OBSERVE

After re-reading the report and re-running the notebook end-to-end, the core pipeline works: data ingestion, differentiable battery simulation, gradient-based optimization, and a three-case validation suite all execute without errors. The model does learn to buy low and sell high on real NYISO price data, and the code now faithfully implements the report's mathematical formulation (including the degradation penalty $\lambda(c_t + d_t)^2$ with $\lambda = 5.0$ and a unified revenue formula across the main loop and validation runner). However, the training curve remains unstable (profit oscillates across epochs with transient SoC violations), and the report still lacks a convex baseline for comparison.

---

## ORIENT

### Strengths (Max 3)
- **Working end-to-end pipeline.** The `DifferentiableBattery` model ingests live NYISO data, optimizes via Adam with a degradation penalty, and produces interpretable charge/discharge schedules. SoC violations are near-zero for most epochs, with only transient violations during oscillatory phases.
- **Clear mathematical formulation.** The report lays out the objective, dynamics, and constraints precisely enough to reproduce the approach.
- **Thoughtful validation design.** The three synthetic test cases (flat, negative, spike) directly probe whether the optimizer exhibits the correct qualitative behavior, and all three pass.

### Areas for Improvement (Max 3)
1. **Convergence instability.** Profit oscillates significantly across epochs — rising to ~\$60 by epoch 500, dropping to ~\$27 at epoch 600 (with 42 transient SoC violations), then recovering to ~\$55. The learning rate (0.1) is likely too high, and the fixed penalty coefficient creates a non-smooth loss landscape. Without stable convergence, the reported profit is not trustworthy.
2. **Missing QP baseline.** The entire project thesis depends on comparing differentiable physics against convex relaxation, but the CVXPY implementation does not yet exist. Without it, the report cannot answer its own research question.
3. **Transient constraint violations.** Although SoC violations are zero for most epochs, 42 violations appeared at epoch 600 during an oscillatory phase. The soft-penalty approach eventually corrects these, but hard constraints (via projection or Lagrangian methods) would guarantee feasibility at every step.

### Critical Risks / Assumptions
We are currently assuming that the soft-penalty approach will scale to 8,760 steps without gradient issues. The current experiment only covers ~177 steps (one day at 5-min resolution). Scaling to a full year may expose vanishing gradients through the cumulative-sum dynamics, which would require architectural changes (e.g., truncated BPTT or a recurrent formulation). We also assume that a single penalty coefficient will work across different price regimes; this has not been tested.

---

## DECIDE

### Concrete Next Actions (Max 3)
1. **Implement the CVXPY baseline** on the same NYISO data to produce a convex-optimal profit number and SoC trajectory. This directly addresses Area #2 and is the highest priority.
2. **Add a learning-rate scheduler** (e.g., cosine annealing or reduce-on-plateau) to the PyTorch loop and re-run to check whether convergence stabilizes. The degradation penalty and unified revenue formula are now implemented; the remaining instability (Area #1) and transient violations (Area #3) are likely addressable via learning-rate tuning.
3. **Scale the experiment to one week of hourly data** (~168 steps) as an intermediate step toward the full-year horizon, and log gradient norms to detect early signs of vanishing gradients.

---

## ACT

### Resource Needs
- Need to learn CVXPY's QP interface for time-series problems with coupling constraints (SoC dynamics). The [CVXPY tutorial on portfolio optimization](https://www.cvxpy.org/examples/index.html) is the closest analog and will be the starting reference.
- Will use `torch.optim.lr_scheduler.CosineAnnealingLR` for the learning-rate schedule; no new dependencies required.
- May need `gridstatus` historical data access for multi-day experiments; need to verify that the free tier supports date-range queries.
