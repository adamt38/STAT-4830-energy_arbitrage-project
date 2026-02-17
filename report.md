# Project Report: Differentiable Physics vs. Convex Relaxation for Battery Arbitrage

## Problem Statement

### What are you optimizing?
We are rigorously benchmarking **Convex Relaxation (QP)** against **Non-Convex Differentiable Physics (PyTorch)** for battery energy arbitrage. The goal is to maximize financial returns while minimizing battery degradation.

### Why does this problem matter?
The core challenge is to determine if the "Optimality Gap" caused by simplifying physics (to achieve convexity) is greater than the "Convergence Error" introduced by the instability of gradient descent in a constrained, non-convex landscape.

### How will you measure success?
Success is defined by the model's ability to "learn" to buy at troughs and sell at peaks purely via gradient descent. We will measure success by comparing the objective value (profit) and constraint violation rates against the QP baseline.

### Constraints & Data
* **Constraints:** The system must strictly respect $0 \le SoC \le 1$ MWh capacity limits.
* **Data:** The current implementation optimizes over a single day of NYISO 5-minute real-time LMP data (~288 intervals). The long-term goal is to scale to an 8,760-step annual time series.

### What could go wrong?
* **Constraint Satisfaction:** Unlike CVXPY, PyTorch does not natively handle hard constraints.
* **Vanishing Gradients:** The deep computational graph (8,760 steps) risks vanishing gradients, making convergence difficult.

---

## Technical Approach

### Mathematical Formulation
We maximize an objective function $J$ composed of Arbitrage Profit minus a Degradation Penalty:

$$
\text{Maximize } J = \sum_{t=1}^{T} \Big( \underbrace{P_t (d_t - c_t)}_{\text{Revenue}} - \underbrace{\lambda (c_t + d_t)^2}_{\text{Degradation Penalty}} \Big)
$$

Where:
* $P_t$: Price at time $t$
* $d_t, c_t$: Discharge and Charge amounts
* $\lambda$: Degradation penalty coefficient. We derive this such that the quadratic penalty roughly equals the average cost per MWh ($k$) at nominal power ($P_{nom}$): $\lambda \approx \frac{k}{P_{nom}}$.

### Algorithm & Implementation
* **Architecture:** We are implementing a custom `DifferentiableBattery` class in PyTorch to simulate charge/discharge dynamics.
* **Optimization:** We use the Adam optimizer to update control variables.
* **Constraint Strategy:** To handle the lack of native constraints, we use a **soft quadratic penalty** on SoC violations (penalty coefficient = 2,000), which penalizes any excursion outside the $[0, SoC_{max}]$ bounds. Future work will explore Projected Gradient Descent and Lagrangian Relaxation as alternatives.

### Validation Methods
We utilize a "Proof of Life" validation suite: training the model on three synthetic price scenarios (flat prices, negative prices, and a single price spike) to verify that the optimizer learns qualitatively correct behavior before evaluating on real-world data.

---

## Initial Results

### Evidence of a Working Implementation
We ran our differentiable battery on **live NYISO real-time 5-minute LMP data** for the N.Y.C. zone (177 intervals, average price \$80.71/MWh). The optimizer was configured with Adam (lr=0.1), 1,000 epochs, a soft SoC penalty coefficient of 2,000, and a degradation penalty $\lambda = 5.0$. Key observations:

* **Profit Trajectory:** The model showed clear learning, rising from \$0 at epoch 0 to \$12.87 by epoch 100, reaching \$59.86 by epoch 500. However, convergence was **highly oscillatory** — profit dropped to \$27.09 at epoch 600 (with 42 transient SoC violations), recovered to \$49.81 by epoch 800, and ended at \$54.85 at epoch 900. This instability is a central finding: gradient descent on this non-convex landscape is noisy, confirming the "Convergence Error" concern from our problem statement.
* **Constraint Satisfaction:** The soft-penalty approach achieved **zero SoC violations** for the majority of epochs, though transient violations (42 at epoch 600) appeared during oscillatory phases. These violations are quickly corrected by the penalty term, but their presence underscores the fragility of the soft-constraint approach.

### Validation Suite Results
We designed three synthetic test cases to verify qualitative behavior:

| Test Case | Expected Behavior | Profit | Max SoC Violation | Runtime |
|---|---|---|---|---|
| Flat Prices ($50/MWh) | Discharge initial stored energy (SoC starts at 50%), then idle | $13.96 | 0.0000 | 0.30 s |
| Negative Prices (-$5 to -$10) | Charge (action > 0) | $1.91 | 0.0000 | 0.27 s |
| Single Price Spike ($500 at t=50) | Discharge at spike | $46.43 | 0.0000 | 0.26 s |

All three tests pass the "eye test": the model charges during negative prices, discharges into the spike, and sells its initial stored energy when prices are flat. The profit in the flat case (\$13.96) comes from selling the battery's initial 50% SoC — this is physically correct behavior, not residual cycling. The degradation penalty ($\lambda = 5.0$) is included in both the main loop and validation, reducing unnecessary cycling.

### Current Limitations
* **Convergence Instability:** The profit oscillation during training suggests that lr=0.1 may be too aggressive, or that the penalty-based constraint strategy creates a rugged loss landscape. A learning-rate schedule or Lagrangian relaxation may help.
* **No QP Baseline Yet:** We cannot quantify the "Optimality Gap" until the CVXPY benchmark is implemented.
* **Short Horizon:** The current experiment uses a single day of 5-minute data (~177 steps), far short of the planned 8,760-step annual horizon. Scaling up may exacerbate vanishing-gradient issues.

### Resource Usage
Each validation case (1,000 epochs on a 100-step horizon) ran in under 0.31 seconds and used less than 0.02 MB of peak memory. The full ~177-step main optimization (1,000 epochs) completed in seconds on a CPU, indicating that compute is not a bottleneck at this scale.

---

## Next Steps

### Immediate Improvements
* **Stabilize Convergence:** Implement a learning-rate scheduler (e.g., cosine annealing or reduce-on-plateau) and experiment with lower initial learning rates. Evaluate whether Lagrangian relaxation produces smoother convergence than the current quadratic penalty.
* ~~**Unify Revenue Formulation:**~~ *Done.* The main training loop and validation runner now use the same revenue formulation (`−control × price − λ × control²`), directly implementing the report's mathematical objective.
* ~~**Add the Degradation Penalty:**~~ *Done.* The $\lambda(c_t + d_t)^2$ degradation term is now included in both the main optimization loop and the validation runner, with $\lambda = 5.0$.

### Technical Challenges to Address
* **Implement the QP Baseline (CVXPY):** This is critical for the core research question. Without the convex benchmark, we cannot measure the optimality gap.
* **Scale to a Full Year:** Move from a single day (~177 steps) to a full year (8,760 hourly steps). Monitor for vanishing gradients and memory growth, and consider chunked / truncated backpropagation through time if needed.
* **Projected Gradient Descent:** Explore hard projection of SoC after each optimizer step (clamp SoC to $[0, SoC_{max}]$) as an alternative to the soft penalty, and compare feasibility and profit.

### Questions for Course Staff
* Is there a recommended approach for benchmarking non-convex PyTorch solutions against CVXPY on problems of this size (8,760 variables)?
* Are there best practices for balancing penalty coefficients vs. Lagrangian multiplier updates in differentiable physics settings?

### Alternative Approaches to Try
* **Augmented Lagrangian Method:** Replace the fixed penalty with adaptive dual-variable updates for tighter constraint satisfaction.
* **Neural Network Policy:** Instead of directly optimizing per-step actions, train a small network that maps price features to actions, enabling generalization to unseen price trajectories.

### What We've Learned So Far
Gradient descent *can* learn basic arbitrage behavior (buy low, sell high) from price signals alone, and the `tanh` parameterization effectively enforces power limits without projection. However, the convergence path is far noisier than expected, reinforcing the value of the planned comparison with a convex solver.
