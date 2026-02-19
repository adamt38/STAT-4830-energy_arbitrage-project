"""
1D gradient descent on a quadratic objective using PyTorch.
Produces a diagnostics figure at figures/gd_torch_quadratic_diagnostics.png.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Quadratic f(x) = (x - 2)^2; minimum at x = 2
def objective(x: torch.Tensor) -> torch.Tensor:
    return (x - 2.0).pow(2).sum()

def main() -> None:
    x = torch.tensor(0.0, requires_grad=True)
    lr = 0.1
    steps = 50
    history = []

    for _ in range(steps):
        loss = objective(x)
        history.append(loss.item())
        loss.backward()
        with torch.no_grad():
            x.sub_(x.grad, alpha=lr)
            x.grad.zero_()

    # Figure path: repo root is parent of script/
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    fig_dir = repo_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "gd_torch_quadratic_diagnostics.png"

    fig, ax = plt.subplots()
    ax.plot(history, color="tab:blue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("1D quadratic GD (PyTorch)")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
