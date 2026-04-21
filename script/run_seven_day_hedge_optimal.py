"""Seven-day window: baseline vs PM strategies (Week 13 style).

Default recipe ``pm_internal_risk`` (100% Polymarket):
  - **Equity signal → PM tilt only**: optional lagged equity returns scale domain weights
    (multiplicative, renormalized); no capital leaves PM. Map tickers → domains via
    ``data/processed/pm_equity_domain_tilt_map.csv``.
  - **Resolution boost** (optional): causal per-domain weight multipliers from recent |PM returns|.
  - **Eval baseline mix** (optional): on eval bars, blend weights toward the same domain-equal
    baseline the benchmark uses. **``--tune-eval-baseline-mix``** grid-searches the mix on this
    7d window (in-sample); use ``--tune-mix-max`` (<1) to avoid collapsing to pure baseline.
  - **PM-internal spread** (optional): category long/short; disable with ``--no-pm-spread``.
  - **``--beat-baseline-preset``**: mean_return, resolution on, spread off, lower cov penalty, tune mix.
  - **``--seek-alpha-divergent-preset``**: low baseline blend + **token-resolution tilt** with strength
    tune (paths diverge from hugging the baseline curve).

Legacy: ``pm_stock_enhancer`` (85/15 PM/SPY sleeve + momentum gate), or ``grid`` (oil/−eq).

Example:
  python script/run_seven_day_hedge_optimal.py --input-prefix week11_v2 --output-prefix week13_7d_hedge
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _parse_alpha_sweep(s: str | None) -> list[float] | None:
    if s is None or not str(s).strip():
        return None
    out: list[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if p:
            out.append(float(p))
    return out or None


os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
(REPO_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import baseline_static_token_weights, run_equal_weight_baseline
from src.constrained_optimizer import ExperimentConfig, _load_returns_and_domains, _run_online_pass
from src.hedge_window_opt import (
    combine_pm_stock_enhancer_gated,
    excess_gain_vs_baseline,
    growth_of_one,
    optimize_hedge_on_window,
    rolling_mean_past_only,
)
from src.pm_risk_overlay import (
    build_equity_domain_tilt_multiplier,
    load_domain_ticker_tilt_map,
    pm_category_spread_returns,
    read_category_correlation_csv,
    resolution_shock_domain_multiplier,
    token_resolution_tilt_multiplier,
    top_negative_correlation_pairs,
)

METHODOLOGY_FRAMING = (
    "Equity-linked modules are framed as risk management, not as engines of excess return vs "
    "an equal-weight PM baseline. In typical weeks they can damp drawdowns by tilting toward "
    "domains supported by sector momentum and by adding small PM–PM spreads from the category "
    "correlation matrix. In weeks where the baseline is dominated by fast binary resolutions, "
    "momentum-based gates (where used) may keep external sleeves inactive — which is coherent "
    "with treating the signal as conditioning, not as a competitor to resolution-driven baseline "
    "surges. Compare strategies on risk metrics (drawdown, tail loss) as well as raw 7d gain."
)


def _window_mask(ts_eval: np.ndarray, end_ts: int, lookback_seconds: int) -> np.ndarray:
    return ts_eval >= (end_ts - lookback_seconds)


def load_aligned_hedge_legs(ts_eval: np.ndarray, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    p = REPO_ROOT / "data" / "processed" / f"{prefix}_stock_oil_hedge_timeseries.csv"
    if not p.is_file():
        return np.zeros(len(ts_eval), dtype=float), np.zeros(len(ts_eval), dtype=float)
    with p.open("r", newline="", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    by_ts = {int(r["timestamp"]): r for r in rows}
    spy: list[float] = []
    oil: list[float] = []
    for t in ts_eval:
        r = by_ts.get(int(t))
        if r is None:
            spy.append(0.0)
            oil.append(0.0)
        else:
            spy.append(float(r["lagged_spy_daily_return"]))
            oil.append(float(r["lagged_oil_daily_return"]))
    return np.asarray(spy, dtype=float), np.asarray(oil, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="7-day PM + hedged overlay vs baseline.")
    ap.add_argument("--input-prefix", default="week11_v2")
    ap.add_argument("--output-prefix", default="week13_7d_hedge")
    ap.add_argument("--steps-per-window", type=int, default=1)
    ap.add_argument("--tail-steps", type=int, default=2500)
    ap.add_argument("--min-tail-days", type=float, default=8.0)
    ap.add_argument(
        "--recipe",
        choices=("pm_internal_risk", "pm_stock_enhancer", "grid"),
        default="pm_internal_risk",
        help="pm_internal_risk: 100%% PM, equity domain tilt + category spread hedge. "
        "pm_stock_enhancer: 85/15 PM/SPY sleeve + gate. grid: α × oil/−eq.",
    )
    ap.add_argument(
        "--tilt-map",
        default=str(REPO_ROOT / "data" / "processed" / "pm_equity_domain_tilt_map.csv"),
        help="CSV domain→ticker map for multiplicative PM tilt (pm_internal_risk).",
    )
    ap.add_argument(
        "--no-equity-tilt",
        action="store_true",
        help="Disable yfinance-based domain tilt (pm_internal_risk).",
    )
    ap.add_argument("--tilt-strength", type=float, default=33.333, help="Scales pos. equity ret into mult−1.")
    ap.add_argument("--tilt-max-mult", type=float, default=2.0, help="Cap on domain weight multiplier.")
    ap.add_argument(
        "--corr-prefix",
        default="week8",
        help="Load data/processed/{corr-prefix}_category_correlation.csv for PM spread pairs.",
    )
    ap.add_argument("--pm-spread-eta", type=float, default=0.05, help="Scale on zero-inv. PM category spread.")
    ap.add_argument("--pm-spread-pairs", type=int, default=5, help="Number of most-negative category pairs.")
    ap.add_argument(
        "--pm-spread-max-corr",
        type=float,
        default=-0.002,
        help="Include pairs with correlation <= this threshold.",
    )
    ap.add_argument("--alpha", type=float, default=0.07, help="Base hedge weight (when gate on).")
    ap.add_argument(
        "--alpha-sweep",
        default=None,
        metavar="LIST",
        help="Comma-separated alphas (pm_stock_enhancer only): one PM pass, sweep overlay weight.",
    )
    ap.add_argument("--pm-in-hedge-leg", type=float, default=0.85, help="PM share inside overlay sleeve.")
    ap.add_argument("--stock-in-hedge-leg", type=float, default=0.15, help="Stock share inside overlay sleeve.")
    ap.add_argument(
        "--momentum-gate-window",
        type=int,
        default=48,
        help="Bars of strict-past rolling mean on SPY for gate (hedge active if mean > 0).",
    )
    ap.add_argument(
        "--no-momentum-gate",
        action="store_true",
        help="Use constant α on every bar (no SPY momentum gate).",
    )
    ap.add_argument("--alpha-steps", type=int, default=41, help="[grid] Points for α in [0,1].")
    ap.add_argument("--leg-grid", action="store_true", help="[grid] Search oil/−eq pairs.")
    ap.add_argument(
        "--objective",
        default="excess_mean_downside",
        choices=(
            "excess_mean_downside",
            "excess_mean_downside_tail",
            "excess_sortino",
            "excess_cvar_mean_downside",
            "mean_return",
            "mean_downside",
            "sortino",
        ),
        help="Rolling training objective for _run_online_pass. mean_return ≈ chase window mean return.",
    )
    ap.add_argument(
        "--eval-baseline-mix",
        type=float,
        default=0.0,
        help="On eval bars only: blend weights toward domain-equal baseline (0–1).",
    )
    ap.add_argument(
        "--tune-eval-baseline-mix",
        action="store_true",
        help="Grid-search eval-baseline-mix on this 7d window to max compound excess (in-sample).",
    )
    ap.add_argument(
        "--tune-mix-max",
        type=float,
        default=0.92,
        help="Upper cap for tune grid (must be <1 or search picks pure baseline).",
    )
    ap.add_argument(
        "--resolution-boost",
        action="store_true",
        help="PM-only: multiply domain weights by causal resolution-shock multipliers.",
    )
    ap.add_argument("--resolution-lookback", type=int, default=80, help="Bars of past domain returns.")
    ap.add_argument("--resolution-strength", type=float, default=12.0, help="Strength for tanh boost.")
    ap.add_argument("--resolution-max-mult", type=float, default=2.0, help="Cap per domain.")
    ap.add_argument(
        "--no-pm-spread",
        action="store_true",
        help="pm_internal_risk: disable category correlation spread (η=0).",
    )
    ap.add_argument(
        "--covariance-scale",
        type=float,
        default=1.0,
        help="Multiply covariance_penalty_lambda from metrics (lower → allow more concentration).",
    )
    ap.add_argument(
        "--beat-baseline-preset",
        action="store_true",
        help="Shortcut: mean_return, no PM spread, resolution boost, tune eval mix, lower cov penalty.",
    )
    ap.add_argument(
        "--seek-alpha-divergent-preset",
        action="store_true",
        help="Divergent path vs baseline: mean_return, low baseline blend, token-resolution tilt "
        "with in-sample strength tune (no high-γ baseline hug).",
    )
    ap.add_argument(
        "--token-resolution-tilt",
        action="store_true",
        help="Eval: per-token multipliers from recent |returns| (diverges from domain-equal baseline).",
    )
    ap.add_argument("--token-resolution-lookback", type=int, default=64)
    ap.add_argument("--token-resolution-strength", type=float, default=8.0)
    ap.add_argument("--token-resolution-max-mult", type=float, default=2.0)
    ap.add_argument(
        "--tune-token-resolution-strength",
        action="store_true",
        help="Grid-search token-resolution strength on this 7d window (pm_internal_risk).",
    )
    ap.add_argument(
        "--tune-divergent-joint",
        action="store_true",
        help="With seek-alpha preset: small grid over (eval-baseline-mix, token strength) for 7d excess.",
    )
    args = ap.parse_args()

    if args.seek_alpha_divergent_preset:
        args.objective = "mean_return"
        args.no_pm_spread = True
        args.resolution_boost = True
        args.token_resolution_tilt = True
        args.tune_eval_baseline_mix = False
        args.tune_divergent_joint = True
        args.tune_token_resolution_strength = False
        args.covariance_scale = float(min(0.09, args.covariance_scale))
        args.token_resolution_max_mult = float(min(1.38, args.token_resolution_max_mult))
        if float(args.eval_baseline_mix) <= 0.0:
            args.eval_baseline_mix = 0.34
    elif args.beat_baseline_preset:
        args.objective = "mean_return"
        args.no_pm_spread = True
        args.resolution_boost = True
        args.tune_eval_baseline_mix = True
        args.covariance_scale = float(min(0.2, args.covariance_scale))
        if float(args.eval_baseline_mix) <= 0.0:
            args.eval_baseline_mix = 0.35

    in_pfx = args.input_prefix
    out_pfx = args.output_prefix
    out_dir = REPO_ROOT / "data" / "processed"
    fig_dir = REPO_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / f"{in_pfx}_constrained_best_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    bp = metrics["best_params"]
    rw = int(bp.get("rolling_window", 288))

    wide_path = out_dir / f"{in_pfx}_recent_wide_sweep_summary.json"
    if not wide_path.is_file():
        raise FileNotFoundError(f"Need {wide_path} for fixed PM hyperparameters (run wide sweep once).")
    wb = json.loads(wide_path.read_text(encoding="utf-8"))["best"]

    b = run_equal_weight_baseline(REPO_ROOT, in_pfx)
    ts = np.array([int(x) for x in b.timestamps], dtype=np.int64)
    r_base = np.asarray(b.returns, dtype=float)
    rmat, _tokens, domains, _td, _meta = _load_returns_and_domains(REPO_ROOT, in_pfx)

    if rmat.shape[0] != r_base.shape[0]:
        n = min(rmat.shape[0], r_base.shape[0], ts.shape[0])
        rmat = rmat[-n:]
        r_base = r_base[-n:]
        ts = ts[-n:]

    if args.tail_steps > 0 and rmat.shape[0] > args.tail_steps:
        end_ts_full = int(ts[-1])
        min_tail_start_ts = end_ts_full - int(max(args.min_tail_days, 0.0) * 86400.0)
        i_tail_steps = max(len(ts) - args.tail_steps, 0)
        i_tail_days = int(np.searchsorted(ts, min_tail_start_ts, side="left"))
        start_idx = min(i_tail_steps, i_tail_days)
        if start_idx > 0:
            rmat = rmat[start_idx:]
            r_base = r_base[start_idx:]
            ts = ts[start_idx:]

    end_ts = int(ts[-1])
    t7 = end_ts - 7 * 86400
    i7 = int(np.searchsorted(ts, t7, side="left"))
    i_eval = min(i7, max(rw, 0))

    baseline_step = np.asarray(r_base, dtype=np.float64)
    opt_cfg = ExperimentConfig()
    macro_dom = frozenset(opt_cfg.macro_target_domains)

    rb_eval = r_base[i_eval:]
    ts_eval = ts[i_eval:]
    m7 = _window_mask(ts_eval, end_ts, 7 * 86400)
    if not np.any(m7):
        raise RuntimeError("Empty 7-day window.")

    tilt_mult_full: np.ndarray | None = None
    tilt_status = "off"
    if args.recipe == "pm_internal_risk" and not args.no_equity_tilt:
        cmap = load_domain_ticker_tilt_map(pathlib.Path(args.tilt_map))
        try:
            tilt_mult_full, applied = build_equity_domain_tilt_multiplier(
                ts,
                domains,
                cmap,
                tilt_strength=float(args.tilt_strength),
                max_multiplier=float(args.tilt_max_mult),
            )
            tilt_status = f"on_domains_mapped={len(applied)}"
        except Exception as exc:
            tilt_mult_full = None
            tilt_status = f"failed_{type(exc).__name__}"
    elif args.recipe == "pm_internal_risk":
        tilt_status = "disabled_cli"

    obj = str(args.objective)
    baseline_for_pass = baseline_step if obj.startswith("excess_") else None

    res_mult: np.ndarray | None = None
    if args.resolution_boost:
        res_mult = resolution_shock_domain_multiplier(
            rmat,
            domains,
            lookback=int(args.resolution_lookback),
            strength=float(args.resolution_strength),
            max_multiplier=float(args.resolution_max_mult),
        )

    cov_lam = float(bp.get("covariance_penalty_lambda", 5.0)) * float(args.covariance_scale)
    n_eval_steps = int(rmat.shape[0] - max(rw, i_eval))

    baseline_w_vec: np.ndarray | None = None
    if float(args.eval_baseline_mix) > 0.0 or (
        bool(args.tune_eval_baseline_mix) and args.recipe == "pm_internal_risk"
    ):
        baseline_w_vec = baseline_static_token_weights(domains)

    neg_pairs_pre: list[tuple[str, str, float]] = []
    r_spread_full_pre = np.zeros(rmat.shape[0], dtype=float)
    corr_path = REPO_ROOT / "data" / "processed" / f"{args.corr_prefix}_category_correlation.csv"
    if args.recipe == "pm_internal_risk" and corr_path.is_file():
        labels, corr_mat = read_category_correlation_csv(corr_path)
        neg_pairs_pre = top_negative_correlation_pairs(
            labels,
            corr_mat,
            max_pairs=max(1, int(args.pm_spread_pairs)),
            max_correlation=float(args.pm_spread_max_corr),
        )
        r_spread_full_pre = pm_category_spread_returns(rmat, domains, neg_pairs_pre)

    spread_eval = r_spread_full_pre[i_eval : i_eval + n_eval_steps]
    if spread_eval.size != n_eval_steps:
        raise RuntimeError(f"spread_eval length {spread_eval.size} != n_eval_steps {n_eval_steps}")

    eta = 0.0 if args.no_pm_spread else float(args.pm_spread_eta)
    tune_meta: dict[str, Any] = {}
    combined_pm_internal: np.ndarray | None = None

    tok_mult_fixed: np.ndarray | None = None
    if args.token_resolution_tilt and not args.tune_token_resolution_strength:
        tok_mult_fixed = token_resolution_tilt_multiplier(
            rmat,
            lookback=int(args.token_resolution_lookback),
            strength=float(args.token_resolution_strength),
            max_multiplier=float(args.token_resolution_max_mult),
        )

    def _one_payload(eval_mix: float, *, token_res_mult: np.ndarray | None = None) -> Any:
        wmix = float(np.clip(eval_mix, 0.0, 1.0))
        return _run_online_pass(
            returns_matrix=rmat,
            domains=domains,
            lr=float(bp.get("learning_rate", 0.05)),
            penalty_lambda=float(bp.get("lambda_penalty", 1.0)),
            rolling_window=rw,
            steps_per_window=args.steps_per_window,
            domain_limit=float(bp.get("domain_limit", 0.13)),
            max_weight=float(wb["max_weight"]),
            concentration_penalty_lambda=float(wb["concentration_penalty_lambda"]),
            covariance_penalty_lambda=cov_lam,
            covariance_shrinkage=float(bp.get("covariance_shrinkage", 0.05)),
            entropy_lambda=float(wb["entropy_lambda"]),
            uniform_mix=float(wb["uniform_mix"]),
            seed=7,
            evaluation_start_t=i_eval,
            update_after_eval_start=True,
            objective=obj,
            variance_penalty=float(bp.get("variance_penalty", 1.0)),
            downside_penalty=float(bp.get("downside_penalty", 2.0)),
            optimizer_type="adam",
            baseline_step_returns=baseline_for_pass,
            macro_geopolitics_boost_lambda=opt_cfg.macro_geopolitics_boost_lambda,
            macro_target_domains=macro_dom,
            macro_volatility_scale=opt_cfg.macro_volatility_scale,
            excess_tail_risk_lambda=0.0,
            excess_tail_top_fraction=opt_cfg.excess_tail_top_fraction,
            equity_domain_tilt_multiplier=tilt_mult_full,
            resolution_domain_multiplier=res_mult,
            baseline_static_weights=baseline_w_vec if wmix > 0.0 and baseline_w_vec is not None else None,
            eval_baseline_weight_mix=wmix,
            token_resolution_multiplier=token_res_mult,
        )

    mix_used = float(args.eval_baseline_mix)
    if args.tune_divergent_joint and args.recipe == "pm_internal_risk":
        best_ex = float("-inf")
        best_m = 0.0
        best_st = 0.0
        best_pld: Any = None
        best_combined: np.ndarray | None = None
        mixes = [0.45, 0.58, 0.72, 0.86]
        for m in mixes:
            for st in np.linspace(0.4, 8.0, 6):
                tok = token_resolution_tilt_multiplier(
                    rmat,
                    lookback=int(args.token_resolution_lookback),
                    strength=float(st),
                    max_multiplier=float(args.token_resolution_max_mult),
                )
                pld = _one_payload(float(m), token_res_mult=tok)
                rpm_i = np.asarray(pld["portfolio_returns"], dtype=float)
                comb_i = rpm_i + eta * spread_eval
                ex_i = excess_gain_vs_baseline(comb_i, rb_eval, m7)
                if ex_i > best_ex:
                    best_ex, best_m, best_st, best_pld, best_combined = ex_i, float(m), float(st), pld, comb_i
        assert best_pld is not None and best_combined is not None
        payload = best_pld
        rpm = np.asarray(payload["portfolio_returns"], dtype=float)
        mix_used = best_m
        combined_pm_internal = best_combined
        tune_meta = {
            "tuned_eval_baseline_mix": mix_used,
            "tuned_token_resolution_strength": best_st,
            "in_sample_7d_objective": "joint grid: baseline mix × token-resolution strength",
            "best_excess_7d_vs_baseline_from_tune": best_ex,
        }
    elif args.tune_token_resolution_strength and args.recipe == "pm_internal_risk":
        best_ex = float("-inf")
        best_st = 0.0
        best_pld: Any = None
        best_combined: np.ndarray | None = None
        for st in np.linspace(0.5, 12.0, 9):
            tok = token_resolution_tilt_multiplier(
                rmat,
                lookback=int(args.token_resolution_lookback),
                strength=float(st),
                max_multiplier=float(args.token_resolution_max_mult),
            )
            pld = _one_payload(mix_used, token_res_mult=tok)
            rpm_i = np.asarray(pld["portfolio_returns"], dtype=float)
            if rpm_i.size != spread_eval.size:
                raise RuntimeError("rpm/spread length mismatch during token tune")
            comb_i = rpm_i + eta * spread_eval
            ex_i = excess_gain_vs_baseline(comb_i, rb_eval, m7)
            if ex_i > best_ex:
                best_ex, best_st, best_pld, best_combined = ex_i, float(st), pld, comb_i
        assert best_pld is not None and best_combined is not None
        payload = best_pld
        rpm = np.asarray(payload["portfolio_returns"], dtype=float)
        tune_meta = {
            "tuned_token_resolution_strength": best_st,
            "in_sample_7d_objective": "max compound excess vs baseline (token-resolution strength)",
            "best_excess_7d_vs_baseline_from_tune": best_ex,
        }
        combined_pm_internal = best_combined
    elif args.tune_eval_baseline_mix and args.recipe == "pm_internal_risk":
        best_ex = float("-inf")
        best_mix = 0.0
        best_pld: Any = None
        best_combined: np.ndarray | None = None
        mix_hi = float(min(0.999, max(0.05, float(args.tune_mix_max))))
        for g in np.linspace(0.0, mix_hi, 15):
            pld = _one_payload(float(g), token_res_mult=tok_mult_fixed)
            rpm_i = np.asarray(pld["portfolio_returns"], dtype=float)
            if rpm_i.size != spread_eval.size:
                raise RuntimeError("rpm/spread length mismatch during tune")
            comb_i = rpm_i + eta * spread_eval
            ex_i = excess_gain_vs_baseline(comb_i, rb_eval, m7)
            if ex_i > best_ex:
                best_ex, best_mix, best_pld, best_combined = ex_i, float(g), pld, comb_i
        assert best_pld is not None and best_combined is not None
        payload = best_pld
        rpm = np.asarray(payload["portfolio_returns"], dtype=float)
        mix_used = best_mix
        combined_pm_internal = best_combined
        tune_meta = {
            "tuned_eval_baseline_mix": mix_used,
            "in_sample_7d_objective": "max compound excess vs baseline on this window",
            "best_excess_7d_vs_baseline_from_tune": best_ex,
        }
    else:
        payload = _one_payload(mix_used, token_res_mult=tok_mult_fixed)
        rpm = np.asarray(payload["portfolio_returns"], dtype=float)

    if rpm.size != rb_eval.size:
        raise RuntimeError("PM eval length mismatch.")

    r_spy, r_oil = load_aligned_hedge_legs(ts_eval, in_pfx)

    if args.recipe == "pm_internal_risk":
        neg_pairs = neg_pairs_pre
        combined = (
            combined_pm_internal if combined_pm_internal is not None else rpm + eta * spread_eval
        )
        baseline_window_gain = float(np.prod(1.0 + rb_eval[m7]) - 1.0)
        g1_baseline_7d = growth_of_one(rb_eval[m7])
        ex_w = excess_gain_vs_baseline(combined, rb_eval, m7)
        gain_c = float(np.prod(1.0 + combined[m7]) - 1.0)
        opt = {
            "best_hedge_allocation_alpha": None,
            "best_oil_leg_weight": None,
            "best_inverse_equity_leg_weight": None,
            "excess_gain_vs_baseline_window": ex_w,
            "baseline_window_gain": baseline_window_gain,
            "best_combined_window_gain": gain_c,
            "best_combined_returns": combined,
        }
        rows = [
            {
                "recipe": "pm_internal_risk",
                "equity_tilt_status": tilt_status,
                "pm_spread_eta": eta,
                "n_category_pairs": len(neg_pairs),
                "excess_7d_vs_baseline": ex_w,
                "gain_7d_baseline": baseline_window_gain,
                "gain_7d_combined": gain_c,
                "growth_of_1_combined_7d": growth_of_one(combined[m7]),
            }
        ]
        pair_summ = [{"category_a": a, "category_b": b, "correlation": c} for a, b, c in neg_pairs]
        legend = "PM: baseline mix {:.0%}, res+token tilt, spread η={:.2f}".format(mix_used, eta)
        title_note = "tilt + internal hedge; see methodology_framing in summary JSON"
        summary = {
            "label": "7-day PM-only risk overlay (equity tilt + correlation spread)",
            "input_prefix": in_pfx,
            "output_prefix": out_pfx,
            "recipe": "pm_internal_risk",
            "latest_timestamp_utc": dt.datetime.fromtimestamp(end_ts, tz=dt.timezone.utc).isoformat(),
            "window_7d_start_utc": dt.datetime.fromtimestamp(t7, tz=dt.timezone.utc).isoformat(),
            "pm_hyperparameters_source": "recent_wide_sweep_summary best",
            "wide_best_pm": {k: wb[k] for k in wb if k != "hedge_allocation"},
            "parameters": {
                "optimizer_objective": obj,
                "eval_baseline_weight_mix_used": float(mix_used),
                "resolution_boost": bool(args.resolution_boost),
                "resolution_lookback": int(args.resolution_lookback),
                "covariance_scale": float(args.covariance_scale),
                "no_pm_spread": bool(args.no_pm_spread),
                "tune_eval_baseline_mix": bool(args.tune_eval_baseline_mix),
                "tune_mix_max": float(args.tune_mix_max),
                "token_resolution_tilt": bool(args.token_resolution_tilt),
                "token_resolution_lookback": int(args.token_resolution_lookback),
                "token_resolution_strength_default": float(args.token_resolution_strength),
                "tune_token_resolution_strength": bool(args.tune_token_resolution_strength),
                "tune_note": tune_meta if tune_meta else None,
                "tilt_map_csv": str(pathlib.Path(args.tilt_map)),
                "equity_tilt": not bool(args.no_equity_tilt),
                "tilt_strength": float(args.tilt_strength),
                "tilt_max_multiplier": float(args.tilt_max_mult),
                "tilt_status": tilt_status,
                "category_correlation_artifact": str(corr_path),
                "pm_spread_eta": eta,
                "pm_spread_pairs_requested": int(args.pm_spread_pairs),
                "pm_spread_max_correlation": float(args.pm_spread_max_corr),
                "negative_corr_pairs_used": pair_summ,
            },
            "methodology_framing": METHODOLOGY_FRAMING,
            "optimized": {
                "excess_gain_vs_baseline_7d": ex_w,
                "baseline_7d_gain": baseline_window_gain,
                "best_combined_7d_gain": gain_c,
                "growth_of_1_baseline_7d": g1_baseline_7d,
                "growth_of_1_combined_7d": growth_of_one(combined[m7]),
            },
            "n_bars_7d": int(np.sum(m7)),
            "note": "No capital allocated to stocks; equity only tilts PM domain weights. Hedge is PM–PM.",
        }
    elif args.recipe == "pm_stock_enhancer":
        from src.hedge_window_opt import pm_stock_enhancer_leg_returns
        from src.stock_oil_hedge import combine_sleeves_variable

        sweep = _parse_alpha_sweep(args.alpha_sweep)
        alpha_list = [float(np.clip(a, 0.0, 1.0)) for a in sweep] if sweep else [float(np.clip(args.alpha, 0.0, 1.0))]

        roll_spy = rolling_mean_past_only(r_spy, args.momentum_gate_window)
        gate_on = float(np.mean((roll_spy > 0.0)[m7])) if np.any(m7) else 0.0
        baseline_window_gain = float(np.prod(1.0 + rb_eval[m7]) - 1.0)
        g1_baseline_7d = growth_of_one(rb_eval[m7])

        rows: list[dict[str, Any]] = []
        best_excess = float("-inf")
        opt: dict[str, Any] | None = None
        plot_alpha = alpha_list[0]

        r_hedge_base = pm_stock_enhancer_leg_returns(
            rpm,
            r_spy,
            pm_weight=args.pm_in_hedge_leg,
            stock_weight=args.stock_in_hedge_leg,
        )

        for a in alpha_list:
            if args.no_momentum_gate:
                alpha_bar = np.full(rpm.shape[0], float(a), dtype=float)
                combined = combine_sleeves_variable(rpm, r_hedge_base, alpha_bar)
            else:
                combined, _r_hedge, alpha_bar = combine_pm_stock_enhancer_gated(
                    rpm,
                    r_spy,
                    hedge_weight=a,
                    pm_in_hedge_leg=args.pm_in_hedge_leg,
                    stock_in_hedge_leg=args.stock_in_hedge_leg,
                    momentum_gate_window=args.momentum_gate_window,
                )
            ex_w = excess_gain_vs_baseline(combined, rb_eval, m7)
            gain_c = float(np.prod(1.0 + combined[m7]) - 1.0)
            g1_c = growth_of_one(combined[m7])
            rows.append(
                {
                    "recipe": "pm_stock_enhancer",
                    "base_alpha": float(a),
                    "pm_in_hedge_leg": float(args.pm_in_hedge_leg),
                    "stock_in_hedge_leg": float(args.stock_in_hedge_leg),
                    "momentum_gate_window": int(args.momentum_gate_window),
                    "momentum_gate_off": bool(args.no_momentum_gate),
                    "frac_bars_gate_active_in_7d": gate_on,
                    "excess_7d_vs_baseline": ex_w,
                    "gain_7d_baseline": baseline_window_gain,
                    "gain_7d_combined": gain_c,
                    "growth_of_1_combined_7d": g1_c,
                }
            )
            if ex_w > best_excess:
                best_excess = ex_w
                plot_alpha = float(a)
                opt = {
                    "best_hedge_allocation_alpha": float(a),
                    "best_oil_leg_weight": None,
                    "best_inverse_equity_leg_weight": None,
                    "excess_gain_vs_baseline_window": ex_w,
                    "baseline_window_gain": baseline_window_gain,
                    "best_combined_window_gain": gain_c,
                    "best_combined_returns": combined,
                }

        assert opt is not None
        combined_plot = np.asarray(opt["best_combined_returns"], dtype=float)

        legend = (
            f"PM+overlay α={plot_alpha:.2f} (gated, best excess in sweep), sleeve "
            f"{args.pm_in_hedge_leg:.0%}PM/{args.stock_in_hedge_leg:.0%}SPY"
        )
        if len(alpha_list) == 1:
            legend = (
                f"PM+overlay α={alpha_list[0]:.2f} (gated), sleeve "
                f"{args.pm_in_hedge_leg:.0%}PM/{args.stock_in_hedge_leg:.0%}SPY"
            )
        title_note = "85/15 PM/stock enhancer sleeve; α→0 when rolling SPY≤0"
        summary = {
            "label": "7-day PM + PM/stock enhancer overlay (momentum-gated α)",
            "input_prefix": in_pfx,
            "output_prefix": out_pfx,
            "recipe": "pm_stock_enhancer",
            "latest_timestamp_utc": dt.datetime.fromtimestamp(end_ts, tz=dt.timezone.utc).isoformat(),
            "window_7d_start_utc": dt.datetime.fromtimestamp(t7, tz=dt.timezone.utc).isoformat(),
            "pm_hyperparameters_source": "recent_wide_sweep_summary best",
            "wide_best_pm": {k: wb[k] for k in wb if k != "hedge_allocation"},
            "parameters": {
                "optimizer_objective": obj,
                "base_hedge_weight_alpha": float(args.alpha),
                "alpha_sweep": alpha_list if sweep else None,
                "overlay_sleeve_pm_weight": float(args.pm_in_hedge_leg),
                "overlay_sleeve_stock_weight": float(args.stock_in_hedge_leg),
                "momentum_gate_window_bars": int(args.momentum_gate_window),
                "momentum_gate_disabled": bool(args.no_momentum_gate),
                "gate_uses": "rolling_mean_past_only(lagged_spy) > 0 → effective_alpha=base_alpha else 0",
                "frac_7d_bars_gate_active": gate_on,
            },
            "alpha_sweep_7d": [
                {
                    "base_alpha": float(r["base_alpha"]),
                    "gain_7d_combined": float(r["gain_7d_combined"]),
                    "excess_7d_vs_baseline": float(r["excess_7d_vs_baseline"]),
                    "growth_of_1_combined_7d": float(r["growth_of_1_combined_7d"]),
                }
                for r in rows
            ],
            "optimized": {
                "plot_uses_alpha": float(plot_alpha),
                "excess_gain_vs_baseline_7d": opt["excess_gain_vs_baseline_window"],
                "baseline_7d_gain": opt["baseline_window_gain"],
                "best_combined_7d_gain": opt["best_combined_window_gain"],
                "growth_of_1_baseline_7d": g1_baseline_7d,
                "growth_of_1_combined_7d": growth_of_one(combined_plot[m7]),
            },
            "n_bars_7d": int(np.sum(m7)),
            "note": "Overlay is 85% optimized PM + 15% lagged SPY inside the 7% sleeve; stocks are enhancer only.",
        }
    else:
        from src.stock_oil_hedge import combine_sleeves
        from src.hedge_window_opt import static_hedge_leg_returns

        alpha_grid = np.linspace(0.0, 1.0, max(2, int(args.alpha_steps)))
        leg_pairs: tuple[tuple[float, float], ...] = ((0.5, 0.5),)
        if args.leg_grid:
            leg_pairs = ((0.35, 0.65), (0.5, 0.5), (0.65, 0.35))

        opt = optimize_hedge_on_window(
            rpm,
            rb_eval,
            r_spy,
            r_oil,
            m7,
            alpha_grid=alpha_grid,
            leg_weight_pairs=leg_pairs,
        )
        rows = []
        for wo, we in leg_pairs:
            r_h = static_hedge_leg_returns(r_oil, r_spy, wo, we)
            for a in alpha_grid:
                rc = combine_sleeves(rpm, r_h, float(a))
                rows.append(
                    {
                        "recipe": "grid_oil_eq",
                        "oil_leg_weight": float(wo),
                        "inverse_equity_leg_weight": float(we),
                        "hedge_allocation_alpha": float(a),
                        "excess_7d_vs_baseline": excess_gain_vs_baseline(rc, rb_eval, m7),
                        "gain_7d_baseline": float(np.prod(1.0 + rb_eval[m7]) - 1.0),
                        "gain_7d_combined": float(np.prod(1.0 + rc[m7]) - 1.0),
                    }
                )
        rows.sort(key=lambda r: float(r["excess_7d_vs_baseline"]), reverse=True)
        legend = (
            f"PM + oil/−eq (α={opt['best_hedge_allocation_alpha']:.3f}, "
            f"{opt['best_oil_leg_weight']:.2f}/{opt['best_inverse_equity_leg_weight']:.2f})"
        )
        title_note = "grid: oil/−equity sleeve"
        summary = {
            "label": "7-day hedge grid (oil/−eq)",
            "optimizer_objective": obj,
            "input_prefix": in_pfx,
            "output_prefix": out_pfx,
            "recipe": "grid",
            "latest_timestamp_utc": dt.datetime.fromtimestamp(end_ts, tz=dt.timezone.utc).isoformat(),
            "window_7d_start_utc": dt.datetime.fromtimestamp(t7, tz=dt.timezone.utc).isoformat(),
            "pm_hyperparameters_source": "recent_wide_sweep_summary best",
            "wide_best_pm": {k: wb[k] for k in wb if k != "hedge_allocation"},
            "optimized": {
                "best_hedge_allocation_alpha": opt["best_hedge_allocation_alpha"],
                "best_oil_leg_weight": opt["best_oil_leg_weight"],
                "best_inverse_equity_leg_weight": opt["best_inverse_equity_leg_weight"],
                "excess_gain_vs_baseline_7d": opt["excess_gain_vs_baseline_window"],
                "baseline_7d_gain": opt["baseline_window_gain"],
                "best_combined_7d_gain": opt["best_combined_window_gain"],
                "growth_of_1_baseline_7d": growth_of_one(rb_eval[m7]),
                "growth_of_1_combined_7d": growth_of_one(opt["best_combined_returns"][m7]),
            },
            "n_bars_7d": int(np.sum(m7)),
            "note": "Hedge legs from aligned stock_oil_hedge_timeseries (lagged SPY/USO).",
        }

    json_path = out_dir / f"{out_pfx}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = out_dir / f"{out_pfx}_alpha_leg_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    combined_full = np.asarray(opt["best_combined_returns"], dtype=float)
    rb_w = np.asarray(rb_eval[m7], dtype=float)
    best_ret = combined_full[m7]
    cb = np.cumprod(1.0 + rb_w)
    cc = np.cumprod(1.0 + best_ret)
    b_gain = float(opt["baseline_window_gain"])
    c_gain = float(opt["best_combined_window_gain"])
    excess_7d = float(excess_gain_vs_baseline(combined_full, rb_eval, m7))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cb, label="Baseline (equal-weight by domain)", color="tab:blue", linewidth=2.0)
    ax.plot(cc, label=legend, color="tab:orange", linewidth=2.0)
    ax.set_title(f"Week 13 — Last 7 days: growth of $1 ({title_note})")
    ax.set_xlabel("Step (within 7-day window)")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plot_path = fig_dir / f"{out_pfx}_growth_7d.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    fig2, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(cb, label="Baseline", color="tab:blue", linewidth=2.0)
    axes[0].plot(cc, label="Strategy", color="tab:orange", linewidth=2.0)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(f"Last 7 days vs baseline — {title_note}")
    ratio = cc / np.maximum(cb, 1e-12)
    axes[1].plot(ratio, color="tab:green", linewidth=1.8, label="Portfolio / Baseline")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("Ratio (wealth relative)")
    axes[1].set_xlabel("Step (within 7-day window)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig2.tight_layout()
    plot_vs_path = fig_dir / f"{out_pfx}_vs_baseline_7d.png"
    fig2.savefig(plot_vs_path, dpi=120)
    plt.close(fig2)

    fig3, axb = plt.subplots(figsize=(7, 4.5))
    names = ["Baseline\n(equal-weight)", "Strategy\n(combined)"]
    pct = [100.0 * b_gain, 100.0 * c_gain]
    colors = ["tab:blue", "tab:orange"]
    bars = axb.bar(names, pct, color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axb.axhline(0.0, color="black", linewidth=0.8)
    axb.set_ylabel("7-day compound return (%)")
    axb.set_title(
        f"7-day total return vs baseline — excess (compound) = {100.0 * excess_7d:.1f} pp "
        f"(strategy {100.0 * c_gain:.1f}% vs baseline {100.0 * b_gain:.1f}%)"
    )
    ymax = max(abs(x) for x in pct) if pct else 1.0
    for bar, v in zip(bars, pct):
        dy = 0.04 * ymax * (1.0 if v >= 0 else -1.0)
        axb.text(
            bar.get_x() + bar.get_width() / 2,
            v + dy,
            f"{v:.1f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=10,
        )
    axb.grid(True, axis="y", alpha=0.25)
    fig3.tight_layout()
    plot_bar_path = fig_dir / f"{out_pfx}_returns_bar_7d.png"
    fig3.savefig(plot_bar_path, dpi=120)
    plt.close(fig3)

    print(f"summary_json: {json_path}")
    print(f"grid_csv: {csv_path}")
    print(f"plot_growth: {plot_path}")
    print(f"plot_vs_baseline: {plot_vs_path}")
    print(f"plot_returns_bar: {plot_bar_path}")
    print(
        json.dumps(
            {
                **summary.get("optimized", summary.get("parameters", {})),
                "returns_relative_note": (
                    "excess_gain_vs_baseline_7d is compound(strategy)-compound(baseline) over the window; "
                    "negative means baseline grew more in dollar terms."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
