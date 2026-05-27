"""
Scaling of TUR/TSL bounds with system size in the hierarchical model.

Question
--------
Does the >= 10^5 looseness of TUR/TSL in the hierarchical model persist
as we grow the number of genes (N), proteins (M), and raw molecules
(L)? Does the survival region shrink, expand, or move?

Method
------
For a grid of (N, M, L), we
  1) run a small Optuna re-tuning so each size has a sensible operating
     point (otherwise small N already broke and large N would all break),
  2) run an ensemble of replicates at that tuned point,
  3) compute TUR/TSL/MEPP exactly as in bounds.py, and
  4) plot the bounds and their slack vs system size.

We also benchmark the minimal replicator at varying population scale to
separate "more particles" from "more reaction channels".
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from tune import TunableHierAutocat, run_tunable
from bounds import (run_ensemble, compute_bounds, hellinger)
from minimal_replicator import run_minimal_ensemble, tur_product, tsl_quantities


# ---------------------------------------------------------------------------
# 1) HIERARCHICAL MODEL: size scaling
# ---------------------------------------------------------------------------
def tune_for_size(N, M, L, n_trials=25, T=400, dt=0.02):
    """Small Optuna re-tune around the previously-found survival region
    so the (N, M, L) we ask about actually survives."""
    def obj(trial):
        kw = dict(
            N=N, M=M, L=L, seed=3,
            beta       = trial.suggest_float("beta", 0.8, 2.5),
            gamma      = trial.suggest_float("gamma", 0.05, 0.5),
            eps        = trial.suggest_float("eps", 1e-4, 1e-2, log=True),
            coef_bias  = trial.suggest_float("coef_bias", 0.30, 0.70),
            prot_decay = trial.suggest_float("prot_decay", 0.8, 2.5),
            diff_scale = trial.suggest_float("diff_scale", 0.5, 2.5),
            th_scale   = trial.suggest_float("th_scale", 0.1, 0.7),
            gene_decay = trial.suggest_float("gene_decay", 5e-3, 0.1, log=True),
        )
        try:
            _, traj = run_tunable(T=T, dt=dt, **kw)
        except Exception:
            return -1e6
        from tune import score_run
        s, _ = score_run(traj)
        return s

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    if study.best_value < -100:
        return None
    return study.best_params, study.best_value


def measure_at_size(N, M, L, n_rep=10, T=800, dt=0.02, n_trials=25):
    """Returns dict with size, TUR, TSL, MEPP, status."""
    print(f"  tuning (N={N}, M={M}, L={L}) ...", end=" ", flush=True)
    tuned = tune_for_size(N, M, L, n_trials=n_trials, T=400, dt=dt)
    if tuned is None:
        print("FAILED (no surviving config)")
        return dict(N=N, M=M, L=L, status="no-survival",
                    TUR_g=np.nan, TUR_e=np.nan, TSL=np.nan,
                    MEPP=np.nan, sigma=np.nan, survival_score=np.nan,
                    growth=np.nan, evol=np.nan, n_params=N*M*M + M)
    params, score = tuned
    print(f"score={score:.2f}")

    runs = run_ensemble(n_rep=n_rep, T=T, dt=dt,
                        N=N, M=M, L=L, seed=3, **params)
    out = compute_bounds(runs, window=200)
    return dict(N=N, M=M, L=L, status="ok",
                TUR_g=float(np.nanmedian(out["TUR_growth"])),
                TUR_e=float(np.nanmedian(out["TUR_evol"])),
                TSL=float(np.nanmedian(out["TSL_ratio"])),
                MEPP=float(np.nanmean(out["MEPP_sat"])),
                sigma=float(np.nanmean(out["sigma_mean"])),
                survival_score=score,
                growth=float(np.nanmean(np.abs(out["growth_rate"]))),
                evol=float(np.nanmean(np.abs(out["evol_rate"]))),
                n_params=N * M * M + M)   # |W| + |th|


def hier_size_scan(sizes, **kw):
    rows = []
    for (N, M, L) in sizes:
        rows.append(measure_at_size(N, M, L, **kw))
    return rows


# ---------------------------------------------------------------------------
# 2) MINIMAL REPLICATOR: scale of particle population
# ---------------------------------------------------------------------------
def minimal_size_scan(X0_values, *, T=100, dt=0.05, n_rep=300,
                       rev=0.20):
    """Vary initial population X0 (and proportionally A0) to see how the
    TUR product changes with system scale in the *minimal* model."""
    rows = dict(X0=[], A0=[], TUR=[], TSL=[], growth=[], sigma=[],
                ceil=[], obs=[])
    for X0 in X0_values:
        A0 = X0 / 4.0      # keep substrate/protein ratio
        Js, Sigs, X = run_minimal_ensemble(
            n_rep=n_rep, T=T, dt=dt,
            k_bind=0.10, k_unbind=0.10 * rev,
            k_deg=0.15, dmu_deg=2.0, A0=A0, X0=int(X0),
            eps=1e-3, seed=42)
        t = tur_product(Js, Sigs)
        s = tsl_quantities(X, Sigs, dt)
        rows["X0"].append(int(X0));        rows["A0"].append(A0)
        rows["TUR"].append(t);             rows["TSL"].append(s["tsl_ratio"])
        rows["growth"].append(Js.mean()/(T*dt))
        rows["sigma"].append(Sigs.mean()/(T*dt))
        rows["ceil"].append(s["ceil"]);    rows["obs"].append(s["obs_speed"])
        print(f"  X0={X0:>4}  A0={A0:.1f}  TUR={t:.3g}"
              f"  TSL={s['tsl_ratio']:.3g}  σ={Sigs.mean()/(T*dt):.3g}")
    for k in rows:
        rows[k] = np.array(rows[k])
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_hier_scaling(rows, fname="/home/claude/scaling_hier.png"):
    fig = plt.figure(figsize=(13.5, 9), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)

    n_par = np.array([r["n_params"] for r in rows])
    sizes = [f"({r['N']},{r['M']},{r['L']})" for r in rows]

    def plot_one(ax, key, ylog=True, title="", ylabel="",
                 bound=None, bound_label=""):
        vals = np.array([r[key] for r in rows], dtype=float)
        ok = np.array([r["status"] == "ok" for r in rows])
        ax.plot(n_par[ok], vals[ok], "o-", color="#2c3e50", lw=1.2)
        ax.scatter(n_par[~ok], np.ones((~ok).sum()) * np.nan,
                   marker="x", color="#c0392b", s=40, label="failed")
        if bound is not None:
            ax.axhline(bound, color="grey", ls="--", lw=1.0,
                       label=bound_label)
        for i, s in enumerate(sizes):
            if ok[i]:
                ax.annotate(s, (n_par[i], vals[i]), fontsize=7, alpha=0.6,
                            xytext=(3, 3), textcoords="offset points")
        if ylog: ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("# free parameters in W  ($N\\cdot M^2$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, ls=":", alpha=0.5, which="both")
        if bound is not None:
            ax.legend(fontsize=8)

    plot_one(fig.add_subplot(gs[0, 0]), "TUR_g",
             title="TUR product (growth current)", ylabel="TUR product",
             bound=2.0, bound_label="bound = 2")
    plot_one(fig.add_subplot(gs[0, 1]), "TUR_e",
             title="TUR product (evolution current)", ylabel="TUR product",
             bound=2.0, bound_label="bound = 2")
    plot_one(fig.add_subplot(gs[0, 2]), "TSL",
             title="TSL ratio  $\\tau \\cdot 2\\langle\\Sigma\\rangle\\langle A\\rangle / L^2$",
             ylabel="TSL ratio",
             bound=1.0, bound_label="bound = 1")
    plot_one(fig.add_subplot(gs[1, 0]), "MEPP",
             ylog=False,
             title="MEPP saturation  $\\langle\\sigma\\rangle / \\sigma_\\mathrm{max}$",
             ylabel="MEPP saturation")
    plot_one(fig.add_subplot(gs[1, 1]), "sigma",
             title="Mean entropy production rate  $\\langle\\sigma\\rangle$",
             ylabel="$\\sigma$")
    plot_one(fig.add_subplot(gs[1, 2]), "survival_score",
             ylog=False,
             title="Survival score from Optuna re-tuning",
             ylabel="best score")

    fig.suptitle("Hierarchical model: how the bounds scale with $(N,M,L)$",
                 fontsize=13, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


def plot_minimal_scaling(rows, fname="/home/claude/scaling_minimal.png"):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    axs[0].plot(rows["X0"], rows["TUR"], "o-", color="#2c3e50", lw=1.3)
    axs[0].axhline(2.0, color="grey", ls="--", lw=1.0, label="bound = 2")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("initial population $X_0$")
    axs[0].set_ylabel("TUR product")
    axs[0].set_title("TUR product vs population size\n(minimal replicator)")
    axs[0].legend(fontsize=9)
    axs[0].grid(True, ls=":", alpha=0.5, which="both")

    axs[1].plot(rows["X0"], rows["TSL"], "o-", color="#c0392b", lw=1.3)
    axs[1].axhline(1.0, color="grey", ls="--", lw=1.0, label="bound = 1")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel("$X_0$"); axs[1].set_ylabel("TSL ratio")
    axs[1].set_title("TSL ratio vs population size")
    axs[1].legend(fontsize=9)
    axs[1].grid(True, ls=":", alpha=0.5, which="both")

    axs[2].plot(rows["sigma"], rows["growth"], "o-", color="#8e44ad", lw=1.3)
    for i, x in enumerate(rows["X0"]):
        axs[2].annotate(f"X={x}", (rows["sigma"][i], rows["growth"][i]),
                         fontsize=7, alpha=0.7)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("$\\langle\\sigma\\rangle$")
    axs[2].set_ylabel("growth rate")
    axs[2].set_title("Growth vs EP across scales")
    axs[2].grid(True, ls=":", alpha=0.5)

    fig.suptitle("Minimal replicator: scaling with particle number",
                 fontsize=12, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) hierarchical model: scan (N, M, L)
    print("=== Hierarchical model size scaling ===")
    # we keep L fixed at 3 (raw molecules) since that's the bath dimension,
    # and vary (N, M) which control the polymer- and gene-space
    sizes = [
        (4,  3, 3),
        (8,  4, 3),
        (12, 4, 3),
        (16, 6, 3),
        (24, 6, 3),
        (32, 8, 3),
    ]
    rows = hier_size_scan(sizes, n_rep=8, T=600, dt=0.02, n_trials=25)
    plot_hier_scaling(rows)
    print("\nSummary:")
    for r in rows:
        print(f"  (N={r['N']:>2}, M={r['M']}, L={r['L']})"
              f"  status={r['status']:<11}"
              f"  TUR_g={r['TUR_g']: .2e}"
              f"  TSL={r['TSL']: .2e}"
              f"  MEPP={r['MEPP']: .3f}"
              f"  σ={r['sigma']: .2e}")

    # 2) minimal model: scan particle scale
    print("\n=== Minimal replicator population scaling ===")
    X0s = np.array([5, 10, 20, 40, 80, 160])
    mrows = minimal_size_scan(X0s, T=100, dt=0.05, n_rep=400, rev=0.20)
    plot_minimal_scaling(mrows)
