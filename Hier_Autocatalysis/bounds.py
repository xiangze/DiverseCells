"""
Stochastic-thermodynamic bounds applied to the hierarchical autocatalytic
model:

  (1) Thermodynamic Uncertainty Relation (TUR)
        Var(J_tau) / <J_tau>^2  *  Sigma_tau  >=  2
      Applied to two currents:
        J_growth(tau) = log(<Wp>(tau)) - log(<Wp>(0))      (cell growth)
        J_evol(tau)   = -[S_gene(tau) - S_gene(0)]         (evolution = order)

  (2) Thermodynamic Speed Limit (TSL)   --   Shiraishi-Funo-Saito form
        tau  >=  L(p_0, p_tau)^2  /  (2 * <Sigma>_tau * <A>_tau)
      where
        p_t      = Wp/sum(Wp)        gene-frequency distribution
        L        = Hellinger distance between p_0 and p_tau
        <A>_tau  = mean dynamical activity over [0, tau]
        <Sigma>  = mean entropy production rate over [0, tau]
      => upper bound on the speed |dp/dt|.

  (3) MEPP saturation
        For each window we measure  sigma / sigma_max   where sigma_max is
        the largest sigma encountered along an ensemble of replicates with
        slightly perturbed initial conditions. Closer to 1 means the
        chosen trajectory is selecting the maximum-EP branch (MEPP).

Outputs: time-resolved TUR product, TSL ratio, MEPP saturation, and
two summary plots that show how the bounds constrain growth and
evolution rates.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tune import TunableHierAutocat, run_tunable

# Optuna best config from previous step --------------------------------
BEST = dict(
    N=8, M=4, L=3, seed=3,
    beta=1.451, gamma=0.1265, eps=2.79e-3,
    coef_bias=0.5982, prot_decay=1.094,
    diff_scale=1.546, th_scale=0.3856, gene_decay=0.01998,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hellinger(p, q):
    """Hellinger distance between two discrete probability vectors. Robust
    to NaN by replacing with 0 and renormalising."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
    q = np.where(np.isfinite(q) & (q > 0), q, 0.0)
    sp = p.sum(); sq = q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = p / sp; q = q / sq
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def total_variation(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    return float(0.5 * np.abs(p - q).sum())


# ---------------------------------------------------------------------------
# Run a single trajectory and record everything we need for the bounds
# ---------------------------------------------------------------------------
def run_full(T=1500, dt=0.02, **kw):
    m = TunableHierAutocat(**kw)
    keys = ["sigma", "sigma_react", "sigma_diff", "S_gene", "growth",
            "Wp", "pp", "a"]
    rec = {k: [] for k in keys}
    for _ in range(T):
        out = m.step(dt=dt)
        for k in keys:
            rec[k].append(out[k])
    for k in keys:
        rec[k] = np.array(rec[k])
    rec["dt"] = dt
    return m, rec


# ---------------------------------------------------------------------------
# Ensemble run for TUR variance estimation and MEPP saturation
# ---------------------------------------------------------------------------
def run_ensemble(n_rep=20, T=1500, dt=0.02, jitter=0.02, **kw):
    """Run several trajectories with slightly perturbed seeds and initial
    conditions, so we can estimate Var(J_tau) for the TUR and identify
    the maximum-EP branch for MEPP saturation."""
    runs = []
    for rep in range(n_rep):
        kw2 = dict(kw)
        kw2["seed"] = kw.get("seed", 0) + rep
        _, rec = run_full(T=T, dt=dt, **kw2)
        # add a tiny Gaussian jitter to recorded quantities to emulate
        # initial-condition variability beyond what `seed` already changes
        rec_j = {k: (v + jitter * np.std(v) * np.random.standard_normal(v.shape)
                     if v.dtype.kind == "f" else v)
                 for k, v in rec.items() if k != "dt"}
        rec_j["dt"] = dt
        runs.append(rec_j)
    return runs


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
def compute_bounds(runs, window=200):
    """Compute the TUR product, TSL ratio, and MEPP saturation in sliding
    windows of length `window`.

    Returns
    -------
    out : dict of arrays indexed by window-end time t
        keys: 't',
              'TUR_growth', 'TUR_evol',
              'TSL_ratio', 'TSL_max_speed', 'observed_speed',
              'MEPP_sat',
              'growth_rate', 'evol_rate', 'sigma_mean'
    """
    n_rep = len(runs)
    T = runs[0]["sigma"].shape[0]
    dt = runs[0]["dt"]
    starts = np.arange(0, T - window, window // 4)   # 75% overlap

    out = {k: [] for k in
           ["t", "TUR_growth", "TUR_evol", "TSL_ratio",
            "TSL_max_speed", "observed_speed", "MEPP_sat",
            "growth_rate", "evol_rate", "sigma_mean"]}

    for s in starts:
        e = s + window
        out["t"].append(e)

        # ----- per-replicate window quantities -----
        Sigma_w   = np.zeros(n_rep)   # cumulative EP in window
        J_growth  = np.zeros(n_rep)   # log-mass change
        J_evol    = np.zeros(n_rep)   # -ΔS_gene
        A_w       = np.zeros(n_rep)   # dynamical activity
        L_w       = np.zeros(n_rep)   # Hellinger distance over window

        for r, rec in enumerate(runs):
            sig = rec["sigma"][s:e]
            if np.any(~np.isfinite(sig)):
                Sigma_w[r] = np.nan; J_growth[r] = np.nan
                J_evol[r] = np.nan; A_w[r] = np.nan; L_w[r] = np.nan
                continue
            Sigma_w[r] = float(np.sum(sig) * dt)

            Wp_sum = rec["Wp"][s:e].sum(axis=1)
            Wp_sum = np.where(np.isfinite(Wp_sum) & (Wp_sum > 0),
                              Wp_sum, np.nan)
            logW = np.log(Wp_sum + 1e-12)
            J_growth[r] = logW[-1] - logW[0]

            Sg = rec["S_gene"][s:e]
            J_evol[r] = -(Sg[-1] - Sg[0])

            p0 = rec["Wp"][s]
            pt = rec["Wp"][e - 1]
            L_w[r] = hellinger(p0, pt)

            # dynamical activity: time integral of total |dp/dt|
            Wp_seq = rec["Wp"][s:e]
            tot = Wp_seq.sum(axis=1, keepdims=True)
            tot = np.where(np.isfinite(tot) & (tot > 0), tot, np.nan)
            p_seq = Wp_seq / tot
            dp = np.diff(p_seq, axis=0)
            A_w[r] = float(np.nansum(np.abs(dp))) / (window * dt)

        # drop NaN replicates from window statistics
        valid = np.isfinite(Sigma_w) & np.isfinite(J_growth) \
                & np.isfinite(L_w) & np.isfinite(A_w)
        if valid.sum() < 2:
            for k in ["TUR_growth", "TUR_evol", "TSL_ratio",
                      "TSL_max_speed", "observed_speed", "MEPP_sat",
                      "growth_rate", "evol_rate", "sigma_mean"]:
                out[k].append(np.nan)
            continue
        Sigma_w  = Sigma_w[valid]
        J_growth = J_growth[valid]
        J_evol   = J_evol[valid]
        A_w      = A_w[valid]
        L_w      = L_w[valid]

        # ----- TUR products -----
        # Var(J)/<J>^2 * Sigma  should be >= 2 (lower bound on dispersion)
        mu_g = J_growth.mean()
        var_g = J_growth.var()
        sig_mean = Sigma_w.mean()
        if abs(mu_g) > 1e-9 and sig_mean > 0:
            TUR_g = (var_g / mu_g ** 2) * sig_mean
        else:
            TUR_g = np.nan

        mu_e = J_evol.mean()
        var_e = J_evol.var()
        if abs(mu_e) > 1e-9 and sig_mean > 0:
            TUR_e = (var_e / mu_e ** 2) * sig_mean
        else:
            TUR_e = np.nan

        # ----- TSL -----
        # τ >= L^2 / (2 <Σ> <A>)
        # =>  max speed L/τ <= sqrt(2 <Σ> <A>) / sqrt(τ)
        # We compute the ratio  τ_observed * 2<Σ><A> / L^2  ;  >=1 means
        # the speed limit is satisfied (and how loose it is = how big).
        L_mean = L_w.mean()
        A_mean = A_w.mean()
        tau    = window * dt
        if L_mean > 1e-9 and sig_mean > 0 and A_mean > 0:
            tsl_ratio = (tau * 2.0 * sig_mean * A_mean) / (L_mean ** 2)
            tsl_max_speed = np.sqrt(2.0 * sig_mean * A_mean / tau)
        else:
            tsl_ratio = np.nan
            tsl_max_speed = np.nan
        observed_speed = L_mean / tau

        # ----- MEPP saturation -----
        sigma_means_each = []
        for rec in runs:
            sig = rec["sigma"][s:e]
            if np.all(np.isfinite(sig)):
                sigma_means_each.append(float(sig.mean()))
        sigma_means_each = np.array(sigma_means_each)
        if sigma_means_each.size > 0 and sigma_means_each.max() > 0:
            MEPP_sat = sigma_means_each.mean() / sigma_means_each.max()
        else:
            MEPP_sat = np.nan

        out["TUR_growth"].append(TUR_g)
        out["TUR_evol"].append(TUR_e)
        out["TSL_ratio"].append(tsl_ratio)
        out["TSL_max_speed"].append(tsl_max_speed)
        out["observed_speed"].append(observed_speed)
        out["MEPP_sat"].append(MEPP_sat)
        out["growth_rate"].append(mu_g / tau)
        out["evol_rate"].append(mu_e / tau)
        out["sigma_mean"].append(sig_mean / tau)   # average rate

    for k in out:
        out[k] = np.array(out[k])
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_tur(out, fname="/home/claude/tur_tsl_bounds.png"):
    fig = plt.figure(figsize=(13.5, 9), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    t = out["t"]

    # ---- TUR products vs the universal bound 2 ----
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(2.0, color="grey", ls="--", lw=1.0,
               label=r"TUR bound = 2")
    ax.plot(t, out["TUR_growth"], color="#2c3e50", lw=1.2,
            label=r"$\mathrm{Var}(J_g)/\langle J_g\rangle^2 \cdot \Sigma$ : growth")
    ax.plot(t, out["TUR_evol"], color="#16a085", lw=1.2,
            label=r"$\mathrm{Var}(J_e)/\langle J_e\rangle^2 \cdot \Sigma$ : evolution")
    ax.set_yscale("symlog")
    ax.set_xlabel("step (window end)")
    ax.set_ylabel("TUR product")
    ax.set_title("Thermodynamic uncertainty relation\n"
                 r"Var($J$)/$\langle J\rangle^2 \cdot \Sigma_\tau \ \geq\ 2$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", alpha=0.5)

    # ---- TSL ratio: τ * 2<Σ><A> / L^2 >= 1 ----
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(1.0, color="grey", ls="--", lw=1.0,
               label="TSL bound = 1")
    ax.plot(t, out["TSL_ratio"], color="#c0392b", lw=1.2)
    ax.set_yscale("symlog")
    ax.set_xlabel("step (window end)")
    ax.set_ylabel(r"$\tau \cdot 2\langle\Sigma\rangle\langle A\rangle\ /\ \mathcal{L}^2$")
    ax.set_title("Thermodynamic speed limit\n"
                 r"$\tau \geq \mathcal{L}^2 / (2 \langle\Sigma\rangle \langle A\rangle)$")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # ---- growth rate vs TSL-allowed max speed ----
    ax = fig.add_subplot(gs[1, 0])
    # protect against log of zero
    obs = np.maximum(np.abs(out["observed_speed"]), 1e-12)
    ceil = np.maximum(out["TSL_max_speed"], 1e-12)
    ax.semilogy(t, obs, color="#2c3e50", lw=1.2,
                label="observed |dp/dt|")
    ax.semilogy(t, ceil, color="#c0392b", lw=1.0, ls="--",
                label=r"TSL ceiling $\sqrt{2\langle\Sigma\rangle\langle A\rangle/\tau}$")
    # the gap between them = how loose the TSL is
    ax.fill_between(t, obs, ceil, color="#c0392b", alpha=0.08,
                    label="slack")
    ax.set_xlabel("step")
    ax.set_ylabel("rate (log scale)")
    ax.set_title("Evolution speed vs TSL ceiling (huge slack)")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5, which="both")

    # ---- MEPP saturation ----
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, out["MEPP_sat"], color="#8e44ad", lw=1.2)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8, label="MEPP branch")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\langle\sigma\rangle / \sigma_{\max}$")
    ax.set_title("MEPP saturation across replicates")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    # ---- growth & evolution rates as time series, with σ overlaid ----
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, np.abs(out["growth_rate"]), color="#2c3e50", lw=1.2,
            label="|growth|")
    ax.plot(t, np.abs(out["evol_rate"]), color="#16a085", lw=1.2,
            label="|evolution|  $|dS_\\mathrm{gene}/dt|$")
    ax.set_xlabel("step"); ax.set_ylabel("rate")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("Rates of growth and evolution")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(t, out["sigma_mean"], color="#c0392b", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\langle\sigma\rangle$  (mean EP rate)")
    ax.set_yscale("symlog")
    ax.set_title("Mean entropy-production rate per window")
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle(
        "Stochastic-thermodynamic bounds on growth and evolution\n"
        "TUR + TSL + MEPP saturation",
        fontsize=13, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


def plot_bound_scan(scan, fname="/home/claude/bounds_vs_sigma.png"):
    """Companion figure: show how the TSL ceiling shifts as we
    artificially clamp entropy production by varying eps / gamma."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)

    sigs = scan["sigma_mean"]
    axs[0].plot(sigs, scan["max_growth"], "o-", color="#2c3e50",
                label="achieved |growth|")
    axs[0].plot(sigs, scan["tsl_ceiling"], "o--", color="#c0392b",
                label="TSL ceiling")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel(r"mean $\langle\sigma\rangle$")
    axs[0].set_ylabel("rate")
    axs[0].set_title("Growth rate scales with entropy production")
    axs[0].legend(fontsize=9); axs[0].grid(True, ls=":", alpha=0.5)

    axs[1].plot(sigs, scan["max_evol"], "o-", color="#16a085",
                label="achieved |evolution|")
    axs[1].plot(sigs, scan["tsl_ceiling"], "o--", color="#c0392b",
                label="TSL ceiling")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set_xlabel(r"mean $\langle\sigma\rangle$")
    axs[1].set_ylabel("rate")
    axs[1].set_title("Evolution rate also scales with EP")
    axs[1].legend(fontsize=9); axs[1].grid(True, ls=":", alpha=0.5)

    fig.suptitle("Speed limits scale with the budget of entropy production",
                 fontsize=12, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
# Sweep: artificially modulate eps and measure how the TSL ceiling and
# the achievable rates change.
# ---------------------------------------------------------------------------
def scan_sigma_budget(n_rep=6, T=1000, dt=0.02, param="coef_bias",
                      values=None):
    """Vary `param` to control the sigma budget. coef_bias and
    diff_scale are the parameters with the largest effect on the
    sustained sigma at steady state.
    """
    if values is None:
        if param == "coef_bias":
            values = np.linspace(0.30, 0.70, 5)
        elif param == "diff_scale":
            values = np.linspace(0.3, 3.0, 5)
        else:
            values = np.logspace(-4, -1, 5)

    rows = dict(sigma_mean=[], max_growth=[], max_evol=[],
                tsl_ceiling=[], param_val=[])
    for v in values:
        kw = dict(BEST); kw[param] = float(v)
        runs = run_ensemble(n_rep=n_rep, T=T, dt=dt, **kw)
        out  = compute_bounds(runs, window=200)
        rows["param_val"].append(float(v))
        rows["sigma_mean"].append(float(np.nanmean(out["sigma_mean"])))
        rows["max_growth"].append(float(np.nanmean(np.abs(out["growth_rate"]))))
        rows["max_evol"].append(float(np.nanmean(np.abs(out["evol_rate"]))))
        rows["tsl_ceiling"].append(float(np.nanmean(out["TSL_max_speed"])))
        print(f"  {param}={v:.3g} :  <σ>={rows['sigma_mean'][-1]: .3e}"
              f"  |growth|={rows['max_growth'][-1]: .3e}"
              f"  |evol|={rows['max_evol'][-1]: .3e}"
              f"  TSL_ceil={rows['tsl_ceiling'][-1]: .3e}")
    for k in rows:
        rows[k] = np.array(rows[k])
    rows["param"] = param
    return rows


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Ensemble run for TUR / TSL / MEPP ===")
    runs = run_ensemble(n_rep=15, T=1500, dt=0.02, jitter=0.0, **BEST)
    out = compute_bounds(runs, window=200)

    n_valid = (~np.isnan(out["TUR_growth"])).sum()
    print(f"valid windows: {n_valid}/{len(out['t'])}")
    print(f"median TUR_growth  = {np.nanmedian(out['TUR_growth']): .3e}"
          "  (should be >= 2)")
    print(f"median TUR_evol    = {np.nanmedian(out['TUR_evol']): .3e}"
          "  (should be >= 2)")
    print(f"median TSL ratio   = {np.nanmedian(out['TSL_ratio']): .3e}"
          "  (should be >= 1)")
    print(f"mean MEPP sat.     = {np.nanmean(out['MEPP_sat']): .3f}"
          "  (1 = on the max-EP branch)")

    plot_tur(out)

    print("\n=== Sweep over coef_bias to see how TSL/TUR scale with σ budget ===")
    scan = scan_sigma_budget(n_rep=5, T=1000, dt=0.02,
                              param="coef_bias",
                              values=np.linspace(0.30, 0.70, 5))
    plot_bound_scan(scan)
