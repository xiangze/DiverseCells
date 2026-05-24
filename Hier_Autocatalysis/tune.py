"""
Parameter search for the hierarchical autocatalytic model.

Provides:
  - score_run(traj):  compose a scalar score from a single trajectory that
                      rewards non-trivial non-equilibrium steady states and
                      penalises extinction or divergence.
  - grid_search(...): exhaustive sweep over a user-supplied grid.
  - optuna_search(...): Bayesian search via Optuna (TPE).

The search wraps `HierAutocat` from hier_v2.py so the model definition
itself is untouched. Extra "tunable" knobs are exposed as constructor-time
multipliers on Coef bias, protein decay, diffusion scale, and threshold
scale, which we patch in via a thin subclass.
"""

from __future__ import annotations

import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler

from hier_v2 import HierAutocat, run as base_run


# ---------------------------------------------------------------------------
# Tunable model: same physics, exposes a few global multipliers so a search
# routine can vary the dynamics without rewriting __init__.
# ---------------------------------------------------------------------------
class TunableHierAutocat(HierAutocat):
    def __init__(self, *, coef_bias=0.35, prot_decay=2.5,
                 diff_scale=1.0, th_scale=0.2, gene_decay=0.05, **kw):
        super().__init__(**kw)
        # rebuild Coef with the requested bias
        r = self.rng
        L, M = self.L, self.M
        self.Coef = r.standard_normal((L + M, M, L + M)) * 0.15
        for l in range(M):
            for j in range(L):
                self.Coef[L + l, l, j] += coef_bias
        self.Coef[L:, :, L:] *= 0.1

        self.d = self.d * diff_scale
        self.th = r.standard_normal(M) * th_scale
        self._prot_decay = prot_decay
        self._coef_bias = coef_bias
        self._gene_decay = gene_decay

    def step(self, dt=0.02):
        # we override only the protein-decay coefficient
        L, M, N = self.L, self.M, self.N
        rng = self.rng

        self.pa = self.fold_activity()
        mod = np.tanh(self.pa)
        y = np.concatenate([self.a, self.pp])
        mass = np.outer(y, y)

        r_full = np.einsum("ilj,l,ij->i", self.Coef, mod, mass)
        r_full = r_full + self.eps * rng.standard_normal(L + M)
        r_a, r_p = r_full[:L], r_full[L:]

        diff = self.d * (self.a_ext - self.a)
        consumption = (r_p.sum() / L) * np.ones(L) * 0.5
        da = self.Kon * r_a + diff - consumption
        self.a = np.clip(self.a + dt * da, 1e-6, 50.0)

        dpp = r_p - self.gamma * self._prot_decay * self.pp
        self.pp = np.clip(self.pp + dt * dpp, 1e-6, 50.0)

        align = np.einsum("lmk,l,m->k", self.W, self.pa, self.pp)
        fitness = np.tanh(align)
        substrate = self.a.sum()
        K = 1.0 + substrate
        total = self.Wp.sum()
        dWp = self.Wp * fitness * (1.0 - total / K) - self._gene_decay * self.Wp
        self.Wp = np.clip(self.Wp + dt * dWp, 1e-9, None)

        self.W += dt * self.eps * 5.0 * rng.standard_normal(self.W.shape)
        self.symbols = self._symbols_from_W()

        fwd = np.maximum(r_full, 0) + 1e-9
        bwd = np.maximum(-r_full, 0) + self.eps + 1e-9
        sigma_react = float(np.sum((fwd - bwd) * np.log(fwd / bwd)))
        sigma_diff = float(np.sum(
            self.d * (self.a_ext - self.a)
            * np.log((self.a_ext + 1e-9) / (self.a + 1e-9))))
        sigma = sigma_react + sigma_diff

        growth_rate = float(np.mean(dWp / (self.Wp + 1e-9)))

        w = self.Wp / self.Wp.sum()
        bins: dict[str, float] = {}
        for s, wk in zip(self.symbols, w):
            bins[s] = bins.get(s, 0.0) + wk
        p = np.array(list(bins.values()))
        S_gene = float(-np.sum(p * np.log(p + 1e-12)))

        return dict(sigma=sigma, growth=growth_rate, S_gene=S_gene,
                    sigma_react=sigma_react, sigma_diff=sigma_diff,
                    a=self.a.copy(), pp=self.pp.copy(),
                    pa=self.pa.copy(), Wp=self.Wp.copy(),
                    n_species=len(bins))


def run_tunable(T=800, dt=0.02, **kw):
    m = TunableHierAutocat(**kw)
    keys = ["sigma", "growth", "S_gene", "sigma_react", "sigma_diff",
            "a", "pp", "pa", "Wp", "n_species"]
    traj = {k: [] for k in keys}
    for _ in range(T):
        out = m.step(dt=dt)
        for k in keys:
            traj[k].append(out[k])
    for k in keys:
        traj[k] = np.array(traj[k])
    return m, traj


# ---------------------------------------------------------------------------
# Scoring: reward non-trivial NESS, penalise extinction and divergence
# ---------------------------------------------------------------------------
def score_run(traj, *, tail=200,
              w_growth=1.0, w_sgene=2.0, w_sigma=0.3, w_mass=0.5,
              extinction_thr=5e-2, divergence_thr=40.0):
    """Return (score, diagnostics_dict).

    Higher score = healthier NESS with diversity and finite dissipation.

    Failure modes (return large negative score):
      * NaN / inf anywhere
      * extinction:  mean(total_Wp) or mean(total_pp) over the tail below
                     `extinction_thr` (raised from 1e-3 so the tuner does
                     not pick "almost-dead" runs)
      * divergence:  any state pegged near the clip ceiling
      * collapsing tail: the second half of the tail has less than 50 % of
                     the first half's biomass (i.e. still on its way to
                     extinction even if currently above threshold)
    """
    diag = {}

    if np.any(~np.isfinite(traj["sigma"])):
        return -1e6, dict(reason="nan", **diag)

    Wp_tail = traj["Wp"][-tail:]
    pp_tail = traj["pp"][-tail:]
    a_tail  = traj["a"][-tail:]

    total_Wp = Wp_tail.sum(axis=1)
    total_pp = pp_tail.sum(axis=1)

    # extinction (strict: meaningful biomass required)
    if total_Wp.mean() < extinction_thr or total_pp.mean() < extinction_thr:
        return (-1e3 + total_Wp.mean() + total_pp.mean(),
                dict(reason="extinction",
                     tot_Wp=float(total_Wp.mean()),
                     tot_pp=float(total_pp.mean())))

    # divergence
    if pp_tail.max() > divergence_thr or a_tail.max() > divergence_thr:
        return -1e3, dict(reason="divergence",
                          pp_max=float(pp_tail.max()),
                          a_max=float(a_tail.max()))

    # collapsing tail: not yet extinct, but heading there
    half = tail // 2
    first_half = total_Wp[:half].mean()
    second_half = total_Wp[half:].mean()
    if second_half < 0.5 * first_half:
        return (-500.0 + second_half,
                dict(reason="collapsing",
                     tot_Wp_first=float(first_half),
                     tot_Wp_second=float(second_half)))

    # -------- healthy NESS criteria -------------------------------------
    growth_tail = traj["growth"][-tail:]
    sigma_tail  = traj["sigma"][-tail:]
    sgene_tail  = traj["S_gene"][-tail:]

    growth_steady = -abs(growth_tail.mean())
    growth_var    = -np.std(growth_tail)

    sgene_mean = sgene_tail.mean()
    sigma_mean = max(sigma_tail.mean(), 1e-6)
    mass_score = math.log(max(total_Wp.mean() * total_pp.mean(), 1e-9))

    diag.update(
        tot_Wp=float(total_Wp.mean()),
        tot_pp=float(total_pp.mean()),
        growth_mean=float(growth_tail.mean()),
        growth_std=float(np.std(growth_tail)),
        sigma_mean=float(sigma_mean),
        S_gene_mean=float(sgene_mean),
        mass_score=float(mass_score),
        reason="ok",
    )

    score = (
        w_growth * growth_steady
        + w_sgene * sgene_mean
        + w_sigma * np.log(sigma_mean)
        + w_mass  * mass_score
        + 0.5 * growth_var
    )
    return float(score), diag


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def grid_search(grid: dict, *, T=600, dt=0.02, fixed=None, verbose=True):
    """Exhaustive sweep over the cartesian product of `grid`.

    Parameters
    ----------
    grid : dict[str, list]
        Each key is a TunableHierAutocat keyword; each value is the list of
        candidates to try.
    fixed : dict | None
        Constant keyword arguments shared by every run.
    """
    fixed = dict(fixed or {})
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows = []
    for combo in combos:
        kw = dict(zip(keys, combo))
        kw_full = {**fixed, **kw}
        try:
            _, traj = run_tunable(T=T, dt=dt, **kw_full)
            score, diag = score_run(traj)
        except Exception as e:
            score, diag = -1e6, dict(reason=f"exception:{type(e).__name__}")
        row = {**kw, "score": score, **diag}
        rows.append(row)
        if verbose:
            tags = ", ".join(f"{k}={v}" for k, v in kw.items())
            print(f"  [{tags}]  score={score: .3f}  ({diag.get('reason')})")

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Optuna search (TPE)
# ---------------------------------------------------------------------------
def optuna_search(n_trials=60, T=600, dt=0.02, fixed=None, seed=0,
                  verbose=False):
    """Bayesian search over the continuous-ish parameters."""
    fixed = dict(fixed or {})

    def objective(trial: optuna.Trial):
        kw = dict(
            beta       = trial.suggest_float("beta", 0.5, 4.0),
            gamma      = trial.suggest_float("gamma", 0.1, 1.5),
            eps        = trial.suggest_float("eps", 1e-5, 1e-1, log=True),
            coef_bias  = trial.suggest_float("coef_bias", 0.05, 0.6),
            prot_decay = trial.suggest_float("prot_decay", 1.0, 5.0),
            diff_scale = trial.suggest_float("diff_scale", 0.1, 3.0),
            th_scale   = trial.suggest_float("th_scale", 0.0, 1.0),
            gene_decay = trial.suggest_float("gene_decay", 1e-3, 0.2,
                                              log=True),
        )
        kw_full = {**fixed, **kw}
        try:
            _, traj = run_tunable(T=T, dt=dt, **kw_full)
            score, diag = score_run(traj)
        except Exception:
            return -1e6
        # log diagnostics on the trial
        for k, v in diag.items():
            if isinstance(v, (int, float)):
                trial.set_user_attr(k, v)
        trial.set_user_attr("reason", diag.get("reason", "ok"))
        return score

    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_grid(df: pd.DataFrame, xkey: str, ykey: str,
              fname="/home/claude/grid_heatmap.png"):
    """Two-panel heatmap so that small score differences inside the
    survival region are not crushed by the large penalty values."""
    pivot = df.pivot_table(index=ykey, columns=xkey, values="score",
                           aggfunc="mean")
    # category map of failure modes for the second panel
    reason_pivot = df.pivot_table(index=ykey, columns=xkey, values="reason",
                                  aggfunc=lambda s: s.iloc[0])
    # mask: positive scores ("ok") on the left, raw category on the right
    ok_mask = pivot > -100
    ok_vals = pivot.where(ok_mask)

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

    ax = axs[0]
    im = ax.imshow(ok_vals.values, origin="lower", aspect="auto",
                   cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:g}" for c in pivot.index])
    ax.set_xlabel(xkey); ax.set_ylabel(ykey)
    fig.colorbar(im, ax=ax, label="score (survival cells only)")
    ax.set_title("Score in healthy NESS cells")
    # annotate scores in healthy cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if v > -100:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=9)

    ax = axs[1]
    cat_colors = {"ok": 2, "extinction": 0, "divergence": 1,
                  "collapsing": 3, "nan": 4}
    cat_grid = np.array([[cat_colors.get(reason_pivot.values[i, j], 4)
                          for j in range(len(pivot.columns))]
                         for i in range(len(pivot.index))])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#c0392b", "#e67e22", "#16a085", "#9b59b6",
                           "#7f8c8d"])
    ax.imshow(cat_grid, origin="lower", aspect="auto", cmap=cmap,
              vmin=0, vmax=4)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:g}" for c in pivot.index])
    ax.set_xlabel(xkey); ax.set_ylabel(ykey)
    ax.set_title("Outcome category")
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color="#16a085", label="ok"),
               Patch(color="#c0392b", label="extinction"),
               Patch(color="#e67e22", label="divergence"),
               Patch(color="#9b59b6", label="collapsing"),
               Patch(color="#7f8c8d", label="other")]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            r = reason_pivot.values[i, j]
            if isinstance(r, str):
                ax.text(j, i, r[:4], ha="center", va="center",
                        color="white", fontsize=8)

    fig.suptitle(
        f"Grid search:  ({xkey}, {ykey})  — survival score & outcome",
        fontsize=12, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


def plot_optuna(study: optuna.Study,
                fname="/home/claude/optuna_summary.png"):
    """Convergence + parameter-importance overview."""
    trials = [t for t in study.trials if t.value is not None]
    values = np.array([t.value for t in trials])
    running_best = np.maximum.accumulate(values)
    reasons = [t.user_attrs.get("reason", "?") for t in trials]
    colours = {"ok": "#16a085", "extinction": "#c0392b",
               "divergence": "#e67e22", "nan": "#7f8c8d"}
    cs = [colours.get(r, "#7f8c8d") for r in reasons]

    fig = plt.figure(figsize=(13, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(range(len(values)), values, c=cs, s=22, alpha=0.85)
    ax.plot(running_best, color="black", lw=1.0, label="running best")
    ax.set_xlabel("trial"); ax.set_ylabel("score")
    ax.set_title("Optuna trials (colour = outcome)")
    ax.set_ylim(max(-50, values.min() - 5), values.max() + 1)
    ax.legend(fontsize=8)

    # outcome breakdown
    ax = fig.add_subplot(gs[0, 1])
    from collections import Counter
    cnt = Counter(reasons)
    labs, vals = zip(*sorted(cnt.items(), key=lambda x: -x[1]))
    ax.bar(labs, vals, color=[colours.get(l, "#7f8c8d") for l in labs])
    ax.set_title("Trial outcomes")
    ax.set_ylabel("# trials")

    # parameter importance (only over completed numeric trials)
    ax = fig.add_subplot(gs[0, 2])
    try:
        imp = optuna.importance.get_param_importances(study)
        ax.barh(list(imp.keys()), list(imp.values()), color="#2c3e50")
        ax.set_title("Parameter importance")
        ax.invert_yaxis()
    except Exception as e:
        ax.text(0.5, 0.5, f"importance failed:\n{e}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8)

    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fixed = dict(N=8, M=4, L=3, seed=3)

    # --- 1) small grid search -----------------------------------------
    # Fix coef_bias and prot_decay near the survival regime found earlier,
    # then sweep beta and gamma to see the survival/failure boundary.
    print("\n=== Grid search ===")
    grid_fixed = dict(coef_bias=0.6, prot_decay=1.1, diff_scale=1.5,
                      th_scale=0.4, gene_decay=0.02, eps=3e-3)
    grid = dict(
        beta  = [0.8, 1.5, 2.5, 3.5],
        gamma = [0.1, 0.2, 0.4, 0.8],
    )
    df = grid_search(grid, T=600, dt=0.02,
                     fixed={**fixed, **grid_fixed}, verbose=True)
    print("\nTop 5 by score:")
    cols = [c for c in ["beta", "gamma", "score", "reason",
                        "tot_Wp", "tot_pp", "S_gene_mean", "sigma_mean"]
            if c in df.columns]
    print(df[cols].head(5).to_string(index=False))

    plot_grid(df, "beta", "gamma",
              fname="/home/claude/grid_beta_gamma.png")

    # --- 2) Optuna search ---------------------------------------------
    print("\n=== Optuna search (TPE) ===")
    study = optuna_search(n_trials=60, T=500, dt=0.02,
                          fixed=fixed, seed=1)

    print(f"\nbest score = {study.best_value: .4f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k:11s} = {v:.4g}")
    print("best diagnostics:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k:14s} = {v}")

    plot_optuna(study)

    # --- 3) Re-run the best Optuna config for a final trajectory plot --
    print("\n=== Re-run best Optuna config ===")
    best_kw = {**fixed, **study.best_params}
    _, traj = run_tunable(T=1500, dt=0.02, **best_kw)
    score, diag = score_run(traj, tail=400)
    print(f"long-run score = {score: .3f}, diag = {diag}")

    from hier_v2 import plot as plot_traj
    plot_traj(traj, fname="/home/claude/best_run.png")
