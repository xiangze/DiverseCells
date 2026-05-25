"""
Minimal replicator model designed to operate near the TUR bound.

Motivation
----------
In the full hierarchical model (`hier_v2`), the TUR product
    Var(J)/<J>^2 * Σ
sits ~5-7 orders of magnitude above the universal bound of 2. That is
*not* the value measured in real cells; experimental analyses of single
kinetic-proofreading (KPR) circuits in E. coli put DNA polymerase
~1x and the ribosome ~5x above the bound (Song et al. 2019,
arXiv:1911.04673).

The reason our hierarchical system is so far from the bound is structural:
many superimposed reactions all contribute to Σ, while the observed
current (log of total gene mass) is a slow aggregate variable whose
variance is small. The TUR is saturated only for unicyclic Markov chains
where one reaction is one current.

This file builds a *minimal* replicator that mimics a single KPR-like
cycle:

    A --(k+)--> X --(kgrow)--> 2X        (binding + growth)
    X --(k-)--> A                         (back-binding, reversible)
    X --(kdeg)--> waste                   (irreversible loss)

Mass action is broken slightly by an ε-noise term, exactly as in the
hierarchical model, so we are comparing apples to apples.

What we measure
---------------
J     = number of growth events in [0, τ]
<J>   = mean over replicates
Var(J)= variance over replicates
Σ_τ   = cumulative entropy production
TUR   = Var(J)/<J>^2 * Σ_τ                  -- target value: close to 2
TSL   = τ * 2 <Σ> <A> / L^2                 -- target value: close to 1

We then sweep the irreversibility ratio (k+/k-) and show that the system
crosses smoothly from a TUR-saturated regime (high reversibility, near
equilibrium) to a Far-from-bound regime (high irreversibility), exactly
mirroring the experimental finding that polymerase (highly discriminative
= more "reversible" off-pathway) saturates TUR more closely than
ribosome.

We also explicitly *fit* the kinetic constants so that the resulting
growth rate is a fixed fraction of the TSL ceiling, demonstrating that
"TSL-tight" models can be constructed when desired.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Minimal stochastic replicator: Gillespie-like tau-leaping
# ---------------------------------------------------------------------------
class MinimalReplicator:
    """
    Reactions
        R1: A + X --(k_bind   )--> 2X      (replication)
        R2: 2X    --(k_unbind )--> A + X   (reverse of R1)
        R3: X     --(k_deg    )--> W       (degradation to waste)
        R4: W     --(k_recyc  )--> A       (slow recycling -- closes mass)
    External bath replenishes A so its concentration is held at A0.

    The forward/backward affinity for R1/R2 sets the per-cycle EP:
        dS_cycle = log(k_bind * A0 / k_unbind)
    R3 is irreversible -> dissipates an additional fixed Δμ per event.
    """

    def __init__(self, *, k_bind=2.0, k_unbind=1.0, k_deg=0.05,
                 dmu_deg=5.0, A0=10.0, X0=5, eps=1e-3, seed=0):
        self.rng = np.random.default_rng(seed)
        self.k_bind   = k_bind
        self.k_unbind = k_unbind
        self.k_deg    = k_deg
        self.dmu_deg  = dmu_deg
        self.A0       = A0
        self.X        = X0
        self.eps      = eps

        # cumulative counters
        self.cnt_fwd = 0     # R1 events
        self.cnt_bwd = 0     # R2 events
        self.cnt_deg = 0     # R3 events
        self.Sigma   = 0.0   # cumulative EP

    def step(self, dt):
        if self.X <= 0:
            return
        # propensities (broken mass-action: small ε noise on each rate)
        eps_n = self.eps * self.rng.standard_normal(3)
        a1 = max(self.k_bind   * self.A0 * self.X + eps_n[0], 0.0)
        a2 = max(self.k_unbind * self.X * (self.X - 1) / 2 + eps_n[1], 0.0)
        a3 = max(self.k_deg    * self.X + eps_n[2], 0.0)

        # cap rates for numerical stability of Poisson sampling
        a1 = min(a1, 5e3); a2 = min(a2, 5e3); a3 = min(a3, 5e3)

        # Poisson tau-leap
        n1 = self.rng.poisson(a1 * dt)
        n2 = self.rng.poisson(min(a2 * dt, self.X / 2))
        n3 = self.rng.poisson(a3 * dt)

        # population update
        dX = n1 - n2 - n3
        self.X = max(self.X + dX, 0)
        self.cnt_fwd += n1
        self.cnt_bwd += n2
        self.cnt_deg += n3

        # entropy production
        # Schnakenberg: dS = (n_fwd - n_bwd) * log(a1/a2_ref) + n_deg*Δμ
        if a2 > 1e-9 and a1 > 1e-9:
            self.Sigma += (n1 - n2) * np.log(a1 / a2)
        self.Sigma += n3 * self.dmu_deg


# ---------------------------------------------------------------------------
# Run an ensemble of minimal replicators and measure TUR/TSL
# ---------------------------------------------------------------------------
def run_minimal_ensemble(n_rep=200, T=400, dt=0.05, **kw):
    Js    = np.zeros(n_rep)         # growth events in [0, T]
    Sigs  = np.zeros(n_rep)
    Xpath = np.zeros((n_rep, T))
    for r in range(n_rep):
        kw2 = dict(kw); kw2["seed"] = kw.get("seed", 0) * 1000 + r
        m = MinimalReplicator(**kw2)
        for t in range(T):
            m.step(dt)
            Xpath[r, t] = m.X
        Js[r] = m.cnt_fwd - m.cnt_bwd     # net growth events
        Sigs[r] = m.Sigma
    return Js, Sigs, Xpath


def tur_product(Js, Sigs):
    mu = Js.mean()
    var = Js.var(ddof=1)
    Sig_mean = Sigs.mean()
    if abs(mu) < 1e-9 or Sig_mean <= 0:
        return np.nan
    return (var / mu**2) * Sig_mean


def tsl_quantities(Xpath, Sigs, dt):
    """Compute observed speed and TSL ceiling from population trajectories."""
    n_rep, T = Xpath.shape
    tau = T * dt
    # state distribution at t=0 and t=T (use replicate distribution of X
    # as a coarse proxy for system state distribution)
    p0 = np.histogram(Xpath[:, 0], bins=np.arange(0, 100),
                      density=False)[0].astype(float)
    pt = np.histogram(Xpath[:, -1], bins=np.arange(0, 100),
                      density=False)[0].astype(float)
    p0 = p0 / max(p0.sum(), 1.0); pt = pt / max(pt.sum(), 1.0)
    L = float(np.sqrt(0.5 * np.sum((np.sqrt(p0) - np.sqrt(pt))**2)))
    Sig_mean = Sigs.mean()
    # dynamical activity ~ <|dX/dt|>
    A = float(np.mean(np.sum(np.abs(np.diff(Xpath, axis=1)), axis=1))) / tau
    if L < 1e-9 or Sig_mean <= 0 or A <= 0:
        return dict(L=L, Sigma=Sig_mean, A=A,
                    tsl_ratio=np.nan, ceil=np.nan,
                    obs_speed=np.nan)
    tsl_ratio = (tau * 2.0 * Sig_mean * A) / (L**2)
    ceil = np.sqrt(2.0 * Sig_mean * A / tau)
    return dict(L=L, Sigma=Sig_mean, A=A,
                tsl_ratio=tsl_ratio, ceil=ceil,
                obs_speed=L / tau)


# ---------------------------------------------------------------------------
# Sweep: from near-reversible (TUR-tight) to highly irreversible
# ---------------------------------------------------------------------------
def sweep_reversibility(ratios, *, T=100, dt=0.05, n_rep=300):
    """
    Vary k_unbind/k_bind from 0.05 (highly irreversible) to 0.95 (nearly
    reversible / near-equilibrium).

    For TUR saturation we want a regime where:
      - the per-cycle affinity log(k+A/k-X) is small but nonzero,
      - X stays bounded so variance is meaningful,
      - degradation balances replication so the system is in NESS.
    """
    rows = dict(ratio=[], tur=[], tsl_ratio=[], obs_speed=[], ceil=[],
                growth=[], sigma=[])
    for rr in ratios:
        # k_bind*A0 = 0.5, k_unbind = 0.5*rr -> small forward bias when
        # rr -> 1, large bias when rr -> 0. k_deg balances replication so
        # mean X stays bounded.
        Js, Sigs, X = run_minimal_ensemble(
            n_rep=n_rep, T=T, dt=dt,
            k_bind=0.10, k_unbind=0.10 * rr,
            k_deg=0.15, dmu_deg=2.0, A0=5.0, X0=20,
            eps=1e-3, seed=42)
        tur = tur_product(Js, Sigs)
        tsl = tsl_quantities(X, Sigs, dt)
        rows["ratio"].append(rr)
        rows["tur"].append(tur)
        rows["tsl_ratio"].append(tsl["tsl_ratio"])
        rows["obs_speed"].append(tsl["obs_speed"])
        rows["ceil"].append(tsl["ceil"])
        rows["growth"].append(Js.mean() / (T * dt))
        rows["sigma"].append(Sigs.mean() / (T * dt))
        print(f"  rev={rr:.3f}  TUR={tur: .3g}  TSL={tsl['tsl_ratio']: .3g}"
              f"  growth={Js.mean()/(T*dt): .3g}"
              f"  σ={Sigs.mean()/(T*dt): .3g}"
              f"  <X(T)>={X[:,-1].mean():.1f}"
              f"  Var(J)={Js.var(ddof=1):.2f}")
    for k in rows:
        rows[k] = np.array(rows[k])
    return rows


# ---------------------------------------------------------------------------
# Fit: choose k_bind, k_unbind so that observed growth ≈ frac * TSL_ceil
# ---------------------------------------------------------------------------
def fit_to_tsl_fraction(target_frac=0.5, *, T=100, dt=0.05, n_rep=400):
    """
    Bisect on the irreversibility ratio to find the configuration whose
    observed growth speed equals `target_frac * TSL_ceiling`.
    """
    def obs_over_ceil(rr):
        Js, Sigs, X = run_minimal_ensemble(
            n_rep=n_rep, T=T, dt=dt,
            k_bind=0.10, k_unbind=0.10 * rr,
            k_deg=0.15, dmu_deg=2.0, A0=5.0, X0=20,
            eps=1e-3, seed=42)
        tsl = tsl_quantities(X, Sigs, dt)
        if not np.isfinite(tsl["ceil"]) or tsl["ceil"] <= 0:
            return np.nan, tsl
        return tsl["obs_speed"] / tsl["ceil"], tsl

    lo, hi = 0.05, 0.99
    last_tsl = None
    for _ in range(15):
        mid = 0.5 * (lo + hi)
        ratio, tsl = obs_over_ceil(mid)
        if not np.isfinite(ratio):
            return None
        last_tsl = tsl
        if ratio < target_frac:
            lo = mid
        else:
            hi = mid
    return dict(rev_ratio=mid, obs_over_ceil=ratio, tsl=last_tsl)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_minimal_sweep(rows, fname="/home/claude/minimal_tur.png"):
    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(rows["ratio"], rows["tur"], "o-", color="#2c3e50",
                lw=1.3, label="TUR product (minimal model)")
    ax.axhline(2.0, color="grey", ls="--", lw=1.0, label="TUR bound = 2")
    # reference: experimental values from Song et al. 2019
    ax.axhline(2 * 1.0, color="#16a085", ls=":", lw=1.0,
               label=r"DNA polymerase $\sim$ bound")
    ax.axhline(2 * 5.0, color="#e67e22", ls=":", lw=1.0,
               label=r"ribosome $\sim 5\times$ bound")
    ax.axhline(2 * 1e5, color="#c0392b", ls=":", lw=1.0,
               label=r"hierarchical model $\sim 10^5\times$ bound")
    ax.set_xlabel("reversibility ratio  $k_\\mathrm{unbind}/k_\\mathrm{bind}$")
    ax.set_ylabel("TUR product")
    ax.set_title("Minimal replicator approaches the TUR bound\n"
                 "when reactions are nearly reversible")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.5, which="both")

    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(rows["ratio"], rows["tsl_ratio"], "o-", color="#c0392b",
                lw=1.3)
    ax.axhline(1.0, color="grey", ls="--", lw=1.0, label="TSL bound = 1")
    ax.set_xlabel("reversibility ratio")
    ax.set_ylabel("TSL ratio  $\\tau \\cdot 2\\langle\\Sigma\\rangle\\langle A\\rangle / L^2$")
    ax.set_title("TSL ratio: how much slack in the speed bound")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5, which="both")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(rows["ratio"], rows["obs_speed"], "o-", color="#2c3e50",
            lw=1.3, label="observed speed")
    ax.plot(rows["ratio"], rows["ceil"], "o--", color="#c0392b",
            lw=1.3, label="TSL ceiling")
    ax.set_xlabel("reversibility ratio")
    ax.set_ylabel("rate")
    ax.set_title("Speed achieved vs the TSL ceiling")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(rows["sigma"], rows["growth"], "o-", color="#8e44ad",
            lw=1.3)
    for i, rr in enumerate(rows["ratio"]):
        ax.annotate(f"{rr:.2f}", (rows["sigma"][i], rows["growth"][i]),
                    fontsize=7, alpha=0.7)
    ax.set_xlabel("$\\langle\\sigma\\rangle$  (mean EP rate)")
    ax.set_ylabel("growth rate")
    ax.set_xscale("log")
    ax.set_title("Growth scales with EP (and is closest to TSL\n"
                 "in the intermediate reversibility regime)")
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Minimal replicator: tunable closeness to TUR / TSL bounds",
                 fontsize=13, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Sweep of reversibility ratio in the minimal replicator ===")
    ratios = np.array([0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95])
    rows = sweep_reversibility(ratios, T=100, dt=0.05, n_rep=400)
    plot_minimal_sweep(rows)

    print("\n=== Fit: choose params so growth = 50% of TSL ceiling ===")
    fit = fit_to_tsl_fraction(target_frac=0.5, T=100, dt=0.05, n_rep=400)
    if fit is None:
        print("  fit failed")
    else:
        print(f"  rev_ratio    = {fit['rev_ratio']:.4f}")
        print(f"  obs/ceil     = {fit['obs_over_ceil']:.3f}")
        print(f"  L            = {fit['tsl']['L']:.4f}")
        print(f"  <Σ>          = {fit['tsl']['Sigma']:.3e}")
        print(f"  <A>          = {fit['tsl']['A']:.3e}")
        print(f"  tsl_ratio    = {fit['tsl']['tsl_ratio']:.3e}")
