"""
Visualise the two-fold expectation of the model:

  (1) Environmental dissipation maximises:
        sigma_env  =  d_i (a_ext - a) log(a_ext / a) + (production of polymers)
      grows over time as the system organises -- the Maximum Entropy
      Production Principle (MEPP) regime.

  (2) Genetic information entropy decreases:
        S_gene  =  H( distribution over gene-symbol strings, weighted by Wp )
      drops as selection concentrates the gene population on a few
      adaptive sequences -- analogous to the posterior entropy
      H(q(theta|D)) shrinking during diffusion-model training.

We run the Optuna-optimal config (a healthy NESS) and produce three
figures:

  fig1: time series of sigma_env(t), S_gene(t), and cumulative entropy
        EXPORTED to the bath, together with the negentropy bookkeeping
        sigma_env >= -dS_gene/dt (Schrödinger inequality at the symbolic
        level).
  fig2: scatter of (cumulative sigma_env)  vs (cumulative -dS_gene) with
        time as colour. A diagonal would be "perfect" negentropy
        bookkeeping.
  fig3: an analogy schematic mapping each block of the autocatalytic
        model onto the corresponding block of a diffusion-model training
        loop, with arrows for information / entropy flow.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

from tune import TunableHierAutocat, run_tunable


# --- best config from previous Optuna search -------------------------------
BEST = dict(
    N=8, M=4, L=3, seed=3,
    beta=1.451, gamma=0.1265, eps=2.79e-3,
    coef_bias=0.5982, prot_decay=1.094,
    diff_scale=1.546, th_scale=0.3856, gene_decay=0.01998,
)


# ---------------------------------------------------------------------------
def run_with_environment(T=1500, dt=0.02, **kw):
    """Run the tunable model and additionally accumulate the env-side
    entropy production explicitly."""
    m = TunableHierAutocat(**kw)
    rec = dict(sigma=[], sigma_env=[], sigma_int=[], S_gene=[], growth=[],
               cum_sigma_env=[], cum_neg_dSgene=[], a=[], pp=[], Wp=[])
    cum_env, cum_neg = 0.0, 0.0
    prev_Sgene = None
    for _ in range(T):
        out = m.step(dt=dt)
        # split entropy production into "environmental" (= bath/diffusion)
        # and "internal" (= reaction). The MEPP claim concerns sigma_env
        # in particular; sigma_int is generated inside the metabolic web.
        sigma_env = out["sigma_diff"]
        sigma_int = out["sigma_react"]
        S = out["S_gene"]

        cum_env += sigma_env * dt
        if prev_Sgene is not None:
            cum_neg += -(S - prev_Sgene)        # accumulate -dS_gene
        prev_Sgene = S

        rec["sigma"].append(out["sigma"])
        rec["sigma_env"].append(sigma_env)
        rec["sigma_int"].append(sigma_int)
        rec["S_gene"].append(S)
        rec["growth"].append(out["growth"])
        rec["cum_sigma_env"].append(cum_env)
        rec["cum_neg_dSgene"].append(cum_neg)
        rec["a"].append(out["a"]); rec["pp"].append(out["pp"])
        rec["Wp"].append(out["Wp"])
    for k in rec:
        rec[k] = np.array(rec[k])
    return m, rec


# ---------------------------------------------------------------------------
def fig_mepp(rec, fname="/home/claude/mepp_negentropy.png"):
    """Figure 1: simultaneous σ_env increase + S_gene decrease."""
    t = np.arange(len(rec["sigma_env"]))

    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    # --- σ_env(t) and σ_int(t) -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, rec["sigma_env"], color="#2980b9", lw=1.1,
            label=r"$\sigma_\mathrm{env}(t)$ : bath / diffusion")
    ax.plot(t, rec["sigma_int"], color="#e67e22", lw=0.9,
            label=r"$\sigma_\mathrm{int}(t)$ : reactions")
    ax.set_yscale("symlog")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\sigma$")
    ax.set_title(r"MEPP side: environmental EP grows as the cell organises")
    ax.legend(fontsize=9)

    # --- S_gene(t) -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    Smax = float(np.log(8))   # log of # genes
    ax.plot(t, rec["S_gene"], color="#16a085", lw=1.2,
            label=r"$S_\mathrm{gene}(t)$")
    ax.axhline(Smax, color="grey", ls="--", lw=0.7,
               label=r"$\log N$ (uniform)")
    ax.set_xlabel("step"); ax.set_ylabel("nats")
    ax.set_title(r"Selection side: genetic entropy decreases")
    ax.set_ylim(rec["S_gene"].min() - 0.05, Smax + 0.05)
    ax.legend(fontsize=9, loc="lower left")

    # --- Cumulative bookkeeping: ∫σ_env dt   vs   -ΔS_gene -------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, rec["cum_sigma_env"], color="#2980b9", lw=1.4,
            label=r"$\int_0^t \sigma_\mathrm{env}\, dt$ (exported to bath)")
    ax.plot(t, rec["cum_neg_dSgene"], color="#16a085", lw=1.4,
            label=r"$-\Delta S_\mathrm{gene}$ (order created in genes)")
    ax.set_xlabel("step"); ax.set_ylabel("cumulative nats")
    ax.set_yscale("symlog")
    ax.set_title("Negentropy bookkeeping (Schrödinger): export ≫ internal order")
    ax.legend(fontsize=9)

    # --- two-axes overlay to show their joint trajectory ---------------
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, rec["sigma_env"], color="#2980b9", lw=1.1,
            label=r"$\sigma_\mathrm{env}(t)$")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\sigma_\mathrm{env}$",
                                          color="#2980b9")
    ax.tick_params(axis="y", labelcolor="#2980b9")
    ax.set_yscale("symlog")
    ax2 = ax.twinx()
    ax2.plot(t, rec["S_gene"], color="#16a085", lw=1.2)
    ax2.set_ylabel(r"$S_\mathrm{gene}$", color="#16a085")
    ax2.tick_params(axis="y", labelcolor="#16a085")
    ax.set_title(r"Joint view: $\sigma_\mathrm{env}\uparrow$  and"
                 r"  $S_\mathrm{gene}\downarrow$  co-occur")

    fig.suptitle(
        "MEPP + selection: maximum entropy production into the bath  ⇔  "
        "minimum entropy of the gene distribution",
        fontsize=12, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
def fig_scatter(rec, fname="/home/claude/mepp_negentropy_scatter.png"):
    """Figure 2: trajectory in the (∫σ_env, -ΔS_gene) plane."""
    t = np.arange(len(rec["sigma_env"]))
    x = rec["cum_sigma_env"]
    y = rec["cum_neg_dSgene"]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axs[0]
    sc = ax.scatter(x, y, c=t, cmap="plasma", s=8, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="time step")
    ax.set_xlabel(r"$\int_0^t \sigma_\mathrm{env}\, dt$"
                  r"   (entropy exported to bath)")
    ax.set_ylabel(r"$-\Delta S_\mathrm{gene}$"
                  r"   (order created in gene distribution)")
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_title("System trajectory in negentropy plane")
    ax.grid(True, ls=":", alpha=0.5)

    # ratio (dimensionless "thermodynamic cost of one nat of order")
    ax = axs[1]
    valid = (np.abs(y) > 1e-6) & (t > 50)
    ratio = np.full_like(x, np.nan, dtype=float)
    ratio[valid] = x[valid] / np.maximum(y[valid], 1e-9)
    ax.plot(t, ratio, color="#8e44ad", lw=1.0)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\int\sigma_\mathrm{env}\,dt \ / \ (-\Delta S_\mathrm{gene})$")
    ax.set_title("Thermodynamic cost per nat of genetic order")
    ax.set_yscale("symlog")
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle("Cells pay environmental entropy to compress the gene distribution",
        fontsize=12, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
def fig_analogy(fname="/home/claude/diffusion_analogy.png"):
    """Figure 3: side-by-side conceptual diagram, autocatalytic vs diffusion."""
    fig, ax = plt.subplots(figsize=(13.5, 7), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis("off")

    # palette
    blue = "#3498db"; green = "#16a085"; orange = "#e67e22"
    grey = "#7f8c8d"; navy = "#2c3e50"

    def box(x, y, w, h, color, text, fs=9, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    fc=color, ec="none"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                color=tc, fontsize=fs, weight="bold")

    def arrow(x1, y1, x2, y2, color="black", style="-|>", lw=1.4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, color=color,
                                     mutation_scale=14, lw=lw))

    # ---- left half: autocatalytic system -----------------------------
    ax.text(2.8, 7.55, "Hierarchical autocatalytic system",
            ha="center", fontsize=13, weight="bold", color=navy)

    box(0.4, 6.0, 1.7, 0.9, blue,   "external bath\n$a_\\mathrm{ext}$",      fs=9)
    box(2.5, 6.0, 1.7, 0.9, blue,   "raw molecules\n$a_i$",                  fs=9)
    box(4.6, 6.0, 1.0, 0.9, grey,   "diffusion\n$d_i$",                      fs=8)

    box(0.4, 4.2, 1.7, 1.0, orange, "proteins\n$p_l$, $\\tanh(\\beta W p - \\theta)$", fs=8)
    box(2.5, 4.2, 1.7, 1.0, orange, "Coef$_{i,l,j}$\n(slightly broken\nmass-action)", fs=8)

    box(0.4, 2.4, 1.7, 1.0, green,  "genes $W_p^{(k)}$\n(large RNA/protein)", fs=9)
    box(2.5, 2.4, 1.7, 1.0, green,  "symbol distribution\n$p(s)$",            fs=9)

    box(0.4, 0.6, 3.8, 0.9, navy,   "observable: $S_\\mathrm{gene}=H(p)$  ↓  while  $\\sigma_\\mathrm{env}$  ↑", fs=10)

    # arrows on the left
    arrow(2.1, 6.45, 2.5, 6.45)        # bath -> raw
    arrow(3.5, 6.0, 3.5, 5.2)          # raw -> reactions
    arrow(2.1, 4.7, 2.5, 4.7)          # proteins -> coef
    arrow(1.25, 4.2, 1.25, 3.4)        # proteins -> genes
    arrow(2.1, 2.9, 2.5, 2.9)          # genes -> symbols
    arrow(2.3, 2.4, 2.3, 1.5)          # symbols -> observable
    arrow(4.2, 6.0, 4.2, 5.2, color=blue)   # raw -> down (substrate)
    arrow(5.1, 6.4, 5.1, 7.2, color=blue)   # env EP up to bath
    ax.text(5.4, 7.0, "$\\sigma_\\mathrm{env}\\uparrow$", color=blue,
            fontsize=11, weight="bold")

    # divider
    ax.plot([6.4, 6.4], [0.3, 7.4], color="#bdc3c7", lw=1.2, ls="--")

    # ---- right half: diffusion model ---------------------------------
    ax.text(9.5, 7.55, "Diffusion-model training (analogy)",
            ha="center", fontsize=13, weight="bold", color=navy)

    box(6.8, 6.0, 1.7, 0.9, blue,   "data distribution\n$p_\\mathrm{data}(x)$",  fs=9)
    box(8.9, 6.0, 1.7, 0.9, blue,   "noisy samples\n$x_t = x + \\sigma_t \\xi$",  fs=8)
    box(11.0, 6.0, 1.0, 0.9, grey,  "noise sched.\n$\\sigma_t$",                  fs=8)

    box(6.8, 4.2, 1.7, 1.0, orange, "score net\n$s_\\theta(x_t,t)$",              fs=9)
    box(8.9, 4.2, 1.7, 1.0, orange, "loss\n$\\|s_\\theta - \\nabla\\log p\\|^2$", fs=8)

    box(6.8, 2.4, 1.7, 1.0, green,  "parameters $\\theta$\n(weights)",            fs=9)
    box(8.9, 2.4, 1.7, 1.0, green,  "posterior $q(\\theta\\,|\\,\\mathcal{D})$",  fs=9)

    box(6.8, 0.6, 3.8, 0.9, navy,   "observable: $H[q(\\theta|\\mathcal{D})]$  ↓  while  KL[$p_\\mathrm{data}\\|p_\\theta$]  ↓", fs=10)

    # arrows on the right
    arrow(8.5, 6.45, 8.9, 6.45)
    arrow(9.95, 6.0, 9.95, 5.2)
    arrow(8.5, 4.7, 8.9, 4.7)
    arrow(7.65, 4.2, 7.65, 3.4)
    arrow(8.5, 2.9, 8.9, 2.9)
    arrow(8.7, 2.4, 8.7, 1.5)
    arrow(10.6, 6.0, 10.6, 5.2, color=blue)
    arrow(11.5, 6.4, 11.5, 7.2, color=blue)
    ax.text(11.7, 7.0, "data-info\n flow ↑", color=blue,
            fontsize=9, weight="bold", ha="left")

    # ---- correspondences (bottom bar) --------------------------------
    cor = [
        (r"$a_\mathrm{ext}, a_i$",          r"$p_\mathrm{data}(x)$"),
        (r"$\sigma_\mathrm{env}\uparrow$",  r"data information flow into model"),
        (r"$\tanh(\beta W p - \theta)$",    r"score / denoiser network"),
        (r"replication noise $\epsilon$",   r"forward diffusion noise"),
        (r"selection by fitness",           r"gradient on log-likelihood"),
        (r"$S_\mathrm{gene}\downarrow$",    r"$H[q(\theta|\mathcal{D})]\downarrow$"),
    ]
    fig.suptitle(
        "Correspondence:  thermodynamic order (MEPP + selection)  ↔  "
        "Bayesian order (data flow + posterior contraction)",
        fontsize=12, weight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")
    return fname


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _, rec = run_with_environment(T=1500, dt=0.02, **BEST)

    print(f"\nFinal cumulative env entropy production = {rec['cum_sigma_env'][-1]: .3e}")
    print(f"Final -ΔS_gene                          = {rec['cum_neg_dSgene'][-1]: .3e}")
    print(f"Ratio  ∫σ_env dt / (-ΔS_gene)            "
          f"= {rec['cum_sigma_env'][-1] / max(rec['cum_neg_dSgene'][-1], 1e-9): .3e}")
    print(f"S_gene(0)  = {rec['S_gene'][0]: .4f}")
    print(f"S_gene(T)  = {rec['S_gene'][-1]: .4f}")

    fig_mepp(rec)
    fig_scatter(rec)
    fig_analogy()
