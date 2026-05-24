"""
Hierarchical autocatalytic reaction system with foldable macromolecules.
v2: stronger autocatalytic core, gene-level competition, cleaner entropy
production accounting.

Hierarchy
---------
  a_i (i=1..L)  raw molecules (CO2, H2O, phosphate ...). Exchanged with
                external bath a_ext via diffusion d_i.
  p_l (l=1..M)  proteins / small ribozymes. Their populations p_p obey a
                simple autocatalytic law: each protein consumes raw a and
                catalyses its own production, with rate modulated in
                [-1,1] by the activity of the genes.
  x_k (k=1..N)  large RNA / protein "genes". W[:,:,k] encodes the fold:
                the protein activity is
                  pa = tanh(beta * (W_eff @ p_p) - th)
                where W_eff is the gene-population weighted W. The
                modulation factor entering the protein dynamics is
                  mod_l = tanh(beta_mod * pa_l)        in [-1,1].
  Coef[i,l,j]   slightly-broken mass-action stoichiometry tensor.

Outputs
-------
  - sigma(t)   total entropy production (reaction + diffusion)
  - growth(t)  mean d(log W_p)/dt over genes
  - S_gene(t)  Shannon entropy of the gene-symbol-string distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
class HierAutocat:
    def __init__(self, N=8, M=4, L=3,
                 beta=2.0, gamma=0.4, eps=1e-3, seed=3):
        self.N, self.M, self.L = N, M, L
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        r = self.rng

        # gene "fold" tensor: each gene k carries an M x M matrix of
        # connection weights from proteins to fold-stable states.
        self.W = r.standard_normal((M, M, N)) * 0.9

        # populations
        self.Wp = r.uniform(0.05, 0.15, N)          # genes
        self.pp = r.uniform(0.15, 0.35, M)          # proteins
        self.a  = np.ones(L) * 1.0                   # raw molecules

        self.a_ext = np.array([3.0, 2.5, 2.0])[:L]   # external bath

        # autocatalytic stoichiometry tensor (slightly biased positive so
        # the system has a non-trivial fixed point above zero).
        self.Coef = r.standard_normal((L + M, M, L + M)) * 0.15
        # diagonal autocatalytic boost: protein l catalyses its own
        # production from raw molecules only (avoid pp*pp blow-up).
        for l in range(M):
            for j in range(L):
                self.Coef[L + l, l, j] += 0.35       # a_j -> p_l by p_l
        # explicitly damp protein-protein quadratic channels
        self.Coef[L:, :, L:] *= 0.1

        self.Kon = np.ones(L) * 1.0
        self.d   = np.array([0.4, 0.3, 0.25])[:L]
        self.th  = r.standard_normal(M) * 0.2

        self.pa = np.zeros(M)
        self.symbols = self._symbols_from_W()

    # -------------------------------------------------------------------
    def _symbols_from_W(self):
        """Binary fold-signature string per gene (length M)."""
        syms = []
        for k in range(self.N):
            A = 0.5 * (self.W[:, :, k] + self.W[:, :, k].T)
            _, v = np.linalg.eigh(A)
            sig = (v[:, -1] > 0).astype(int)
            syms.append("".join(map(str, sig)))
        return syms

    def fold_activity(self):
        """pa = tanh(beta * W_eff @ pp - th)."""
        w = self.Wp / (self.Wp.sum() + 1e-9)
        W_eff = np.einsum("lmk,k->lm", self.W, w)
        return np.tanh(self.beta * (W_eff @ self.pp - self.th))

    # -------------------------------------------------------------------
    def step(self, dt=0.02):
        L, M, N = self.L, self.M, self.N
        rng = self.rng

        # 1) refresh protein activity from current fold
        self.pa = self.fold_activity()

        # 2) modulation in [-1, 1]
        mod = np.tanh(self.pa)                                # shape (M,)

        # 3) full state for the reaction tensor
        y = np.concatenate([self.a, self.pp])                 # (L+M,)
        mass = np.outer(y, y)                                 # (L+M, L+M)

        # 4) reaction rates with slight mass-action breaking via eps noise
        r_full = np.einsum("ilj,l,ij->i", self.Coef, mod, mass)
        r_full = r_full + self.eps * rng.standard_normal(L + M)
        r_a, r_p = r_full[:L], r_full[L:]

        # 5) diffusion with external bath
        diff = self.d * (self.a_ext - self.a)

        # 6) raw molecules: catalysis consumes them, diffusion replenishes
        # We subtract consumption proportional to total protein production
        # so mass is approximately conserved between layers.
        consumption = (r_p.sum() / L) * np.ones(L) * 0.5
        da = self.Kon * r_a + diff - consumption
        self.a = np.clip(self.a + dt * da, 1e-6, 50.0)

        # 7) protein dynamics: catalytic production - fragile decay
        dpp = r_p - self.gamma * 2.5 * self.pp
        self.pp = np.clip(self.pp + dt * dpp, 1e-6, 50.0)

        # 8) gene dynamics. Each gene k has a "fitness" given by how well
        # its fold-signature aligns with the current protein activity and
        # how much substrate is available. Genes compete for substrate.
        # The "fitness" of gene k is the alignment of its fold pattern
        # with the realised protein population.
        align = np.einsum("lmk,l,m->k", self.W, self.pa, self.pp)
        fitness = np.tanh(align)                              # in [-1,1]
        substrate = self.a.sum()

        # logistic competition with total carrying capacity ~ substrate
        K = 1.0 + substrate
        total = self.Wp.sum()
        gene_growth = fitness * (1.0 - total / K)
        dWp = self.Wp * gene_growth - 0.05 * self.Wp          # fragile decay
        self.Wp = np.clip(self.Wp + dt * dWp, 1e-9, None)

        # replication mutation noise on W
        self.W += dt * self.eps * 5.0 * rng.standard_normal(self.W.shape)
        self.symbols = self._symbols_from_W()

        # -------- entropy production -----------------------------------
        # Reaction contribution: sigma_react = sum_i (f_i - b_i) log(f_i/b_i)
        # with backward floor for the eps-broken mass-action regime.
        fwd = np.maximum(r_full, 0) + 1e-9
        bwd = np.maximum(-r_full, 0) + self.eps + 1e-9
        sigma_react = np.sum((fwd - bwd) * np.log(fwd / bwd))

        # Diffusive contribution: sigma_diff = sum_i d_i (a_ext - a) log(a_ext/a)
        sigma_diff = np.sum(self.d * (self.a_ext - self.a)
                            * np.log((self.a_ext + 1e-9) / (self.a + 1e-9)))
        sigma = sigma_react + sigma_diff

        # -------- gene growth rate -------------------------------------
        growth_rate = float(np.mean(dWp / (self.Wp + 1e-9)))

        # -------- genetic entropy --------------------------------------
        w = self.Wp / self.Wp.sum()
        bins = {}
        for s, wk in zip(self.symbols, w):
            bins[s] = bins.get(s, 0.0) + wk
        p = np.array(list(bins.values()))
        S_gene = float(-np.sum(p * np.log(p + 1e-12)))

        return dict(sigma=sigma, growth=growth_rate, S_gene=S_gene,
                    sigma_react=sigma_react, sigma_diff=sigma_diff,
                    a=self.a.copy(), pp=self.pp.copy(),
                    pa=self.pa.copy(), Wp=self.Wp.copy(),
                    n_species=len(bins))


# ---------------------------------------------------------------------------
def run(T=1500, dt=0.02, **kw):
    m = HierAutocat(**kw)
    keys = ["sigma", "growth", "S_gene", "sigma_react", "sigma_diff",
            "a", "pp", "pa", "Wp", "n_species"]
    traj = {k: [] for k in keys}
    for _ in range(T):
        out = m.step(dt=dt)
        for k in keys:
            traj[k].append(out[k])
    traj={k:np.array(traj[k]) for k in keys}
    return m, traj

def _plots(ax,t,data:list,title:str,xlabel="step",ylabel="σ",yscale="symlog",axhline=False):
    for d in data:
        if(d["color"]==None):
            d["color"]="#2c3e50"
        ax.plot(t, d["traj"],color=d["color"], lw=d["lw"], label=d["label"])
    ax.set_title(title) #"Entropy production  σ(t)")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale(yscale); ax.legend(fontsize=8)
    if(axhline):
        ax.axhline(0, color="grey", lw=0.5)        
    ax.legend(fontsize=8, ncol=2)
    return ax
# ---------------------------------------------------------------------------
def plot(traj, fname="result/result.png"):
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    t = np.arange(len(traj["sigma"]))

    ax = fig.add_subplot(gs[0, 0])
    _plots(ax,t,[
            {"traj":traj["sigma"],"color":"#c0392b","lw":1.0,"label":"total σ"},
            {"traj":traj["sigma_react"],"color":"#e67e22","lw":0.8,"label":"reaction"},
            {"traj":traj["sigma_diff"],"color":"#3498db","lw":0.8,"label":"diffution"},
            ],"Entropy production  σ(t)")

    ax = fig.add_subplot(gs[0, 1])
    _plots(ax,t,[
        {"traj":traj["growth"],"color":"#2c3e50", "lw":1.0}
        ],
        "Mean growth rate of large RNA/protein  ⟨d log W_p / dt⟩",axhline=True,ylabel="growth")

    ax = fig.add_subplot(gs[1, 0])
    _plots(ax,t,[
            {"traj":traj["S_gene"],"color":"#16a085","lw":1.0,"label":"S_gene"},        
            {"traj":np.log(traj["n_species"]),"color":"#7f8c8d","lw":0.7,"label":"S_gene"},
            ],"Genetic entropy  S_gene = H(symbol-string distribution)",
            yscale="log(#species)  [max bound]",ylabel="nats")

    ax = fig.add_subplot(gs[1, 1])
    _plots(ax,t,[
             {"traj":traj["a"][:, i], "lw":0.8, "label":f"a_{i}"}  for i in range(traj["a"].shape[1]) ],
             "Raw molecules  a_i(t)")

    ax = fig.add_subplot(gs[2, 0])
    _plots(ax,t,[
            {"traj":traj["pp"][:, l], "lw":0.8, "label":f"a_{l}"} for l in range(traj["pp"].shape[1]) ],
            "Protein populations  p_p(t)")

    ax = fig.add_subplot(gs[2, 1])
    _plots(ax,t,[
        {"traj":traj["Wp"][:, k], "lw":0.8, "label":f"gene {k}"} for k in range(traj["Wp"].shape[1])],
        "Gene populations  W_p(t)",
        yscale="log"
        )

    fig.suptitle("Hierarchical autocatalytic system: σ vs growth vs S_gene",
                 fontsize=13, fontweight="bold")
    fig.savefig(fname, dpi=130)
    print(f"saved -> {fname}")

    # joint relations
    fig2, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    cm = plt.get_cmap("viridis")

    sc = axs[0].scatter(traj["sigma"], traj["growth"], s=4, alpha=0.5, c=t,
                        cmap=cm)
    axs[0].set_xlabel("σ"); axs[0].set_ylabel("growth")
    axs[0].set_xscale("symlog"); axs[0].set_title("σ  vs  growth")

    axs[1].scatter(traj["sigma"], traj["S_gene"], s=4, alpha=0.5, c=t, cmap=cm)
    axs[1].set_xlabel("σ"); axs[1].set_ylabel("S_gene")
    axs[1].set_xscale("symlog"); axs[1].set_title("σ  vs  S_gene")

    axs[2].scatter(traj["growth"], traj["S_gene"], s=4, alpha=0.5, c=t, cmap=cm)
    axs[2].set_xlabel("growth"); axs[2].set_ylabel("S_gene")
    axs[2].set_title("growth  vs  S_gene  (colour = time)")

    fig2.colorbar(sc, ax=axs, shrink=0.7, label="time step")
    fig2.suptitle("Joint relations between σ, growth and genetic entropy",
                  fontsize=12, fontweight="bold")
    fname2 = fname.replace(".png", "_joint.png")
    fig2.savefig(fname2, dpi=130)
    print(f"saved -> {fname2}")
    return fname, fname2


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, traj = run(T=1500, dt=0.02,
                      N=8, M=4, L=3,
                      beta=2.0, gamma=0.4, eps=1e-3, seed=3)
    plot(traj)
    n = 50
    print()
    print(f"avg(last {n}) σ            = {traj['sigma'][-n:].mean(): .3e}")
    print(f"avg(last {n}) growth rate  = {traj['growth'][-n:].mean(): .3e}")
    print(f"avg(last {n}) S_gene       = {traj['S_gene'][-n:].mean(): .3e}")
    print(f"final symbol strings        = {model.symbols}")
    print(f"final Wp                    = {model.Wp.round(3)}")
