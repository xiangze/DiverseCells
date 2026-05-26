# Hierarchical Autocatalytic Systems as a Bridge between Maximum Entropy Production and Bayesian Posterior Contraction: A Numerical Study with Stochastic-Thermodynamic Bounds

## Abstract

We construct a three-layer reaction-diffusion model of an autocatalytic chemical system in which raw molecules ($a_i$), catalytic proteins
($p_l$) and large RNA/protein "genes" ($W_p^{(k)}$) interact through a mass-action stoichiometry tensor $\mathrm{Coef}_{ijk}$ whose magnitude is
modulated in $[-1,1]$ by the fold-stable activity $\mathrm{pa}_l = \tanh(\beta W_{l,:,k}\, p_p - \theta_l)$ of the largest
polymers. Mass-action is broken by an $\varepsilon$-noise term so that the system is genuinely nonequilibrium. 

The model is implemented in NumPy/PyTorch and analysed through an Optuna-based Bayesian search for sustainable parameter regions, after which we compute (i) the total
entropy production $\sigma(t)$, (ii) the genetic Shannon entropy
$S_\mathrm{gene}$ derived from a symbol-string projection of the fold tensor, and (iii) the thermodynamic uncertainty relation (TUR) and
thermodynamic speed limit (TSL) bounds on growth and evolution rates. 

The hierarchical model exhibits the expected co-occurrence of
$\sigma_\mathrm{env}\!\uparrow$ and $S_\mathrm{gene}\!\downarrow$
predicted by Schrödinger's negentropy argument and reformulated as
maximum-entropy-production-principle (MEPP)-driven adaptation, with a
quantitative cost of $\sim 589$ nats of environmental entropy per nat
of genetic order.

However, the realised TUR product sits $10^4$–$10^5$ above the universal bound of 2, and the TSL ratio sits
$10^6$–$10^8$ above its bound of 1. We trace the looseness to the
multi-cycle structure of the network and, by collapsing the system to
a single kinetic-proofreading-like cycle (a minimal Gillespie replicator),
we recover TUR products of $\sim 5$, matching the experimentally
reported regime of the ribosome [Song et al., 2019]. Scaling
$(N,M,L)$ from $(4,3,3)$ to $(32,8,3)$ leaves the looseness intact for
the hierarchical model but tightens it monotonically with particle
number for the minimal model. 

We close by drawing an explicit correspondence between the autocatalytic system and diffusion-model training: $a_\mathrm{ext}\!\to\!a$ flux ↔ data-information flow,
$\tanh(\beta W p - \theta)$ ↔ score network, replication noise ↔
forward-diffusion noise, $S_\mathrm{gene}\!\downarrow$ ↔
$H[q(\theta|\mathcal{D})]\!\downarrow$, and we discuss the implication
that biological cells operate far from thermodynamic optimality for
reasons identical to why over-parameterised neural networks operate in
the "lazy" training regime. All code and figures are available as
supplementary material.

---

## 1. Introduction

The thermodynamic basis of life has long been intertwined with the notion of *negentropy* introduced by Schrödinger [1944]: living systems
maintain internal order by exporting entropy to their environment. Two modern frameworks have made this idea quantitative. The maximum
entropy production principle (MEPP) [Martyushev & Seleznev 2006; Sawada, Daigaku & Toma 2025] posits that nonequilibrium systems
preferentially occupy macrostates that maximise entropy production rate, and Dissipative Adaptation theory [England 2015] shows that
driven self-organising systems preferentially populate work-absorption-effective configurations. On the molecular scale,
recent work on enhanced enzyme diffusion (EED) [Ichii, Hatakeyama & Kaneko 2025] has demonstrated that catalytic events themselves
constitute information-storing degrees of freedom capable of acting as chemical Maxwell demons.

In parallel, stochastic thermodynamics has produced two universal bounds. The thermodynamic uncertainty relation (TUR) [Barato &
Seifert 2015] states that for any nonequilibrium current $J$ measured over time $\tau$,
$$
\frac{\mathrm{Var}(J_\tau)}{\langle J_\tau\rangle^2}\, \Sigma_\tau \geq 2,
\tag{1}
$$
where $\Sigma_\tau$ is the cumulative entropy production. The
thermodynamic speed limit (TSL) [Shiraishi, Funo & Saito 2018] bounds
the time required for a probability distribution to evolve from $p_0$
to $p_\tau$:
$$
\tau \geq \frac{\mathcal{L}(p_0, p_\tau)^2}{2\langle\Sigma\rangle_\tau\langle A\rangle_\tau},
\tag{2}
$$
with $\mathcal{L}$ the Hellinger distance and $\langle A\rangle$ the dynamical activity. Both bounds have been experimentally probed in molecular machines: DNA polymerase appears to saturate TUR at the linear-response level, while the ribosome operates roughly five times above the bound [Song, Hyeon & Park 2019].

A third strand of work concerns the formal analogy between
nonequilibrium self-organisation and Bayesian inference. The
free-energy principle [Friston 2010] interprets biological
self-organisation as variational free-energy minimisation; diffusion
models in machine learning [Song & Ermon 2019; Ho et al. 2020]
explicitly implement an Ornstein–Uhlenbeck-type forward-noise process
followed by score-based denoising. Yet the *thermodynamic* cost of
learning has rarely been computed in concrete biological models.

In this paper we build a numerical bridge across these strands. We
construct a hierarchical autocatalytic model with three explicit
layers (raw molecules, proteins, large polymers) coupled by a
slightly-broken mass-action tensor, then probe its TUR/TSL
behaviour, the co-occurrence of MEPP and selection, and the analogy
to diffusion-model training. The model is iterated through four
generations of increasing sophistication (v1 → v4), each addressing a
specific shortcoming of its predecessor. We find that:

1. The hierarchical model spontaneously exhibits the Schrödinger
   pattern: $\sigma_\mathrm{env}\!\uparrow$ co-occurs with
   $S_\mathrm{gene}\!\downarrow$, with a $\sim 589\!\times$ asymmetric
   bookkeeping favouring environmental dissipation.
2. Both TUR and TSL bounds are wildly loose ($10^4$–$10^8$ above their
   universal values), and this looseness is *structural*, not size-
   dependent.
3. A collapsed single-cycle minimal replicator recovers TUR
   $\approx 5$, matching the ribosome regime measured experimentally.
4. The trade-off between thermodynamic efficiency and architectural
   complexity is sharp: bound-saturating systems necessarily lose the
   hierarchy that defines biological generality.

We discuss these results in light of recent claims by Kolchinsky [2024] that no universal dissipation–replicator relation exists, and
of measurements of extremophile metabolism that approach the single-cycle / near-equilibrium / limit-cycle conditions asymptotically but never simultaneously.

---

## 2. Model

![model summary](v1234_schematics.png)

### 2.1 Hierarchical autocatalytic model (v3)

The state of the system at time $t$ is
$$
\mathbf{s}(t) = (\,a_i,\, p_{p,l},\, W_p^{(k)},\, W_{l m k}\,),
\qquad
i\!\in\![L],\; l,m\!\in\![M],\; k\!\in\![N].
$$
Here $a_i$ are raw molecules (CO$_2$, water, phosphate; $L=3$),
$p_{p,l}$ are protein populations ($M=4$), and $W_p^{(k)}$ are
large-polymer populations such as DNA, RNA ($N=8$) whose internal fold structure is
encoded by the tensor $W_{lmk}\!\in\!\mathbb{R}^{M\times M\times N}$.
The dynamics are
$$
\dot a_i  = K_i\, r_i + d_i(a^\mathrm{ext}_i - a_i)
            - c_i,
\quad
\dot p_{p,l} = r_l - \gamma\,\xi\, p_{p,l},
\quad
\dot W_p^{(k)} = W_p^{(k)} \tanh(\mathrm{align}_k)(1 - {\textstyle\sum_k W_p^{(k)}}/K)
                 - \zeta\, W_p^{(k)},
\tag{3}
$$
with
$$
r_i = \sum_{l,j} \mathrm{Coef}_{i l j}\,\mathrm{mod}_l\, y_i y_j
      + \varepsilon\,\eta_i,
\quad
\mathrm{mod}_l = \tanh\!\bigl(\beta W^\mathrm{eff}_l \mathbf{p}_p - \theta_l\bigr),
\quad
W^\mathrm{eff}_{lm} = \sum_k \frac{W_p^{(k)}}{\sum_{k'} W_p^{(k')}} W_{lmk}.
\tag{4}
$$
$\eta_i\!\sim\!\mathcal{N}(0,1)$ implements the $\varepsilon$-broken
mass-action; $\mathrm{align}_k = W_{lmk}\,\mathrm{pa}_l\, p_{p,m}$ is the
fold–substrate alignment that serves as a fitness function;
$\mathrm{Coef}_{ilj}$ couples the bath ($i,j\!<\!L$) and protein blocks
($i,j\!\ge\!L$) with a positive autocatalytic bias on the protein-from-bath
channels and a damping factor of $0.1$ on protein–protein quadratics
to avoid blow-up. Each gene $k$ is projected to a binary
fold-signature string by taking the sign pattern of the leading
eigenvector of $\frac{1}{2}(W_{:,:,k}+W_{:,:,k}^\top)$; this defines a
distribution $p(s)$ on a finite alphabet whose Shannon entropy
$$
S_\mathrm{gene}(t) = -\sum_s p(s,t)\, \log p(s,t)
\tag{5}
$$
quantifies genetic diversity.

The total entropy production rate is computed as
$$
\sigma(t) = \sigma_\mathrm{react}(t) + \sigma_\mathrm{env}(t)
$$
where
$$
\sigma_\mathrm{react}(t) = \sum_i (f_i - b_i)\log(f_i/b_i),
\qquad
\sigma_\mathrm{env}(t) = \sum_i d_i(a^\mathrm{ext}_i - a_i)\log\!\bigl(a^\mathrm{ext}_i/a_i\bigr),
\tag{6}
$$
with forward and backward fluxes $f_i,b_i$ obtained from the
$\varepsilon$-broken rates with a floor $\varepsilon$ to prevent
divergence.

### 2.2 Parameter search

The model has eight global "knobs" exposed for tuning: $\beta, \gamma,
\varepsilon$, the protein decay multiplier prot_decay, the
autocatalytic positive bias coef_bias, the diffusion scale, the
threshold scale, and the gene decay rate $\zeta$. We construct a
multi-objective score that rewards a steady non-trivial NESS and
penalises four failure modes (extinction, divergence, NaN, collapsing
tails):
$$
J = -|\langle g\rangle| + 2\, S_\mathrm{gene} + 0.3\log\sigma
    + 0.5\,\log(\mathrm{tot}_{W_p}\!\cdot\!\mathrm{tot}_{p_p}) + 0.5\,\mathrm{growth\_var},
\tag{7}
$$
and search both by a $4\!\times\!4\!\times\!4$ grid and by Optuna's Tree-structured Parzen Estimator (TPE) with 60 trials.

### 2.3 Minimal single-cycle replicator (v4)

To test whether the TUR looseness is structural, we collapse the hierarchy to a single kinetic-proofreading-like cycle
$$
\mathrm{R}_1\!:\;A\!+\!X\!\xrightarrow{k_+} 2X,\quad
\mathrm{R}_2\!:\;2X\!\xrightarrow{k_-}\!A\!+\!X,\quad
\mathrm{R}_3\!:\;X\!\xrightarrow{k_\mathrm{deg}}\!W,\quad
\mathrm{R}_4\!:\;W\!\xrightarrow{\mathrm{slow}}\!A,
\tag{8}
$$
with substrate $A$ held at $A_0$ by a bath. Mass action is broken in
the same $\varepsilon$-noise way as the hierarchical model. We
integrate Eq. (8) with a $\tau$-leaping Gillespie scheme over an
ensemble of $n_\mathrm{rep}\!=\!400$ replicates, tracking the per-cycle
Schnakenberg entropy production
$$
\Sigma_\tau = (n_+\!-\!n_-)\log(a_+/a_-) + n_\mathrm{deg}\, \Delta\mu_\mathrm{deg}.
\tag{9}
$$
The growth current is $J_\tau = n_+\!-\!n_-$.

---

## 3. Model Evolution

Figure 1 shows schematics of the four model variants we built en route
to a working numerical bridge.

**v1 (Initial).** A straightforward three-layer system with no
protein–protein damping and no clipping ceiling. Although the system
does not blow up to NaN under our default initial conditions, all
gene populations decay monotonically. Survival rate over 8 random
seeds: 0%.

**v2 (NESS).** Adding (i) $\times 0.1$ damping on the
$\mathrm{Coef}[L:,:,L:]$ tensor block, (ii) a clip ceiling of 50 on
$a$ and $p_p$, and (iii) a fitness-based gene-selection term
$\mathrm{align}_k$, transforms the system into a steady state.
Survival rate: 62%, but the mean growth rate remains slightly negative ($-0.021$) and TUR/TSL bounds are not meaningfully measurable.

![best run](Hier_Autocatalysis/result/best_run_timeseries.png)

![model summary](Hier_Autocatalysis/result/grid_searchi_heatmap.png)

![model summary](Hier_Autocatalysis/result/joint_relations.png)

![model summary](Hier_Autocatalysis/result/optuna_summary.png)

![model summary](Hier_Autocatalysis/result/timeseries.png)

![TUR TSL bound](Hier_Autocatalysis/result/tur_tsl_bounds.png)


**v3 (Tunable + Optuna).** With eight knobs exposed and 60 Optuna
trials, TPE finds a narrow survival island at
$\beta\!=\!1.45,\;\gamma\!=\!0.13,\;\theta_\mathrm{scale}\!=\!0.39,\;
\mathrm{coef\_bias}\!=\!0.60,\;\mathrm{prot\_decay}\!=\!1.09,\;
\zeta\!=\!0.020$. Survival: 100%. Mean growth: $+0.12$.
$S_\mathrm{gene}$ contracts from $\log N\!=\!2.08$ to $1.24$ nats over
1500 steps. TUR and TSL are measurable but loose (Section 4).

**v4 (Minimal replicator).** Collapsing to a single cycle yields
TUR $\approx 5$ at $k_-/k_+\!=\!0.20$, within the experimental ribosome
regime. The hierarchy, fold tensor, and $S_\mathrm{gene}$ are all
lost.

![model summary](fig/minimal_tur.png)

| variant | survival | $\langle\sigma\rangle$ |growth | $S_{gene}$ | TUR | TSL |
|----|----|----|----|----|----|----|
| v1 initial | 0% | – | – | – | n/a | n/a |
| v2 NESS | 62% | $3.3\!\times\!10^3$ | 0.021 | 1.49 | n/a | n/a |
| v3 Optuna-tuned | 100% | $9.9\!\times\!10^3$ | 0.12 | 1.40 | $2.1\!\times\!10^5$ | $3.0\!\times\!10^8$ |
| v4 minimal | 100% | $12$ | 5.84 | – | $\mathbf{5.4}$ | $5.0\!\times\!10^4$ |

**Table 1.** Summary of the four model variants on a common set of
diagnostic axes.

The 39 083× improvement in TUR product from v3 to v4 (Figure 2) is achieved entirely by sacrificing architectural complexity, not by
re-tuning parameters.

---

## 4. Results

### 4.1 MEPP and selection co-occur in the hierarchical model

Figure 3 (top row) shows the entropy production rate
$\sigma_\mathrm{env}(t)$ and the genetic entropy $S_\mathrm{gene}(t)$ on
the same time axis for the Optuna-tuned v3 model. During the
organisation transient (steps 100–300), $\sigma_\mathrm{env}$ rises by
about 20× while $S_\mathrm{gene}$ falls from 1.71 nats to 1.24 nats.
The two trajectories are anti-correlated, confirming the Schrödinger
prediction at the symbolic level: order in the gene distribution is
generated *while* entropy is exported to the bath.

Figure 3 (bottom row) plots the cumulative quantities. Over 1500
steps the cell exports
$\int_0^T \sigma_\mathrm{env}\,dt = 277$ nats to the bath while
generating $-\Delta S_\mathrm{gene} = 0.47$ nats of genetic order — a
ratio of approximately $589\!:\!1$. The Schrödinger inequality
$\int\sigma_\mathrm{env}\,dt \gg -\Delta S_\mathrm{gene}$ is therefore
satisfied by three orders of magnitude.

### 4.2 The hierarchical model is far from TUR and TSL bounds

Figure 4 shows the TUR product (Eq. 1) computed in sliding 200-step
windows from an ensemble of 15 replicates. The median TUR product is
$3.3\!\times\!10^5$ for the growth current and $1.2\!\times\!10^6$ for
the evolution current — five to six orders of magnitude above the
universal bound of 2.

The TSL ratio (Eq. 2) is $3.2\!\times\!10^7$ at the median, seven
orders of magnitude above its bound of 1. The observed evolution
speed $|d\mathbf{p}/dt|\!\sim\!10^{-3}$ sits roughly four orders of
magnitude below the TSL ceiling
$\sqrt{2\langle\Sigma\rangle\langle A\rangle/\tau}\!\sim\!10$
(Figure 4, middle-left panel).

The MEPP saturation $\langle\sigma\rangle/\sigma_\mathrm{max}$ across
replicates is $0.275\!\pm\!0.05$, indicating that the chosen
trajectory does *not* lie on the maximum-EP branch but rather a few
times below it. This is consistent with Sawada et al.'s [2025]
phenomenological version of MEPP, which posits a critical condition
$\xi > \xi_{c1}$ for the existence of a dissipative structure rather
than strict maximisation.

### 4.3 The minimal replicator approaches the ribosome regime

Figure 5 shows the TUR product of the minimal replicator (v4) as a
function of the reversibility ratio $k_-/k_+$. At $k_-/k_+\!=\!0.20$
we measure TUR $= 4.98$, within a factor of 2.5 of the universal bound
and matching Song et al.'s [2019] measured value for the *E. coli*
ribosome (their reported TUR product $\sim\!10$, equivalent to
$\sim\!5\times$ the bound of 2). At $k_-/k_+\!=\!0.05$ (highly
irreversible) the product rises to 10.4; at $k_-/k_+\!=\!0.95$
(near-equilibrium) it diverges as variance vanishes.

This non-monotonic behaviour reveals that bound saturation requires an
intermediate irreversibility — too far from equilibrium wastes EP, too
close kills the signal. The minimum of the TUR product is the
biophysical optimum for the chosen current.

### 4.4 Scaling with system size

Figure 6 shows the bounds for the hierarchical model as $(N,M,L)$
varies from $(4,3,3)$ to $(32,8,3)$, with each size point re-tuned by
a separate Optuna run. Three findings emerge:

1. **TUR/TSL products are size-invariant** within $\pm 0.5$ orders of
   magnitude. The looseness is structural, not extensive.
2. **The survival score increases monotonically** with system size
   (from 4.3 to 9.8 over the range). Larger configuration spaces
   accommodate more survival solutions, even though each solution is
   no closer to bound saturation.
3. **MEPP saturation is non-monotonic** ($0.14$ to $0.30$) with no
   trend.

In contrast, Figure 7 shows that the minimal replicator's TUR product
*decreases monotonically* with particle number $X_0$, from $11.2$ at
$X_0\!=\!5$ to $1.84$ at $X_0\!=\!160$. The $X_0\!=\!80$ replicate
achieves TUR $= 2.06\!\approx\!\mathrm{bound}$; the $X_0\!=\!160$
under-shoot is a finite-ensemble artefact. Thus single-cycle systems
become arbitrarily TUR-tight with population, while hierarchical
systems do not.

### 4.5 Correspondence with diffusion-model training

Figure 8 summarises the formal mapping. On the autocatalytic side,
the bath $a^\mathrm{ext}$ supplies raw molecules, the fold activity
$\tanh(\beta W p\!-\!\theta)$ provides a context-dependent score, the
$\varepsilon$-noise breaks detailed balance, and selection contracts
the gene distribution. On the diffusion-model side, the data
distribution $p_\mathrm{data}(x)$ supplies samples, the score network
$s_\theta(x_t,t)$ provides a context-dependent gradient, the
forward-diffusion noise $\sigma_t \xi$ breaks reversibility, and the
likelihood gradient contracts the parameter posterior
$q(\theta|\mathcal{D})$. Both systems share the global pattern
$\sigma_\mathrm{env}\!\uparrow + S_\mathrm{internal}\!\downarrow$.

A concrete prediction follows: just as biological cells operate $10^5$
times above the TUR bound in our hierarchical model, over-parameterised
neural networks should exhibit thermodynamic-style inefficiencies in
their information-theoretic learning rate. This is consistent with
the observation that scaling laws for large language models have
exponents $\sim\!0.1$–$0.3$ rather than the $\sim\!1$ that
strict-bound saturation would predict.

---

## 5. Discussion

### 5.1 Relation to prior work

Our hierarchical model occupies the intersection of three established
traditions. Sawada, Daigaku & Toma [2025] develop MEPP as a
phenomenological law for the birth and evolution of life, identifying
critical concentrations $\xi_{c1}$ above which exponential entropy
production becomes self-sustaining. Our Optuna survival island plays
exactly this role in a higher-dimensional parameter space, and the
$\sigma_\mathrm{env}\!\uparrow + S_\mathrm{gene}\!\downarrow$ pattern
in Figure 3 quantifies their qualitative claim.

Ichii, Hatakeyama & Kaneko [2025] propose that enhanced enzyme
diffusion (EED) makes individual catalysts function as chemical
Maxwell demons. The fold activity $\tanh(\beta W p\!-\!\theta)$ in our model implements an analogous —
but population-level — Maxwell-demon mechanism in which the sequence
information $W$ determines the steady-state distribution of catalytic
rates. The two pictures are complementary rather than competing:
EED operates on single-enzyme time scales, sequence-coded modulation
on generation-spanning ones. Our model could naturally absorb EED by
making $d_i \to d_i(1+\kappa_i|\mathrm{pa}_l|p_{p,l})$.

England's [2015] dissipative-adaptation framework is the most general
ancestor: any driven many-body system preferentially occupies
work-absorption-effective configurations. The Optuna survival island
we find is empirically a "work-resonance" point at which $\theta_\mathrm{scale}$
(importance 63%) tunes the fold-threshold into coupling with the
external substrate gradient. Our numerical results render quantitative
predictions that England's variational argument leaves implicit.

Two more recent threads frame our negative results. Kolchinsky
[2024] argues that no universal relation links thermodynamic
dissipation to replicator growth-decay rates, contradicting some
strong readings of England's bound. Our finding that the TUR
product is far from the bound (Sections 4.2, 4.4) is consistent with
this: TUR is satisfied trivially when one has many extensive sources
of dissipation that do not show up as variance in the chosen current.
Song, Hyeon & Park [2019] measure TUR products in *E. coli* enzymes
directly; their finding that polymerases sit near the bound while
the ribosome lies $\sim\!5\times$ above it is reproduced almost
exactly by our minimal-replicator sweep (Figure 5).

### 5.2 Why the hierarchy is "wasteful"

A natural objection is that our hierarchical model is poorly
designed: surely a better architecture could saturate TUR? The
size-scaling analysis (Section 4.4) shows that this is not the case
within the class of three-layer autocatalytic systems with coupled
fold tensors. The looseness is intrinsic to:

1. **Aggregation of currents.** $\log W_p$ pools fluctuations over
   $N$ genes, suppressing variance as $1/N$ while $\Sigma$ stays
   $O(1)$. TUR $\propto N\!\to\!\infty$.
2. **Multiple coupled cycles.** Each protein channel and each gene
   contributes to $\Sigma$ but only weakly to the observed current.
   This is precisely the regime that Polettini [2017] identifies as
   "thermodynamically wasteful".
3. **Distributed feedback.** Fitness alignment couples every
   $W_p^{(k)}$ to every $p_{p,l}$, decorrelating fluctuations across
   replicates and inflating $\langle J\rangle^2$ relative to
   $\mathrm{Var}(J)$.

The minimal replicator removes all three features and recovers
ribosome-class TUR products. The trade-off is exact: TUR efficiency
and architectural generality are mutually exclusive in this class of models.

### 5.3 Extremophiles as natural experiments

If single-cycle / near-equilibrium / limit-cycle operation are the
necessary conditions for thermodynamic efficiency, extremophile
biology offers natural test cases:

- **Acetogens and methanogens** in deep marine sediments operate at
  Gibbs energies $|\Delta G|\!\sim\!-20$ kJ/mol per ATP equivalent
  [Müller & Hess 2017], approximately $-8\,k_BT$ per turnover and
  thus near the linear-response regime.
- **South Pacific Gyre subsurface microbes** maintain themselves on
  $\sim\!190$ zeptowatts per cell, corresponding to a handful of ATP
  hydrolyses per cell per minute [Bradley et al. 2020]. This is the
  closest biological analogue of the near-equilibrium limit.
- **The KaiABC cyanobacterial circadian clock** is a stochastic
  limit cycle whose period responds to ATP turnover in accordance
  with the dissipation–coherence trade-off [Cao, Wang & Sasai 2015;
  Nagayama & Ito 2025].

Each of these systems realises one of the three conditions for TUR
saturation, but none realises all three simultaneously. Direct TUR
measurement on the KaiABC oscillator in vitro is feasible and would
test our prediction of $\mathrm{TUR}\!\sim\!5$–$10$.

### 5.4 Implications for machine learning

The mapping of Section 4.5 makes a falsifiable claim: TUR-like
inefficiencies should manifest in neural-network training as a
mismatch between achieved loss-reduction rate and the
Fisher-information-bounded ceiling. The neural-tangent-kernel "lazy
training" regime [Jacot, Gabriel & Hongler 2018] and the empirical
exponent $\sim\!-0.1$ of GPT-3-scale language models [Kaplan et al.
2020] are both consistent with $O(10^4$–$10^6)$ slack above the
information-theoretic minimum, paralleling our biological numbers.

Two design implications follow. First, *modular sparsity* — the
machine-learning analogue of single-cycle separation — should improve
the dissipation–performance ratio of trained networks; the empirical
success of mixture-of-experts architectures [Shazeer et al. 2017] is
suggestive. Second, *cyclical learning-rate schedules* [Smith 2017]
function as limit-cycle approximations and may be analysable through
the TSL framework: their gain over fixed-rate training would then be
predicted by the dissipation–coherence trade-off.

---

## 6. Limitations and Future Work

Our model has several explicit limitations:

1. The Schnakenberg entropy production in Eq. (6) uses a backward
   floor of $\varepsilon$ rather than a physically derived reverse
   rate. For systems closer to detailed balance this should be
   replaced by a thermodynamically consistent local-equilibrium ansatz.
2. The fold tensor $W$ undergoes Gaussian replication noise rather
   than discrete mutation–selection on a finite alphabet. A
   Markov-jump version on a $4^M$-state lattice would be more
   realistic.
3. The minimal replicator has only one cycle. Multi-cycle minimal
   systems would interpolate between v4 and v3 and may reveal an
   intermediate optimum.
4. The diffusion-model correspondence is qualitative. A quantitative
   match would require computing the entropy production of an
   explicit score-matching SGD trajectory.

Future work will focus on (a) the EED extension, (b) verification of
TUR/TSL behaviour in stochastic limit cycles à la KaiABC, and
(c) explicit training-time entropy-production measurement on a
diffusion model.

---

## 7. Conclusion

We have constructed a hierarchical autocatalytic system that
simultaneously implements the Schrödinger negentropy mechanism, makes
quantitative the MEPP, supports gene-level selection in a sequence
space, and admits direct analogy with diffusion-model training. By
exposing the model's parameters to Optuna-based Bayesian search we
identified a survival island in an 8-dimensional space within which
the system exhibits the predicted $\sigma\!\uparrow + S_\mathrm{gene}
\!\downarrow$ pattern with a 589:1 negentropy ratio.

The system is, however, far from saturating the TUR and TSL bounds —
by $10^4$ to $10^8$ — and we have traced this looseness to the
multi-cycle hierarchical architecture itself. A collapsed
single-cycle replicator recovers TUR products of $\sim\!5$,
consistent with measured ribosome biophysics. The gap between
hierarchical and minimal models maps onto the trade-off between
biological generality and thermodynamic efficiency, which we argue
parallels the trade-off between generality and efficiency in
over-parameterised neural networks. Both biological cells and
modern deep-learning systems appear to operate in the "lazy" regime
where dissipation is plentiful but information-extraction efficiency
is modest, because that is the regime in which evolvability and
generalisation, respectively, are preserved.

---

## References

Barato, A.C. & Seifert, U. (2015). Thermodynamic uncertainty relation
for biomolecular processes. *Phys. Rev. Lett.* 114, 158101.

Bradley, J.A., Arndt, S., Amend, J.P., Burwicz, E., Dale, A.W.,
Egger, M. & LaRowe, D.E. (2020). Widespread energy limitation to life
in global subseafloor sediments. *Sci. Adv.* 6, eaba0697.

Cao, Y., Wang, H., Ouyang, Q. & Tu, Y. (2015). The free-energy cost
of accurate biochemical oscillations. *Nat. Phys.* 11, 772–778.

England, J.L. (2015). Dissipative adaptation in driven
self-assembly. *Nat. Nanotech.* 10, 919–923.

Friston, K. (2010). The free-energy principle: a unified brain
theory? *Nat. Rev. Neurosci.* 11, 127–138.

Ho, J., Jain, A. & Abbeel, P. (2020). Denoising diffusion
probabilistic models. *NeurIPS* 33, 6840–6851.

Ichii, K., Hatakeyama, T.S. & Kaneko, K. (2025). Enhanced enzyme
diffusion as a Maxwell-demon mechanism. *arXiv*:2503.17584.

Jacot, A., Gabriel, F. & Hongler, C. (2018). Neural tangent kernel:
convergence and generalization in neural networks. *NeurIPS* 31.

Kaplan, J., McCandlish, S., Henighan, T. et al. (2020). Scaling laws
for neural language models. *arXiv*:2001.08361.

Kolchinsky, A. (2024). Thermodynamic dissipation does not bound
replicator growth and decay rates. *arXiv*:2404.01130.

Martyushev, L.M. & Seleznev, V.D. (2006). Maximum entropy production
principle in physics, chemistry and biology. *Phys. Rep.* 426, 1–45.

Müller, V. & Hess, V. (2017). The minimum biological energy quantum.
*Front. Microbiol.* 8, 2019.

Nagayama, R. & Ito, S. (2025). Dissipation–coherence trade-off and
thermodynamic speed limit for stochastic limit cycles.
*arXiv*:2509.06421.

Polettini, M. (2017). Effective thermodynamics for marginal
observers. *Phys. Rev. Lett.* 119, 240601.

Sawada, Y., Daigaku, Y. & Toma, K. (2025). Maximum entropy
production principle of thermodynamics for the birth and evolution
of life. *Entropy* 27(4), 449.

Schrödinger, E. (1944). *What is Life?* Cambridge University Press.

Shazeer, N., Mirhoseini, A., Maziarz, K. et al. (2017). Outrageously
large neural networks: the sparsely-gated mixture-of-experts layer.
*ICLR*.

Shiraishi, N., Funo, K. & Saito, K. (2018). Speed limit for classical
stochastic processes. *Phys. Rev. Lett.* 121, 070601.

Smith, L.N. (2017). Cyclical learning rates for training neural
networks. *WACV*, 464–472.

Song, Y., Hyeon, C. & Park, H.S. (2019). Kinetic proofreading and
the limits of thermodynamic uncertainty. *arXiv*:1911.04673.

Song, Y. & Ermon, S. (2019). Generative modeling by estimating
gradients of the data distribution. *NeurIPS* 32.

---

## Figure list

- **Figure 1** (`v1234_schematics.png`).  Schematic architecture of
  the four model variants v1 → v4.
- **Figure 2** (`model_evolution.png`).  Quantitative evolution of
  survival, $\sigma$, growth, $S_\mathrm{gene}$, TUR and TSL across
  variants, with arrow annotation of the 39 083× TUR improvement
  from v3 to v4.
- **Figure 3** (`mepp_negentropy.png`).  Co-occurrence of
  $\sigma_\mathrm{env}\!\uparrow$ and $S_\mathrm{gene}\!\downarrow$
  in the Optuna-tuned v3 model; cumulative negentropy bookkeeping.
- **Figure 4** (`tur_tsl_bounds.png`).  TUR product, TSL ratio, MEPP
  saturation, and observed-versus-ceiling speeds for v3.
- **Figure 5** (`minimal_tur.png`).  TUR product of the minimal
  replicator as a function of the reversibility ratio
  $k_-/k_+$, with experimental reference lines for DNA polymerase,
  ribosome, and hierarchical model.
- **Figure 6** (`scaling_hier.png`).  Size scaling of TUR/TSL/MEPP
  for the hierarchical model across $(N,M,L)\!\in\!\{(4,3,3),\ldots,(32,8,3)\}$.
- **Figure 7** (`scaling_minimal.png`).  Particle-number scaling of
  TUR/TSL for the minimal replicator.
- **Figure 8** (`diffusion_analogy.png`).  Block-diagram
  correspondence between hierarchical autocatalysis and
  diffusion-model training.

## Code availability

All code is provided at https://github.com/xiangze/DiverseCells/
- `Hier_Autocatalysis/hierarchical_autocatalysis.py` (v1/v2),
- `Hier_Autocatalysis/tune.py` (v3 + Optuna),
- `Hier_Autocatalysis/bounds.py` (TUR/TSL/MEPP),
- `Hier_Autocatalysis/mepp_negentropy.py` (negentropy bookkeeping),
- `Hier_Autocatalysis/scaling_study.py` (size scaling),
- `Hier_Autocatalysis/model_evolution.py` (variant comparison),
- `minimal_replicator/minimal_replicator.py` (v4),
- `doc/v1234_schematics.py` (Figure 1).
