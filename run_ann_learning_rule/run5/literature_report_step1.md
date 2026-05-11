---
title: "Biologically Plausible Learning Rules for Recurrent Neural Networks: A Structured Survey"
bibliography: references.bib
---

# Scope

This report surveys learning rules for recurrent neural networks (RNNs) that relax one or more biological implausibilities of backpropagation-through-time (BPTT): weight transport, non-locality, non-causality, offline operation, and separate forward/backward phases. It is organized into six clusters of mechanisms, each evaluated against five operational desiderata:

- **WT**: no weight transport (forward and feedback pathways do not share weights).
- **Loc**: locality — Δw_{ij} depends only on pre/post activity and local modulators.
- **Onl**: online / bounded-memory per step (no O(T·N) activity history).
- **NoP**: no separate forward/backward phase.
- **Cau**: causal — updates at time t depend only on information at time ≤ t.

The report concludes with a gap analysis that will inform Step 2 of this run (novelty-informed design).

# 1. Forward-mode / online credit assignment

The oldest online RNN credit-assignment algorithm is RTRL (Williams & Zipser 1989): at every step one maintains the influence tensor ∂h_t/∂θ, yielding an exact, online gradient of the instantaneous loss. RTRL is causal and online but has O(N³) per-step cost and is non-local (each synapse requires every other synapse's full influence tensor), so it is biologically implausible at the scale required for any real task.

Approximations reduce RTRL's cost while retaining its online, causal structure:

- **UORO** (Unbiased Online Recurrent Optimization) [@tallec2018] replaces the influence tensor with a rank-1 random factorization, yielding an unbiased stochastic estimate at O(N²) per-step. However, the factorization is still non-local.
- **SnAp** (Sparse n-step Approximation) [@menick2021] keeps only sparse non-zero entries of the influence tensor corresponding to an n-step neighborhood graph, giving a biased but very cheap approximation that is competitive with truncated BPTT on language modeling.
- **RFLO** (Random-Feedback Local Online) [@murray2019] is the cleanest biologically plausible variant: each post-synaptic unit keeps a low-pass-filtered eligibility trace p_{ij} = α·p_{ij} + (1 − α)·φ'(u_i)·x_j of its pre-synaptic drive; output errors are broadcast back through a **fixed random** readout matrix to each unit; the weight update is Δw_{ij} = η·B_i·e·p_{ij}, where B is random feedback and e is the instantaneous readout error. RFLO satisfies WT, Loc, Onl, NoP, and Cau; the price is that its gradient estimate is biased relative to BPTT.
- **e-prop** [@bellec2020; @bellec2019] derived an eligibility-trace decomposition that is exact for the immediate-past contribution of each synapse to the recurrent unit's own output, but replaces the long-range temporal Jacobian with a learning signal. In spiking RNNs (LSNN / ALIF neurons) this yields BPTT-comparable performance on TIMIT and word-level language tasks. The update Δw_{ji} = η·L_j·e_{ji} is local given a global broadcast L_j.
- **e-prop generalizations**: [@millidge2025eprop] scales e-prop to deeper networks; [@liu2022beyond] analyzes generalization properties and loss-landscape geometry of BPTT-like vs. bio-plausible rules; [@liu2025eprop] tests whether e-prop matches BPTT's neural similarity to biology.
- **Real-time recurrent RL** [@lemmel2023] ports RTRL-style credit to actor–critic RL.

**Desiderata summary (forward-mode cluster):**

| Method | WT | Loc | Onl | NoP | Cau | Perf envelope |
|--------|----|-----|-----|-----|-----|-----|
| RTRL | ✗ | ✗ | ✓ | ✓ | ✓ | exact BPTT; intractable at scale |
| UORO | ✗ | ✗ | ✓ | ✓ | ✓ | Copy / language (small) |
| SnAp | ✗ | partial | ✓ | ✓ | ✓ | WikiText-style LMs, close to tBPTT |
| RFLO | ✓ | ✓ | ✓ | ✓ | ✓ | Short supervised tasks; noticeably below BPTT |
| e-prop | ✓ | ✓ | ✓ | ✓ | ✓ | BPTT-competitive on TIMIT / delayed tasks |

**Residual biological criticisms.** e-prop and RFLO still require a *global* low-dimensional error broadcast L_j or B_i·e. e-prop's eligibility trace is derived exactly from the recurrent unit's self-influence only, so any credit that flows through *other* neurons is dropped — on tasks with strong multi-neuron temporal chains this yields measurable gradient bias [@liu2022beyond; @liu2022modprop].

# 2. Random-feedback methods

Feedback alignment (FA) [@lillicrap2016] showed that replacing the transposed forward weight matrix in the backward pass with a **fixed random** matrix B still supports learning: during training, forward weights rotate into partial alignment with B, giving a useful (if biased) descent direction. FA removes weight transport but still requires a sequential backward pass.

Key variants and evaluations:

- **Direct Feedback Alignment (DFA)** [@nokland2016] bypasses the backward pass entirely: each hidden layer receives the output error through a fixed random projection directly, in parallel. DFA has been scaled to modern architectures [@launay2020] but underperforms FA/BP on convolutional/ImageNet-scale benchmarks [@bartunov2018].
- **Sign-concordant feedback** and **weight mirrors** [@akrout2019] learn the feedback weights to match the forward weights' *sign* (not magnitude), giving BP-like performance with fully biologically plausible local updates to the feedback pathway.
- **Fixed random learning signals** without any backward pass [@frenkel2021] train deep nets by broadcasting a global error through per-layer random projections, with a focus on on-chip learning.
- **Meta-learned plasticity with random feedback** [@shervani2022] treats the local plasticity rule's functional form as meta-learnable while keeping feedback weights random.
- **Alignment dynamics** [@refinetti2020] analyze when FA converges and when it fails, linking it to spectral properties of the feedback matrix.
- **Sparse DFA** [@crafton2019] tests whether sparse (hardware-friendly) random feedback still supports learning.
- **Reviews** [@illing2019] show that random-feedback-based rules scale weakly past shallow networks and that the alignment effect weakens with depth.

Extending random feedback to recurrent temporal credit is exactly what RFLO [@murray2019] and **ModProp** [@liu2022modprop] address:

- **ModProp** [@liu2022modprop] frames the long-range temporal Jacobian of the recurrent pre-activation as a sum of cell-type-specific modulatory signals broadcast by different neurochemical channels (dopamine, acetylcholine, serotonin, etc.). Each cell type receives a distinct multiplicative gain, enabling credit flow through multi-neuron chains *without* weight transport. Performance approaches BPTT on pattern-generation and psychophysics-style RNN tasks.
- **Adjoint propagation through modular RNNs** [@zhuoliu2026] (2026) derives a mathematically related formulation in which the error adjoint is split across sub-modules whose backward paths need not share forward weights.

**Desiderata summary (random-feedback cluster):**

| Method | WT | Loc | Onl (for RNNs) | NoP | Cau |
|--------|----|-----|----------------|-----|-----|
| FA | ✓ | partial | ✗ (needs BPTT in time) | ✗ | ✗ |
| DFA | ✓ | ✓ | depends | partial | ✗ |
| Sign-concordant | ✓ | ✓ | ✗ | ✗ | ✗ |
| RFLO | ✓ | ✓ | ✓ | ✓ | ✓ |
| ModProp | ✓ | ✓ | ✓ | ✓ | ✓ |

**Residual biological criticisms.** Even cell-type-specific modulation requires a central broadcaster; and the random feedback matrices themselves are biologically inert (no known plasticity mechanism that freezes them at a suitable random point in development). Meta-learned feedback [@shervani2022] begins to address this but at the cost of an outer BPTT meta-loop.

# 3. Predictive coding and target propagation applied to sequences

## 3a. Predictive coding (PC)

PC networks represent each layer's activity plus a prediction/error unit pair; inference iterates a free-energy minimization, and learning is a local Hebbian rule on the converged errors. [@whittington2017] showed PC approximates backpropagation in feedforward networks under certain limits; [@millidge2020pc] generalized this to arbitrary computation graphs, and [@song2020backprop] derived conditions (infinitesimal inference limit) under which PC is *equivalent* to BP. Reviews and extensions: [@whittington2019; @millidge2022pcreview; @salvatori2023; @millidge2022inference; @salvatori2021].

Temporal PC (PC-RNNs) extends this to recurrent networks: each layer predicts the next timestep's activity, and errors are local in both space and time. PC satisfies Loc and NoP (forward pass computes both prediction and error), but:

- PC is typically **not online** — inference iterates until convergence per example.
- PC still uses symmetric forward/backward weights in the standard derivation; breaking this requires FA-style modifications.
- The correspondence between PC and BPTT along time is delicate; convergence is required per timestep.

## 3b. Target propagation (TP)

TP [@lee2015dtp] replaces the error gradient with a *target* — a desired activity pattern — propagated backward through a learned inverse of the forward map. Difference TP adds a correction term; [@meulemans2020] gives a theoretical framework and fixes the noise amplification of the original. [@ernoult2022dtp] scales DTP by learning the targets so that their gradient aligns with BP. Biological motivations [@ororbia2018; @ahmad2020gaitprop] further localize the target computation.

For RNNs, TP is straightforward in principle (target at time t propagated to time t−1 through a learned inverse) but suffers from error accumulation over long horizons and the need to train the inverse network itself.

**Desiderata summary.** PC: WT (depends on variant), Loc ✓, Onl ✗, NoP ✓, Cau ✗ (iterates bidirectionally). TP: WT ✓ (no gradient transport), Loc ✓, Onl rarely, NoP ✗ (two phases), Cau ✗.

# 4. Dendritic / multi-compartment learning rules

Multi-compartment neuron models use distinct dendritic and somatic compartments to carry forward activity and "teaching" signals on different membrane sites, replacing the need for a separate backward phase.

- **Segregated dendrites** [@guerguiev2017]: apical dendrites carry top-down / teaching signals; basal dendrites carry bottom-up sensory drive; somatic plasticity uses the local difference.
- **Dendritic cortical microcircuits** [@sacramento2018] approximate BP by running apical and basal integration concurrently, with interneurons implementing subtractive error terms; learning is a local Hebbian rule on the apical compartment.
- **Burstprop** [@payeur2021] encodes error in the *fraction of bursts* at each neuron, multiplexing feedforward (firing rate) and feedback (burst fraction) signals on one spike train; plasticity is triggered by burst-dependent co-occurrence.
- **Single-phase deep learning** [@greedy2022] runs a single forward/backward pass in cortico-cortical networks without a nominal teaching phase, closing one gap left by [@sacramento2018].
- **Real-time backprojections** [@max2022] and **Deep Feedback Control** [@meulemans2021dfc] close the loop biologically by using controlled feedback to drive the teaching signal in real time.
- **Biology-constrained deep learning** [@galloni2025] layers compartmentalization, Dale's law, and cell-type specificity into deep networks.

**Desiderata summary.** Most multi-compartment rules satisfy WT, Loc, NoP, and (some) Cau, but **Onl is partial**: many (Sacramento, Guerguiev, Payeur) were derived for *static* inputs with a teaching phase. Extending to temporally-varying RNN inputs remains open [@galloni2025].

# 5. Synaptic tagging and capture / two-timescale consolidation

A separate line of bio-plausible learning concerns not the *direction* of the credit signal but its *temporal integration*: which synaptic changes survive, and over what timescales. The synaptic tagging and capture (STC) hypothesis [@clopath2008; @ziegler2015; @luboeinski2020] posits that recent activity sets a fast, decaying "tag" at each synapse, and a later modulatory event (a dopaminergic burst, a novelty signal) captures tagged synapses into a long-lasting change. This is an explicit mechanism for bridging a long delay between activity and reinforcement.

Cascade and metaplasticity models formalize how many internal states a synapse can occupy and how long it can remember [@benna2016]: a geometrically-spaced cascade of variables gives O(log T) memory with a fixed per-step local update. [@leimer2019] uses decay with selective consolidation to enable fast learning without catastrophic forgetting. [@zenke2024] reviews the link between synaptic consolidation and continual-learning algorithms.

STC in RNN learning is mostly qualitative today: Luboeinski & Tetzlaff [@luboeinski2020] explore consolidation effects in generic recurrent circuits. STC has not been paired with gradient-like temporal credit assignment — a gap.

**Desiderata summary.** STC naturally satisfies Loc, Onl, NoP, and Cau; WT is orthogonal (STC prescribes the *update scheduling*, not the *update direction*).

# 6. Node-perturbation / stochastic RNN learning

Node- or weight-perturbation rules estimate the gradient by correlating a noisy perturbation of neural activity with the resulting change in loss. Applied to RNNs, they are exact, online, local, causal, and phase-less, but have extremely high variance that scales with network size.

- **Reward-modulated Hebbian learning with node perturbation** forms the basis of Miconi 2017 [@miconi2017]: each recurrent unit's pre-activation gets injected noise ξ_i; each synapse accumulates a local eligibility term ξ_i·r_j; a delayed global reward signal R modulates the eligibility to produce Δw_{ij} = η·R·Σ_t ξ_i·r_j. This rule trains RNNs to perform delayed-match-to-sample and comparable cognitive tasks while reproducing cortical dynamical signatures.
- **Differentiable plasticity** [@miconi2018diffplasticity] and **Backpropamine** [@miconi2018backpropamine] meta-learn the plasticity rule itself by BPTT, effectively turning the form of Δw into a differentiable module. These are not biologically plausible at training time but inform which functional forms *can* be discovered.
- **Noise-based reward-modulated learning** [@fernandez2025] revisits the node-perturbation family with modern variance-reduction tricks.

**Desiderata summary.** WT ✓, Loc ✓, Onl ✓, NoP ✓, Cau ✓. Performance envelope: small RNNs on cognitive tasks of sequence length ~10–100. On standard ML benchmarks (sMNIST, PTB), node perturbation scales poorly compared to BPTT and e-prop.

# 7. Cross-cutting: three-factor rules, synthetic gradients, equilibrium propagation

## 7a. Three-factor rules and neuromodulation

Most of the "locally plausible" rules above factor as pre × post × modulator [@gerstner2018; @fremaux2016; @zenke2017] — a three-factor plasticity rule. The modulator carries the global error / reward / gate, and its cell-type specificity [@liu2022modprop] is what lets it approximate multi-step temporal credit. [@schmidgall2023] gives a broad review. This is the *dominant framing* in which any new rule will live.

## 7b. Synthetic gradients

Synthetic gradients [@jaderberg2016; @czarnecki2017] train a small auxiliary network that **predicts the gradient**, eliminating the need to wait for a full backward pass and enabling asynchronous updates. This is a natural home for biologically plausible learning: the synthetic-gradient network can be interpreted as a local predictive-coding error generator. Relatively little work has merged synthetic gradients with explicit biological plausibility — [@assaquib2025; @kohan2022; @ye2025; @nambu2025] are recent attempts, and [@ororbia2024] reviews the broader family of neuroscience-inspired ML approaches.

## 7c. Equilibrium propagation and contrastive Hebbian learning

Equilibrium propagation [@scellier2017] trains energy-based networks with two phases (free and clamped) whose difference yields a local gradient. [@ernoult2019] showed that for static-input RNNs, EqProp's updates exactly match BPTT gradients. [@millidge2022unify] unifies PC, EqProp, and contrastive Hebbian learning in an "infinitesimal inference" framework. [@hoier2023; @falk2023; @scellier2023] extend it to dyadic neurons, temporal contrastive learning, and analog implementations. EqProp's main remaining liability is **non-causality / two phases** — it requires a clamped phase reference.

# 8. Where are the genuine gaps?

Synthesizing the six clusters, several *underexplored mechanism combinations* stand out as candidate directions for a new learning rule. Each of these would be a fresh combination rather than a known one.

1. **Two-timescale STC + temporal credit assignment.** STC explicitly provides a mechanism to defer commitment until a modulatory event arrives; no existing rule (to my reading) uses a fast tag + slow consolidation two-timescale update *in service of* RNN temporal credit. e-prop and RFLO have continuous eligibility traces that decay passively; they do not have a thresholded "capture" step. Combining a fast eligibility tag with a slow consolidation signal driven by post-hoc prediction-error surprisal is not obviously reducible to a decay-only rule.

2. **Cell-type-specific feedback for RFLO/e-prop, but with learned (not hand-chosen) cell-type gains.** ModProp uses hand-constructed cell-type-specific multiplicative gains to reconstruct the long-range Jacobian. A plausible unknown: can the gains themselves be learned *online* by a local Hebbian rule, without any BPTT meta-loop?

3. **Temporal predictive-coding RNNs where the local prediction error drives eligibility-trace consolidation.** PC gives a locally-computable error term; RFLO gives a local eligibility trace; but combining them so that the PC error is the modulator that decides *whether* to consolidate an eligibility trace (analogous to STC's capture step) does not appear explicit in the literature.

4. **Dendritic + temporal.** Multi-compartment rules (Sacramento, Guerguiev, Payeur) mostly assume static inputs or a teaching phase. A rule that places the apical compartment's input as the *temporally delayed* prediction from a slow-timescale modulatory broadcaster could tie dendritic credit to an STC-style tag.

5. **Node perturbation with eligibility gating.** Node-perturbation's variance is its Achilles heel. Gating the accumulation of ξ_i·r_j by a locally-computed "this timestep was informative" signal (derived from a local error or surprise measure) could sharpen the gradient estimate without requiring a global BPTT-like backbone.

6. **Synthetic gradients meeting biology.** A synthetic-gradient module that is *itself* constrained to be a local three-factor Hebbian rule — so that both the online learner and its gradient estimator are biologically plausible — is absent from the literature.

7. **Causal-only EqProp.** EqProp's clamped phase is the least plausible ingredient. A variant where the "nudging" arrives as a temporally lagged modulatory broadcast (rather than a nominal clamped phase) — effectively rewriting EqProp into a one-phase three-factor rule — seems open.

Not all of these are surprising in the sense the mission requires; gaps #1, #3, #5, and #7 strike me as having the right flavor of *unexpected connection*. The Step 2 design should select one and justify its novelty against the most closely related work.

# 9. Key takeaways for Step 2

- The dominant functional form is a **three-factor rule** Δw_{ij} = η · (pre_j) · (post_i) · (modulator). Any new rule will almost certainly live in this family; novelty will come from the form of the modulator, the eligibility, or the consolidation schedule.
- **e-prop and RFLO are the strongest current baselines** for online local RNN learning on supervised tasks. ModProp is the strongest attempt to bridge the long-range temporal Jacobian gap — but its cell-type gains are hand-designed.
- **STC-style two-timescale consolidation is unusually underexplored as a credit-assignment mechanism** (as opposed to a continual-learning mechanism). This is the most promising gap identified here.
- **Vanilla RNN + short supervised task (copy / delayed match at sequence length 10–50)** is the right starting testbed; this matches the mission's Step 3 sanity experiment.
