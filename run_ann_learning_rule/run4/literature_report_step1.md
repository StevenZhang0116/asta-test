---
bibliography: references.bib
---

# Literature Report — Biologically Plausible Learning Rules for Recurrent Neural Networks

**Context.** This report is step 1 of an independent research project (run4) whose mission is to design a *new* biologically plausible learning algorithm for RNNs. The report is not an end product; it is scaffolding for (i) §3 Related Work in `research_state.md`, (ii) a novelty audit for candidate algorithms proposed in subsequent steps. Literature was pulled via Asta Paper Finder across six query clusters covering RTRL approximations, feedback-alignment variants, dendritic/compartmental credit assignment, predictive coding / equilibrium propagation, neuromodulation / three-factor rules, and burst-dependent plasticity. Queries produced ~820 unique candidate papers; ~60 canonical or frontier references are cited below.

BPTT is the implicit baseline throughout. The biological implausibilities we track — following `mission.md` — are: **(WT)** weight transport, **(LOC)** locality, **(CAUS)** causality (no future information), **(GLOB)** no unboundedly rich global error signal, **(MEM)** bounded memory (no full-trajectory storage), **(PHASE)** no separate frozen forward / backward phases, **(CONT)** continuous-time compatibility.

---

## 1. Methodological Clusters

### 1.1 RTRL and its approximations

**Mechanism.** Real-Time Recurrent Learning [@williams1989] maintains an online "influence Jacobian" $\partial h_t / \partial \theta$ and updates it in lockstep with the forward pass, yielding fully online, causal gradients without storing activation history. Its $O(N^4)$ cost per step (for $N$ hidden units) is the barrier, so the literature approximates this Jacobian with lower-rank, sparser, or randomized structures.

**Canonical/frontier citations.** UORO [@tallec2018] (rank-1 unbiased); KF-RTRL [@mujika2018] (Kronecker-factored); SnAp-k [@menick2020] (sparse k-step Jacobian); OSTL [@bohnstingl2020] (spatial/temporal gradient separation, BPTT-equivalent for shallow SNNs); Marschall et al. [@marschall2019] unified framework; sparse activity+parameter RTRL [@subramoney2023]; columnar-constructive RTRL [@javed2023]; online long-range dependencies via modular networks [@zucchet2023]; RTRL for RL [@muratore2020]. E-prop [@bellec2019a; @bellec2020] is historically classified as an RTRL approximation restricted to local eligibility traces (see §1.2). [@millidge2025] extends e-prop to deep recurrent stacks. [@liu2025] compares e-prop to BPTT on *neural-recording similarity* (not just task accuracy).

**BPTT implausibilities addressed.** CAUS, MEM, PHASE are addressed by construction (strictly forward, bounded per-step memory, no frozen backward pass).
**NOT addressed.** WT (most RTRL variants still require the same weight matrix used in the forward pass to appear in the Jacobian update — effectively the *same* matrix, but still non-local to the synapse). LOC is weakened: the Jacobian entries couple cross-synaptic information. GLOB: most variants still use a scalar or layer-wise error signal.

**Best known empirical performance & tasks.** Bohnstingl et al. [@bohnstingl2020] (OSTL) is gradient-equivalent to BPTT on shallow SNNs. Zucchet et al. [@zucchet2023] reports competitive Long-Range Arena results with only ~2× the memory of a forward pass. Javed et al. [@javed2023] show linear-in-parameter cost RTRL on RL benchmarks. Subramoney et al. [@subramoney2023] exploit activity+parameter sparsity to get tractable RTRL on SNNs.

**Frontier map.** The frontier has shifted from "can we approximate the RTRL Jacobian cheaply?" (2017–2020) to "can we exploit architectural priors — modularity, state-space structure, sparsity — to make RTRL fully usable on long-range tasks?" (2023–2026). The open problem is scaling RTRL-class algorithms to large, densely-recurrent networks *without* architectural modularity constraints, and reconciling RTRL variants with local (per-synapse) biology: every current RTRL approximation either concedes locality or truncates to an eligibility trace (thus collapsing into §1.2).

---

### 1.2 Eligibility-trace / three-factor local learning rules

**Mechanism.** A per-synapse eligibility trace $e_{ij}(t)$ integrates a local product of pre- and post-synaptic activity with a temporal filter. The weight update is $\Delta W_{ij} \propto M(t)\, e_{ij}(t)$, where $M(t)$ is a modulatory ("third factor") signal — a scalar, vector, or cell-type-specific modulator. The trace provides the *temporal* component of credit assignment; the modulator provides the *task-relevance* component.

**Canonical/frontier citations.** RFLO [@murray2019]; e-prop [@bellec2019a; @bellec2020]; ModProp [@liu2020; @liu2022]; generalised deep e-prop [@millidge2025]; Bellec et al. alternatives to BPTT [@bellec2019]; generalisation of bio-plausible credit rules [@liu2022a]; neural-similarity evaluation [@liu2025]; meta-learning plasticity rules [@shervanitabar2022]; EchoSpike online predictive plasticity [@graf2024]; Bio-Mamba integrating RTRL + STDP [@qin2024]; how initial connectivity shapes biologically plausible RNN learning [@liu2024]; brain-machine-interface-based distinction between rules [@portes2022]; comprehensive comparison [@lv2024].

**BPTT implausibilities addressed.** LOC, MEM (trace is bounded), PHASE, CAUS when modulator is instantaneous. Many variants also sidestep WT by using random feedback for $M(t)$.
**NOT addressed.** GLOB — the modulator is typically a globally broadcast vector carrying the gradient of the loss w.r.t. output. Also, e-prop-family traces *truncate* temporal contributions beyond a local window, so they pay a bias vs BPTT on tasks requiring long-range temporal credit. ModProp [@liu2022] explicitly targets this by having local diffusive neuromodulators propagate credit across arbitrary timespans, but at the cost of architectural assumptions about cell types and connection sparsity [@liu2020].

**Best known empirical performance & tasks.** e-prop matches BPTT within 10–20% on TIMIT speech and delayed-match tasks [@bellec2020]. ModProp closes the gap on longer-horizon RSNN tasks [@liu2022]. [@zucchet2023] surpasses e-prop on LRA via modularity, suggesting e-prop alone scales poorly to long-range. [@liu2025] finds that e-prop matches BPTT in *neural-similarity* metrics when task performance is matched.

**Frontier map.** The cluster has moved from "do three-factor traces work at all?" (2018–2020) to "how good can truncation-free, biologically grounded three-factor rules be?" (2022–2026). ModProp established that arbitrarily-long credit can flow through *diffusive, cell-type-specific* modulators. EchoSpike [@graf2024] and Bio-Mamba [@qin2024] suggest self-supervised and state-space-structured variants can close more of the gap. Open problems: (a) making the modulator *not* a globally broadcast error vector; (b) handling architectures without cell-type-specific priors; (c) maintaining performance as depth grows [@millidge2025].

---

### 1.3 Feedback Alignment family (in recurrent settings)

**Mechanism.** Replace the transposed forward weights $W^\top$ in the backward pass with a fixed random matrix $B$ (Feedback Alignment [@lillicrap2016]) or its direct variant (DFA — a random projection from the output error straight to each hidden layer). Forward weights align with the random backward weights during training, enabling useful (though noisier) gradients without weight transport. Weight Mirrors [@akrout2019] introduce explicit dynamics that make $B$ track $W$ using only local activity.

**Canonical/frontier citations.** Lillicrap et al. [@lillicrap2016]; Akrout et al. weight mirrors [@akrout2019]; Deep Feedback Control [@meulemans2021]; DFA extended to CNN/RNN (cited in cluster results); Product Feedback Alignment [@li2024] (2024, closely approximates BP in deep ConvNets while preserving locality); meta-learned random feedback [@shervanitabar2022]; sign-concordant microcircuit model [@yang2022]; direct modulation without backward pass PEPITA [@dellaferrera2022]; deep reservoir with DFA [@evanusa2020].

**BPTT implausibilities addressed.** WT (primary target); partially LOC (error propagation is layer-local rather than synapse-local). DFA additionally removes sequential layer-wise unrolling of the backward pass.
**NOT addressed.** GLOB (still needs the output error), CAUS and MEM for *recurrent* tasks (raw FA doesn't change that you still need BPTT-style unrolling — you just replace one matrix). In recurrent settings, FA must be combined with an RTRL-style or eligibility-trace substrate to yield an online rule.

**Best known empirical performance & tasks.** In feedforward settings FA matches BP on small vision tasks, lags on ImageNet. Weight Mirrors close much of the gap. In recurrent settings, FA plugs into DFA (direct projection to each step) or into RTRL-class online updates. PEPITA [@dellaferrera2022] achieves competitive performance *without any backward pass* by modulating a second forward pass with error — a notable "surprise"-worthy departure. [@li2024] PFA matches BP on deep ConvNets.

**Frontier map.** Classical FA/DFA is now well-understood: it works but scales poorly to deep / long-sequence problems unless paired with structure. Post-2022 work has converged on two routes to close the gap: (a) *learn* the feedback pathway (weight mirrors, meta-learning, PFA); (b) replace the backward pass entirely with a second forward pass (PEPITA) or with a control-based equilibrium (DFC). Recurrent-specific FA work is sparse — most recurrent extensions simply use DFA at each timestep plus an eligibility trace, i.e. they reduce to §1.2.

---

### 1.4 Dendritic / compartmental credit assignment

**Mechanism.** Pyramidal neurons with explicit apical (top-down) and basal (bottom-up) dendritic compartments compute *local dendritic prediction errors* as the mismatch between predicted top-down input (from lateral interneurons / feedback) and actual top-down feedback. Plasticity is gated by these local errors, eliminating the need for an explicit backward pass.

**Canonical/frontier citations.** Sacramento et al. dendritic microcircuits [@sacramento2018]; burst-dependent plasticity in hierarchical circuits [@payeur2020; @payeur2021]; BurstCCN single-phase cortico-cortical networks [@greedy2022]; hierarchical PC via dendritic error computation [@mikulasch2022]; efficient backprojections across cortical hierarchies [@max2022]; cortical error-neuron microcircuits [@max2025]; Deep Feedback Control [@meulemans2021]; two-compartment SNN online learning [@yin2024]; burst + dendritic target-based learning [@capone2022; @capone2022a]; BP through space, time, and the brain [@ellenberger2024]; breaking E/I balance to encode errors [@rossbroich2025].

**BPTT implausibilities addressed.** WT (no symmetric backward matrix), LOC (synapse-local dendritic errors), PHASE (single-phase rules — Sacramento and BurstCCN), MEM (partially).
**NOT addressed.** In pure feedforward form: CAUS and MEM for long temporal horizons; temporal credit assignment still needs an eligibility-trace substrate. GLOB is often addressed by construction since "errors" are local dendritic quantities rather than global.

**Best known empirical performance & tasks.** Sacramento et al. show single-phase MNIST-level learning. BurstCCN [@greedy2022] achieves competitive learning on MNIST/CIFAR-10 style tasks under single-phase continuous-time dynamics. [@max2025] demonstrates BP-like performance from error-neuron microcircuits on standard benchmarks. [@capone2022] shows hierarchical imitation learning via target-based bursts.

**Frontier map.** Dendritic models have matured from proofs-of-concept (Sacramento 2018) to concrete cortical circuit models that match BP within a small gap on vision tasks (2022–2025). The *recurrent / temporal* extension is the current frontier: burst-based target propagation [@capone2022], [@ellenberger2024] learning through space+time in dendritic models, and two-compartment SNNs for sequence modeling [@yin2024] are early-stage. Open problems: (a) scaling to long sequences without reverting to BPTT; (b) identifying the minimal compartment / interneuron requirements that are sufficient; (c) reconciling the different top-down signaling mechanisms proposed (bursts vs. apical prediction errors vs. E/I imbalance).

---

### 1.5 Predictive coding / equilibrium propagation / energy-based local learning

**Mechanism.** The network's dynamics are governed by an energy function. Inference runs the network to an equilibrium; learning is a local rule (often Hebbian or anti-Hebbian) that reduces the energy / prediction error at equilibrium. Equilibrium Propagation [@scellier2017] proves that weight updates computed at a pair of equilibria match BPTT gradients in the limit. Predictive Coding (PC) [@whittington2019; @millidge2020a] similarly approximates BP along arbitrary computation graphs by propagating prediction errors through a hierarchy.

**Canonical/frontier citations.** EqProp [@scellier2017]; updates match BPTT gradients [@ernoult2019]; holomorphic EqProp [@laborieux2022]; least-control principle [@meulemans2022]; PC approximates BP along arbitrary computation graphs [@millidge2020a]; relaxing PC constraints [@millidge2020]; temporal PC for prediction [@millidge2024]; sequential memory via tPC [@tang2023]; long-range temporal deps via tPC + RTRL [@potter2026]; dendritic-error hierarchical PC [@mikulasch2022]; sequence-learning EqProp (cited in cluster).

**BPTT implausibilities addressed.** WT, LOC, PHASE (EqProp uses two phases but each is local; recent variants e.g. holomorphic EqProp collapse to a single oscillation phase), MEM (once at equilibrium, no long history stored), CONT (natural fit). Temporal versions inherit challenges in CAUS.
**NOT addressed.** Convergence to equilibrium has high wall-clock cost; the two-phase structure of classical EqProp violates PHASE; temporal PC approximations often truncate temporal credit (mirroring e-prop limits).

**Best known empirical performance & tasks.** EqProp scales to CIFAR-10 ConvNets [@laborieux2022], with recent work pushing to deeper architectures. tPC achieves sequential memory on structured inputs [@tang2023]. [@potter2026] is the most recent attempt to combine tPC with approximate RTRL for long-range temporal tasks, closely matching BPTT on synthetic long-range benchmarks.

**Frontier map.** PC / EqProp has converged on being *algorithmically equivalent* to BP in the limit and *locally implementable* in principle [@salvatori2023]. The 2024–2026 frontier is (a) efficient temporal extensions (tPC + RTRL, @potter2026), (b) scaling EqProp beyond equilibrium — e.g. avoiding the full relaxation-to-equilibrium per step [@meulemans2022]. Open problem: bringing PC/EqProp to mainstream RNN tasks without reverting to BPTT-equivalent truncations.

---

### 1.6 Neuromodulation-heavy / RL-flavored three-factor rules

**Mechanism.** A scalar or low-dimensional neuromodulatory signal (dopamine-like RPE, vector-valued cell-type-specific modulators) gates plasticity. Learning is driven by *reward-modulated Hebbian/STDP* or related schemes. Under proper conditioning, gradient-descent-like directions emerge.

**Canonical/frontier citations.** Cell-type-specific modulatory signals [@liu2020]; ModProp [@liu2022]; e-prop's natural interpretation as three-factor [@bellec2019a]; meta-learned plasticity with random feedback [@shervanitabar2022]; comprehensive comparison [@lv2024]; PC with spiking neurons [@ndri2024]; khacef survey of spike-based local plasticity [@khacef2022].

**BPTT implausibilities addressed.** GLOB — this cluster explicitly targets the "no globally-broadcast error vector" critique by using low-dimensional, biologically-supported modulators. Also LOC and (for spike-based variants) CAUS.
**NOT addressed.** Raw reward-modulated STDP suffers from *variance*, not bias: it gets the expected update direction right but very noisily, and temporal credit assignment remains bottlenecked by the carrying capacity of the modulator. Scaling to deep networks is difficult without re-importing a high-dimensional error signal.

**Best known empirical performance & tasks.** ModProp [@liu2022] handles temporal credit across seconds on RSNN tasks with only cell-type-specific modulators — a notable demonstration that scalar modulators are *not* the only biologically plausible option if you accept diffusive cell-type-specific signals. [@portes2022] shows that at the BMI level of observation, reward-based rules are distinguishable from supervised ones by gradient bias.

**Frontier map.** The cluster's center of gravity has shifted from scalar-dopamine models (pre-2020) to *vector-valued, cell-type-specific* modulators (2020–2025), and further to *meta-learned* plasticity rules where evolution or gradient-based meta-learning discovers the rule [@shervanitabar2022]. Open problems: (a) identifying the minimum-dimensional modulatory signal sufficient for competitive RNN learning; (b) integrating modulatory signals with dendritic error signals; (c) continuous-time, event-driven implementations.

---

### 1.7 Additional clusters encountered

- **Burst-dependent plasticity as a credit-carrying channel.** Beyond dendrites, bursts are hypothesized to multiplex top-down vs. bottom-up information on the same axons ([@payeur2020; @greedy2022; @capone2022]). This straddles §1.4 and §1.6.
- **"No backward pass" (PEPITA-like).** Replacing the backward pass with a second forward pass in which the input is modulated by the output error [@dellaferrera2022]. This sidesteps WT, PHASE, and LOC simultaneously in feedforward settings, but has not been fully extended to recurrent temporal tasks.
- **Perturbation-of-E/I-balance as error signaling.** [@rossbroich2025] (2025) proposes that local deviations from E/I balance, produced by targeted feedback to inhibitory interneurons, encode neuron-specific error signals — a *novel* biological substrate for local error delivery distinct from both modulators and dendritic compartments.
- **Initial-connectivity conditioning for bio-plausible learning.** [@liu2024] shows that initial weight magnitude and spectral structure shape what bio-plausible rules *can* learn — an often-overlooked axis.

---

## 2. Recent (2023–2026) Work Callouts

Recent work is concentrated on three fronts:

1. **Closing the long-range gap for local rules.** [@zucchet2023] (online long-range deps via modularity), [@potter2026] (tPC + RTRL on long sequences), [@qin2024] (Bio-Mamba: RTRL+STDP in selective SSMs), [@javed2023] (columnar RTRL), [@subramoney2023] (activity-and-parameter-sparse RTRL). The consensus: *architectural priors* (modularity, state-space structure, sparsity) are what let local rules remain competitive on long-range tasks.

2. **Scaling dendritic / cortical-circuit models to meaningful tasks.** [@greedy2022] (BurstCCN, single-phase), [@capone2022] (burst+dendrite target-based), [@max2025] (BP-and-the-brain realized in cortical error-neuron microcircuits), [@ellenberger2024] (space-time-brain backpropagation), [@yin2024] (two-compartment SNN online learning). The trend is toward *tightly-specified cortical microcircuits* that match BP on vision benchmarks without weight transport or a frozen backward pass.

3. **Evaluation beyond task accuracy.** [@liu2025] compares rules on *neural recording similarity* (not just task loss), showing e-prop can match BPTT on this axis when matched for accuracy. [@portes2022] proposes brain-machine-interface tests to distinguish learning rules in vivo. [@lv2024] and [@ndri2024] provide comprehensive 2024 surveys with standardized bio-plausibility criteria.

4. **Novel biological substrates for error signals.** [@rossbroich2025] (E/I-imbalance as local error code); [@liu2024] (initial connectivity as a critical factor); [@shervanitabar2022] (meta-learned plasticity rules with random feedback).

---

## 3. Candidate Gaps — Directions for a Novel Rule

These are *under-explored* directions that could host a non-trivial new learning rule. Each is evaluated against the mission's "surprise" requirement: the direction should not be an obvious trivial extension.

### Gap 1 — Explicit two-timescale synaptic tagging + capture for RNN credit

- **Biological mechanism.** Frey & Morris's synaptic tagging-and-capture: a short-lived, silent "tag" is set at recently-active synapses and is later "captured" by a delayed, slower, more global signal that determines whether the tag is consolidated. The two timescales are separately observable in biology (minutes vs. hours).
- **BPTT implausibility addressed.** CAUS (tags are strictly causal), MEM (tag has a finite lifetime), and — uniquely — the "non-causality of BPTT" is re-read as *tag-consolidation* rather than *backward propagation*.
- **Why unused.** Current eligibility-trace rules typically use a *single-timescale* trace (e-prop, RFLO, ModProp). A small number of papers mention tagging but do not exploit the *capture* asymmetry: they treat the tag as just a decaying trace. The *selectivity* of capture — the capture signal only consolidates tags that are still above threshold — is what makes this different from a continuous three-factor rule. The surprise is that the capture signal can be very *sparse* (not always present) and yet the rule can outperform continuous three-factor rules on long-horizon tasks because sparse capture reduces variance.
- **Sketch.** Two eligibility traces per synapse: a fast tag $e^f_{ij}(t)$ (decay $\tau_f \sim 10$ steps) and a slow "capturable" tag $e^s_{ij}(t)$ that only updates when a discrete "capture event" $c(t) \in \{0,1\}$ fires. $\Delta W_{ij} \propto M(t)\cdot e^s_{ij}(t)$. The capture event is triggered by a global signal (surprise, uncertainty, reward boundary). A theoretical angle: show that sparse capture yields a *variance-reduced unbiased* estimator of the BPTT gradient on piecewise-stationary tasks.

### Gap 2 — E/I-balance perturbations as a vector-valued error code for recurrent networks

- **Biological mechanism.** [@rossbroich2025] proposed E/I-imbalance as a local error code for *feedforward* networks. No one has extended this to recurrent temporal credit assignment.
- **BPTT implausibility addressed.** WT, LOC, GLOB — crucially, the "error signal" is carried by physiologically measurable perturbations of excitation/inhibition balance rather than a globally broadcast scalar or vector.
- **Why unused.** [@rossbroich2025] is very recent and explicitly feedforward. Extending to RNN requires coupling the E/I-imbalance error signal with an eligibility-trace substrate that respects the *signed* nature of the balance perturbation.
- **Sketch.** Each recurrent unit has separate excitatory and inhibitory drive; the *signed deviation* from balance at that unit's soma modulates an eligibility trace of its incoming synapses. Recurrent temporal credit is propagated implicitly via the recurrent dynamics of E/I populations. Surprise: error signals arise as a *natural byproduct of cortical dynamics* rather than an additional broadcast channel.

### Gap 3 — Meta-plasticity-gated eligibility: the synapse learns *when* to be plastic

- **Biological mechanism.** Metaplasticity — plasticity of plasticity — where each synapse's learning rate is itself dynamic, set by recent history of pre/post activity and global modulatory state. Experimentally well-supported, but rarely used as the *mechanism* for solving temporal credit assignment.
- **BPTT implausibility addressed.** MEM, CAUS. The key move: the synapse's local meta-variable acts as a *proxy* for the sensitivity $\partial h_t / \partial W_{ij}$ that RTRL computes explicitly but that e-prop truncates.
- **Why unused.** Meta-plasticity is usually treated as a *regularizer* (continual learning, consolidation) rather than as a mechanism for carrying gradient information across time. Treating meta-plasticity as a *credit-carrying variable* rather than a regularizer is the conceptual twist.
- **Sketch.** Per synapse, maintain $\eta_{ij}(t)$ (the effective local learning rate) which itself evolves by a local rule driven by a combination of pre/post activity, post-synaptic voltage variance, and a scalar modulator. The eligibility trace is gated by $\eta_{ij}(t)$. The theoretical claim: for certain RNN architectures, $\eta_{ij}$ can be chosen to absorb the non-local portion of the BPTT Jacobian into a locally-tractable variable.

### Gap 4 — Predict-the-future-state as the sole error signal, with no explicit task loss

- **Biological mechanism.** Cortex is widely thought to be a prediction machine [@whittington2019; @mikulasch2022]. Most predictive-coding RNN work uses PC as a *computational substrate* for approximating BP on a supervised loss. An under-explored inversion: use *only* self-supervised next-state prediction as the learning signal, and let task performance emerge via a shallow readout trained with BPTT-free local rules on top of a *purely unsupervised* PC-trained recurrent core.
- **BPTT implausibility addressed.** GLOB (no global task-error signal is needed for the recurrent layers), PHASE, MEM.
- **Why unused.** Most PC-in-RNN work uses task supervision [@tang2023; @potter2026]. The idea of using *only* predictive-coding-driven self-supervision for the recurrent core is hinted at in recent Bio-Mamba [@qin2024] and EchoSpike [@graf2024] but not fully exploited. Surprise: if the self-supervised representations are sufficient, the "credit assignment problem" for the recurrent core vanishes — it's not assigning credit for a task, it's assigning credit for a *local* prediction error.
- **Sketch.** RNN is trained by temporal predictive coding with local errors only [@tang2023]. Readout is a linear + softmax layer trained by a local three-factor rule on task error. The recurrent weights never see the task loss. Compare this against BPTT + full end-to-end training. This is a kind of "credit assignment decoupling" that is rarely benchmarked cleanly against BPTT.

### Gap 5 — Dendritic segregation of *temporal* vs. *spatial* error signals

- **Biological mechanism.** Apical vs. basal dendrites naturally segregate top-down vs. bottom-up input. Current dendritic models [@sacramento2018; @greedy2022] use this segregation for spatial credit assignment (across layers). An under-explored move: use a *third* compartment (e.g., oblique dendrites or a voltage-compartment distinction) to segregate *temporal* credit signals from *spatial* ones.
- **BPTT implausibility addressed.** LOC, MEM, CAUS. Separating temporal and spatial error computation into distinct compartments means the eligibility trace lives in a *dedicated* compartment rather than sharing the same state as the spatial error.
- **Why unused.** Compartmental models have been framed in terms of spatial hierarchy (apical = top-down, basal = bottom-up). Treating dendritic geometry as a substrate for segregating *temporal dimensions* of credit is conceptually distinct and has been gestured at in reviews [@greedy2022; @capone2022] but not realized as a concrete learning rule.
- **Sketch.** Three-compartment neuron: basal = forward input, apical = top-down error, oblique = temporal context (integrator with a long time constant). Plasticity on forward weights is gated by basal × apical (spatial error); plasticity on recurrent weights is gated by basal × oblique (temporal error). Separately-gated STP and burst dynamics would close the loop.

### Gap 6 — Explicit *astrocytic* slow-modulation as a credit-carrying substrate

- **Biological mechanism.** Astrocytes operate on seconds-to-minutes timescales, integrate activity over spatially-structured domains, and release gliotransmitters that gate synaptic plasticity. They are increasingly implicated in learning. No existing bio-plausible RNN learning rule uses astrocytic-like slow modulation as a *credit-carrying variable*.
- **BPTT implausibility addressed.** MEM, GLOB — the astrocyte's slow integration gives a natural *free* long time constant, and its spatially-structured release field gives a natural *low-dimensional-but-not-scalar* modulator.
- **Why unused.** This is at the boundary of computational neuroscience and ML. No RNN learning rule I found in the survey exploits astrocyte-like slow modulation as a computational substrate — it is always discussed as biological context.
- **Sketch.** A slow, spatially-structured "glial state" $g(t)$ is driven by local neural activity averages and releases a region-specific modulator that gates eligibility-trace updates on a long timescale. The learning rule combines fast per-synapse traces with slow glial modulation. Surprise: the glial compartment gives *free* temporal credit carriage across timescales that trace decay cannot easily span.

### Summary of gaps

| Gap | Addresses | Surprise / Non-triviality |
|---|---|---|
| 1. Two-timescale tagging + sparse capture | CAUS, MEM | Sparse discrete capture vs. continuous three-factor |
| 2. E/I-balance perturbation errors (recurrent) | WT, LOC, GLOB | Error as emergent dynamic variable rather than broadcast signal |
| 3. Metaplasticity as credit carrier | MEM, CAUS | Meta-plasticity as algorithmic device, not regularizer |
| 4. Self-supervised PC core + local readout | GLOB, PHASE, MEM | Decoupling: recurrent core never sees task loss |
| 5. Dendritic segregation of temporal vs. spatial | LOC, MEM, CAUS | Third compartment for time |
| 6. Astrocytic slow-modulation credit channel | MEM, GLOB | Glia as compute, not context |

---

## 4. Implications for §3 Related Work and §5 Candidate Design

- §3 Related Work should be organized around the six clusters above (plus §1.7 miscellany), with canonical citations per cluster.
- §5 candidate-design work should pick one or two of the six gaps above as the core novelty axis. Gaps 1, 2, 3 are most likely to yield a *single-sentence novelty claim* that is both distinct from the current frontier and testable on short supervised RNN tasks. Gaps 4, 5, 6 are more ambitious.
- Critical caveat: the novelty audit at Step 2 must explicitly check each candidate design against the 2023–2026 references, not just the pre-2022 anchors. Several of the frontier references ([@potter2026], [@liu2025], [@rossbroich2025], [@max2025], [@graf2024], [@qin2024], [@millidge2025]) were published very recently and cover the natural first directions a designer might pursue.

---

## 5. Open Questions Surfaced by the Survey

1. *Is temporal credit assignment the hard part, or is spatial credit in deep hierarchies?* The dendritic/compartmental cluster has made strong progress on spatial; e-prop and RTRL-variants on temporal; no single rule yet does both as well as BPTT.
2. *How important is globally-broadcast vs. locally-delivered error?* [@rossbroich2025] and [@liu2022] suggest local delivery is feasible; but few works isolate the contribution of this axis.
3. *Which initial-connectivity regimes are most favorable to bio-plausible rules?* [@liu2024] identifies this as an independent axis that few prior rules control for.
4. *What is the right evaluation axis beyond task accuracy?* [@liu2025] and [@portes2022] argue for neural-similarity / BMI-discriminability metrics. The mission should consider adopting at least one non-accuracy metric at eval time.
