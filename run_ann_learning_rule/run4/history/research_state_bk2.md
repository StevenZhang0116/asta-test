# Research State — Biologically Plausible RNN Learning Rule (run4, independent)

*Independent replicate — do not reference sibling `runN/` directories.*

---

## 1. Research Question & Scope

**Primary question.** What new learning rule can we design for recurrent neural networks that addresses known biological implausibilities of BPTT — weight transport (WT), non-locality (LOC), non-causality (CAUS), need for global error signals (GLOB), memory cost (MEM), separate forward/backward phases (PHASE), continuous-time incompatibility (CONT) — while remaining competitive on ML tasks, and what biological principles should it be grounded in?

**Scope.**
- Architectures: vanilla RNN first, then GRU/LSTM.
- Tasks: short supervised temporal tasks (T ~10–50) first, then longer; later extend to RL / unsupervised.
- Deliverable: a *new* algorithm — must pass a novelty + "surprise" bar.

---

## 2. Operational Definitions

- **Biologically plausible (working):** (a) no weight transport, (b) locality (each synaptic update uses only pre-synaptic activity, post-synaptic activity, and scalar/low-dimensional modulatory signals at the synapse), (c) online or near-online operation (bounded memory, no full-trajectory storage), (d) no separate backward pass requiring the whole-sequence frozen activity.
- **Competitive on ML tasks (working):** within ~1.5× the error of BPTT on matched architectures for short-sequence supervised tasks (copy, delayed-match, adding-task, permuted-MNIST subset).
- **Novelty:** not previously described in the literature, re-verified at every material design change.
- **Surprise:** not a trivial combination of known ingredients.

---

## 3. Related Work

Populated in Step 1 via a structured literature survey (see `literature_report_step1.md` and `references.bib`, 59 entries, 820 unique candidates screened). The literature organizes into six methodological clusters plus additional miscellany:

### 3.1 RTRL approximations
Maintain an online Jacobian of hidden state w.r.t. parameters; approximate its $O(N^4)$ cost with low-rank or sparse structure. Canonical: RTRL itself [@williams1989]; UORO [@tallec2018]; KF-RTRL [@mujika2018]; SnAp-k [@menick2020]; OSTL [@bohnstingl2020]; Marschall et al. unified framework [@marschall2019]. Recent frontier: columnar-constructive RTRL [@javed2023], sparsity-based RTRL [@subramoney2023], modular online learning of long-range deps [@zucchet2023]. Addresses CAUS/MEM/PHASE; struggles with WT/LOC.

### 3.2 Eligibility-trace / three-factor rules
Per-synapse trace $e_{ij}(t)$ + modulator $M(t)$; update $\Delta W_{ij} \propto M(t)\, e_{ij}(t)$. Canonical: RFLO [@murray2019]; e-prop [@bellec2019a; @bellec2020]; ModProp [@liu2020; @liu2022]. Recent frontier: deep e-prop [@millidge2025]; evaluation against BPTT on neural similarity [@liu2025]; meta-learned plasticity rules [@shervanitabar2022]; EchoSpike online self-supervised [@graf2024]; Bio-Mamba (RTRL+STDP in SSMs) [@qin2024]; initial-connectivity effects [@liu2024]; distinguishing rules via BMI [@portes2022]. Addresses LOC/MEM/PHASE and often CAUS; struggles with GLOB and long-range temporal credit.

### 3.3 Feedback alignment family (recurrent extensions)
Random or learned backward-path matrices. Canonical: FA [@lillicrap2016]; weight mirrors [@akrout2019]. Recent: Product Feedback Alignment [@li2024]; Deep Feedback Control [@meulemans2021]; sign-concordant microcircuits [@yang2022]; PEPITA (no backward pass) [@dellaferrera2022]; DFA extended to CNN/RNN [@evanusa2020]. Primarily addresses WT; in recurrent settings must combine with RTRL or eligibility traces to yield an online rule.

### 3.4 Dendritic / compartmental credit assignment
Multi-compartment neurons compute local dendritic prediction errors. Canonical: Sacramento 2018 [@sacramento2018]; Payeur burst-dependent [@payeur2020; @payeur2021]. Recent: BurstCCN single-phase [@greedy2022]; burst+dendrite target-based [@capone2022]; cortical error-neuron microcircuits [@max2025]; BP through space+time+brain [@ellenberger2024]; two-compartment SNN online [@yin2024]; hierarchical PC via dendritic error [@mikulasch2022]; efficient backprojections [@max2022]; breaking E/I balance to encode errors [@rossbroich2025]. Addresses WT/LOC/PHASE; temporal extension is an open frontier.

### 3.5 Predictive coding / equilibrium propagation
Local error signals at equilibrium; weight updates reduce energy. Canonical: EqProp [@scellier2017]; EqProp = BPTT gradients [@ernoult2019]; PC approximates BP [@millidge2020a; @whittington2019]. Recent: holomorphic EqProp [@laborieux2022]; least-control principle [@meulemans2022]; temporal PC [@millidge2024; @tang2023]; tPC + RTRL for long-range [@potter2026]; brain-inspired computational intelligence via PC [@salvatori2023]; relaxing PC constraints [@millidge2020]. Addresses WT/LOC/PHASE/CONT; full long-range temporal credit remains open.

### 3.6 Neuromodulation-heavy / three-factor (RL flavor)
Scalar or vector-valued modulators gate plasticity. Canonical e-prop lineage [@bellec2019a]; cell-type-specific modulators [@liu2020; @liu2022]; generalization studies [@liu2022a]; surveys: [@lv2024], [@khacef2022], [@ndri2024]. Primarily addresses GLOB (and LOC/CAUS in spike-based variants); struggles with variance and depth.

### 3.7 Additional axes
- Burst-dependent plasticity as a multiplexed top-down/bottom-up channel [@payeur2020; @greedy2022; @capone2022].
- "No backward pass" approaches (PEPITA) [@dellaferrera2022].
- E/I-balance perturbations as a neuron-specific error code [@rossbroich2025] (2025 — feedforward only so far).
- Initial connectivity as a conditioning axis [@liu2024].

### Frontier summary (2023–2026)
Three concurrent frontiers are active:
1. Closing the long-range-dependency gap for local rules through architectural priors (modularity, state-space structure, sparsity).
2. Scaling dendritic / cortical-circuit models to meaningful tasks with tightly-specified microcircuits.
3. Evaluation beyond task accuracy (neural similarity, BMI-discriminability, initial-connectivity robustness).

---

## 4. Hypotheses

Initial priors (from Step 0) and their status after Step 1's literature scan:

- **H1 (eligibility-trace primacy).** A local eligibility trace + low-rank/random projection of error can approximate BPTT gradients on short tasks. *Prior confidence: 60%. Post-Step-1: 65%* — confirmed as the dominant mature paradigm (e-prop, RFLO, ModProp), but with clear evidence that pure eligibility traces truncate long-range credit unless paired with cell-type or modular priors.
- **H2 (non-random feedback beats random).** Structured, non-symmetric feedback outperforms random FA on sequential tasks. *Prior: 55%. Post-Step-1: 60%* — weight mirrors [@akrout2019], PFA [@li2024], and learned-feedback DFC [@meulemans2021] confirm this in feedforward settings; recurrent-specific evidence is thinner.
- **H3 (modulatory bottleneck is sufficient).** Scalar or low-dim (≤10) neuromodulator is sufficient for short supervised tasks. *Prior: 40%. Post-Step-1: 35%* — pure scalar-dopamine rules remain variance-limited; vector-valued *cell-type-specific* modulators [@liu2020; @liu2022] are more promising but raise the dimensionality of the "modulator".
- **H4 (temporal decomposition).** Decomposing credit into causal-forward + anticipatory-predictive signals can replace BPTT's backward pass. *Prior: 35%. Post-Step-1: 45%* — temporal PC [@potter2026; @tang2023] and the least-control principle [@meulemans2022] partially realize this; a cleaner, more surprise-worthy version has not been done.

### New hypotheses introduced by Step 1 (drawn from the candidate-gap analysis in `literature_report_step1.md`)

- **H5 (sparse-capture two-timescale tagging).** A two-timescale synaptic tagging mechanism — a fast decaying tag + a slow capturable tag that only updates on *discrete capture events* — can provide a variance-reduced unbiased estimator of BPTT gradients on short supervised tasks. *Prior confidence: 35%.* Addresses CAUS, MEM, partially GLOB. **Surprise angle:** sparse discrete capture instead of continuous three-factor modulation.
- **H6 (E/I-balance recurrent error signaling).** Local deviations from E/I balance, driven by feedback to inhibitory interneurons, can serve as a *recurrent* error code — extending [@rossbroich2025] from feedforward to RNN. *Prior: 30%.* Addresses WT, LOC, GLOB. **Surprise:** error signals as emergent dynamic variables rather than broadcast.
- **H7 (metaplasticity as a credit-carrying variable).** Per-synapse meta-variables ("plasticity of plasticity") can absorb the non-local portion of the BPTT Jacobian that e-prop truncates, when meta-plasticity is treated as a *computational* rather than a regularizing mechanism. *Prior: 30%.* Addresses MEM, CAUS.
- **H8 (self-supervised PC core + local readout).** A recurrent core trained *purely* by temporal predictive-coding self-supervision, with a shallow readout trained by a local three-factor rule on task error, can approach BPTT performance on short supervised tasks. *Prior: 30%.* Addresses GLOB, PHASE, MEM. **Surprise:** the recurrent core never sees task loss.
- **H9 (dendritic segregation of temporal vs. spatial errors).** A three-compartment neuron (basal / apical / oblique) that segregates *temporal* credit signals into a dedicated compartment, distinct from spatial error, can outperform two-compartment models on sequential tasks. *Prior: 25%.* Addresses LOC, MEM, CAUS. **Surprise:** compartmental geometry as a substrate for *time*, not just hierarchy.
- **H10 (astrocytic slow-modulation credit channel).** A slow, spatially-structured glial state carrying a low-dim modulator integrated over seconds can provide a free long-time-constant credit-carrying variable. *Prior: 20%.* Addresses MEM, GLOB. **Surprise:** glia as compute, not context.

---

## 5. Experimental Designs

*To be filled as experiments are defined.* Standard harness anticipated:

- Tasks: copy task (T=10–30), delayed XOR, adding problem, sequential-MNIST subset.
- Baselines: BPTT, truncated BPTT, RFLO, e-prop, FA.
- Metrics: final loss/accuracy, wall-clock per epoch, memory footprint, gradient-alignment cosine with true BPTT gradient. Consider adding a neural-similarity-style metric [@liu2025] or init-connectivity-robustness axis [@liu2024] later.
- Ablations: remove modulatory signal, remove eligibility trace, freeze feedback weights vs. learn them.

---

## 6. Results Summary

### Step 1 — Literature survey
- 6 parallel Asta Paper Finder queries across 6 methodological clusters → 820 unique candidates.
- Produced `literature_report_step1.md` (31 KB, 57 citations, all resolved) and `references.bib` (59 entries).
- Organized the field into 6 primary clusters + 4 additional axes (§3).
- Identified 6 *candidate novelty gaps* (H5–H10 above) explicitly evaluated against the mission's "surprise" bar.
- Recent (2023–2026) work is well-covered — the novelty audit at Step 2 will have coverage of the current frontier, not just pre-2022 anchors.

---

## 7. Open Questions & Confusions

1. *Is temporal credit assignment the hard part, or is spatial credit in deep hierarchies?* Dendritic/compartmental cluster has made strong progress on spatial; e-prop/RTRL on temporal; no single rule yet does both as well as BPTT.
2. *How important is globally-broadcast vs. locally-delivered error?* [@rossbroich2025] and [@liu2022] suggest local delivery is feasible; but few works isolate this axis.
3. *Which initial-connectivity regimes are most favorable to bio-plausible rules?* [@liu2024] identifies this as an independent axis; consider controlling for it in Step-3+ experiments.
4. *What is the right evaluation axis beyond task accuracy?* [@liu2025; @portes2022] argue for neural-similarity / BMI-discriminability metrics.
5. *Which of H5–H10 is the most promising?* Gap-1 (H5), Gap-2 (H6), and Gap-3 (H7) look most like they could yield a single-sentence novelty claim both distinct from the current frontier and testable on short RNN tasks. Gaps 4/5/6 are more ambitious and should be held in reserve.

---

## Next Tasks (priority order)

1. **[NEXT]** Choose *one* of H5–H10 as the primary candidate novelty axis. Work up a concrete mathematical formulation (equations + pseudocode) of the proposed learning rule, including eligibility-trace / modulator / update equations. Record design decisions and trade-offs.
2. Run a novelty audit against the full frontier (the 59 entries in `references.bib` plus a targeted secondary search): formal search queries, top hits considered per candidate component, explicit novel/near-match/overlap verdict, recorded under a "Novelty audit" subsection.
3. Implement a minimal experimental harness (vanilla RNN + copy task, PyTorch, `panda` conda env, GPU if available per mission) and run baselines (BPTT, truncated BPTT, RFLO, e-prop) before testing the candidate rule.
4. Iterate.

---

## Appendix — Glossary of BPTT implausibilities (used throughout)

| Code | Implausibility |
|---|---|
| WT | Weight transport — backward path = transpose of forward |
| LOC | Non-locality — update needs non-local information |
| CAUS | Non-causality — future info needed to compute present update |
| GLOB | Global error signal required for all neurons |
| MEM | Unbounded memory — requires full-trajectory storage |
| PHASE | Separate forward and backward phases |
| CONT | Continuous-time incompatibility |
