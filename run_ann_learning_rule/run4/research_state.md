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

- **H5 (sparse-capture two-timescale tagging).** A two-timescale synaptic tagging mechanism — a fast decaying tag + a slow capturable tag that only updates on *discrete capture events* — can provide a variance-reduced unbiased estimator of BPTT gradients on short supervised tasks. *Prior confidence: 35%. Post-Step-3: 20%* — first instantiation (STC-Prop with local-activation-surprise trigger) failed on the copy task; the no-gate ablation matched RFLO, isolating the capture trigger as the problem. Class of rules not yet falsified; specific trigger choice is. Addresses CAUS, MEM, partially GLOB. **Surprise angle:** sparse discrete capture instead of continuous three-factor modulation.
- **H6 (E/I-balance recurrent error signaling).** Local deviations from E/I balance, driven by feedback to inhibitory interneurons, can serve as a *recurrent* error code — extending [@rossbroich2025] from feedforward to RNN. *Prior: 30%.* Addresses WT, LOC, GLOB. **Surprise:** error signals as emergent dynamic variables rather than broadcast.
- **H7 (metaplasticity as a credit-carrying variable).** Per-synapse meta-variables ("plasticity of plasticity") can absorb the non-local portion of the BPTT Jacobian that e-prop truncates, when meta-plasticity is treated as a *computational* rather than a regularizing mechanism. *Prior: 30%.* Addresses MEM, CAUS.
- **H8 (self-supervised PC core + local readout).** A recurrent core trained *purely* by temporal predictive-coding self-supervision, with a shallow readout trained by a local three-factor rule on task error, can approach BPTT performance on short supervised tasks. *Prior: 30%.* Addresses GLOB, PHASE, MEM. **Surprise:** the recurrent core never sees task loss.
- **H9 (dendritic segregation of temporal vs. spatial errors).** A three-compartment neuron (basal / apical / oblique) that segregates *temporal* credit signals into a dedicated compartment, distinct from spatial error, can outperform two-compartment models on sequential tasks. *Prior: 25%.* Addresses LOC, MEM, CAUS. **Surprise:** compartmental geometry as a substrate for *time*, not just hierarchy.
- **H10 (astrocytic slow-modulation credit channel).** A slow, spatially-structured glial state carrying a low-dim modulator integrated over seconds can provide a free long-time-constant credit-carrying variable. *Prior: 20%.* Addresses MEM, GLOB. **Surprise:** glia as compute, not context.

---

## 4a. Candidate Design — STC-Prop (selected at Step 2)

**Chosen hypothesis:** H5 — two-timescale synaptic tagging with *discrete, local, surprise-triggered capture events* as a variance-reduced credit-assignment mechanism for RNNs. See `design_step2.md` for full equations, pseudocode, biological mapping, and novelty audit.

**One-line summary.** Per synapse, maintain a fast RFLO-style tag $e_{ij}(t)$ and a slow commit trace $s_{ij}(t)$; per neuron, compute a local surprise $\sigma_i(t)$ and a homeostatic threshold $\theta_i(t)$ that produce a *discrete* capture indicator $c_i(t)\in\{0,1\}$; only when $c_i(t)=1$ does the slow trace integrate the fast tag and a weight update $\Delta W_{ij} \propto c_i(t)\, M_i(t)\, s_{ij}(t)$ fire. $M_i(t)$ is a random-feedback projection of output error (FA-style, no weight transport).

**BPTT implausibilities addressed.** WT ✅, LOC ✅, CAUS ✅, GLOB ⚠️ (partial — gated & low-rank), MEM ✅, PHASE ✅, CONT ⚠️ (discrete capture).

**Novelty verdict (recorded Step 2):** NOVEL with near-matches to document. Closest near-matches are AGMP 2026 (astrocyte-gated multi-timescale plasticity for continual learning — different substrate and gate polarity), meta-learned three-factor rules with sparse feedback 2025 (different sense of "sparse"), and Luboeinski 2020 / Györgyi 2022 (STC used for memory consolidation, not as a learning rule). The specific combination (per-neuron discrete capture gate driven by local surprise, two-timescale trace, used as a credit-assignment mechanism for supervised RNN training) does not appear in the retrieved literature. Re-audit required on any material design change.

## 5. Experimental Designs

*Updated at Step 2 to reflect the chosen candidate.*

### Step 3 plan — first comparison

- **Task:** copy task (T=10–30), delayed-XOR, adding problem on vanilla RNN.
- **Baselines:** BPTT, truncated BPTT, RFLO [@murray2019], e-prop [@bellec2019a; @bellec2020], FA-only (no temporal credit).
- **Candidate:** STC-Prop per `design_step2.md`.
- **Metrics:**
  - Final loss / accuracy
  - Wall-clock per epoch
  - Memory footprint
  - Gradient-alignment cosine with true BPTT gradient
  - **Variance of weight updates** (central to the rule's claim)
  - Fraction of timesteps with $c_i(t)=1$ per layer
- **Ablations:**
  - $c_i(t) \equiv 1$ (remove capture) — should collapse to an RFLO-like rule
  - $c_i(t) \sim \mathrm{Bernoulli}(\rho)$ i.i.d. (random capture) — isolates the surprise signal's contribution
  - $\beta = 0$ (remove slow commit trace) — tests whether the two-timescale structure matters
  - Structured feedback $B$ (sign-concordant or weight-mirror) vs. random
- **Hyperparameters to sweep:**
  - $\rho \in \{0.05, 0.1, 0.2\}$ (capture rate)
  - $\alpha \in \{0.5, 0.8, 0.9\}$ (fast tag decay)
  - $\beta \in \{0.9, 0.95, 0.99\}$ (slow commit decay)
  - $k$ in $B\in\mathbb{R}^{N\times k}$ for random feedback

### Later steps (Step 4+)

- Scale to sequential-MNIST subset; investigate long-range tasks via modular RNN blocks [@zucchet2023]-style architecture.
- Consider a neural-similarity metric [@liu2025] or initial-connectivity-robustness axis [@liu2024].
- GRU/LSTM extension.

---

## 6. Results Summary

### Step 1 — Literature survey
- 6 parallel Asta Paper Finder queries across 6 methodological clusters → 820 unique candidates.
- Produced `literature_report_step1.md` (31 KB, 57 citations, all resolved) and `references.bib` (59 entries).
- Organized the field into 6 primary clusters + 4 additional axes (§3).
- Identified 6 *candidate novelty gaps* (H5–H10 above) explicitly evaluated against the mission's "surprise" bar.
- Recent (2023–2026) work is well-covered — the novelty audit at Step 2 will have coverage of the current frontier, not just pre-2022 anchors.

### Step 2 — Candidate rule design + novelty audit
- Chose H5 (STC-Prop) over H6–H10; rationale recorded in `design_step2.md` §1.2.
- Wrote full mathematical formulation (equations, pseudocode, biological mapping) in `design_step2.md`.
- Ran a 12-query Semantic Scholar novelty audit (queries + top hits listed in `design_step2.md` §4).
- Identified three near-matches — AGMP 2026, meta-learned sparse-feedback three-factor 2025, and STC-for-consolidation papers Luboeinski 2020 & Györgyi 2022 — and differentiated STC-Prop from each.
- Verdict: NOVEL with near-matches to cite. `references.bib` now at 61 entries (added `luboeinski2020`, `gyorgyi2022`).
- Step 3 experiment plan recorded in §5 above.

### Step 3 — First experimental comparison on the copy task

**Harness.** `step3_experiment.py` — self-contained PyTorch script. Copy task with K=8, S=3, L=5 (T=12); vanilla RNN N=64; batch 64; 1000 iterations; GPU-accelerated (CUDA detected and used). Implements BPTT (autograd+Adam), RFLO (Murray 2019), STC-Prop (per `design_step2.md`), and STC-Prop-no-gate ablation (sets $c_i(t)\equiv 1$). Results in `step3_results.json`, learning curves in `step3_learning_curves.png`.

**Final results (1000 iterations, one seed):**

| Rule | Final loss | Final copy-window accuracy | Wall-clock (s) |
|---|---|---|---|
| BPTT | 0.0041 | **1.000** | 6.3 |
| RFLO | 0.358 | 0.380 | 33.3 |
| STC-Prop (gated) | 0.845 | **0.000** | 58.1 |
| STC-Prop-no-gate (ablation) | 0.331 | 0.359 | 57.9 |

**Key findings:**

1. **BPTT solves the task within 1000 iterations; RFLO learns partially (~38% copy accuracy).** This establishes the baselines work correctly and the harness is sound.
2. **STC-Prop as specified in `design_step2.md` *fails to learn* — it does worse than the naive no-gate ablation.** Loss is *higher* than RFLO, not lower. This is a clear negative result for the rule as specified.
3. **The no-gate ablation recovers RFLO-like performance.** That is, *removing* the capture gate $c_i(t)$ makes the rule learn again. So the culprit is the capture mechanism, not the slow commit trace or the feedback path.
4. **Homeostasis works as designed.** Capture rate converges to the target $\rho=0.1$ within ~100 iterations and stays there. So the failure is *not* that captures never fire; it is that when they *do* fire, the updates aren't useful.
5. **Plausible cause (hypothesis for Step 4).** The capture signal is currently pegged to *local activation surprise* $|h_i - \hat h_i|$, which is driven by input novelty — not by *task-relevant credit*. In the copy task, surprise is high at the "go" token and the original input tokens, not at the moments where the RNN *needs* to receive a plasticity signal (which would be wherever the error gradient is large). A capture signal anti-correlated with plasticity need is worse than no gate.

**Implication for the design.** The central idea — *sparse, discrete, locally-triggered captures* — is not falsified by this experiment; what *is* falsified is the specific choice of trigger (local activation surprise). Two reasonable pivots:
- **Pivot A:** Trigger capture on *feedback-signal magnitude* $|M_i(t)|$ instead of activation surprise, so captures coincide with moments when credit is actually informative. This is a small but meaningful change.
- **Pivot B:** Trigger capture on *output-error magnitude* (a global scalar) rather than per-neuron surprise. This costs some locality but keeps the sparse-discrete-capture claim intact.
- Either pivot requires a re-run of the novelty audit (per the mission's re-verification clause) since the capture trigger is a material part of the design.

**Caveats.** Single seed; no hyperparameter sweep; only one task. These results establish that the rule *as specified in `design_step2.md` §2.2 with surprise option (A) "local prediction error"* is not learning on this task, not that the *class* of two-timescale-tag-with-capture rules is unviable.

---

## 7. Open Questions & Confusions

1. *Is temporal credit assignment the hard part, or is spatial credit in deep hierarchies?* Dendritic/compartmental cluster has made strong progress on spatial; e-prop/RTRL on temporal; no single rule yet does both as well as BPTT.
2. *How important is globally-broadcast vs. locally-delivered error?* [@rossbroich2025] and [@liu2022] suggest local delivery is feasible; but few works isolate this axis.
3. *Which initial-connectivity regimes are most favorable to bio-plausible rules?* [@liu2024] identifies this as an independent axis; consider controlling for it in Step-3+ experiments.
4. *What is the right evaluation axis beyond task accuracy?* [@liu2025; @portes2022] argue for neural-similarity / BMI-discriminability metrics.
5. *Which of H5–H10 is the most promising?* Gap-1 (H5), Gap-2 (H6), and Gap-3 (H7) look most like they could yield a single-sentence novelty claim both distinct from the current frontier and testable on short RNN tasks. Gaps 4/5/6 are more ambitious and should be held in reserve.

---

## Next Tasks (priority order)

1. ~~Choose one of H5–H10 and write a concrete formulation + novelty audit.~~ **Done at Step 2.** → `design_step2.md`.
2. ~~Implement minimal experimental harness and run first comparison.~~ **Done at Step 3.** → `step3_experiment.py`, `step3_results.json`, `step3_learning_curves.png`. STC-Prop as specified **fails** on the copy task; no-gate ablation matches RFLO, indicating the capture-trigger choice (local activation surprise) is the problem.
3. **[NEXT — Step 4]** Pivot the capture-trigger:
   a. Pivot A: trigger capture on feedback-signal magnitude $|M_i(t)|$.
   b. Pivot B: trigger capture on output-error magnitude (global scalar).
   Re-run the 3-rule comparison under each pivot. Re-verify novelty before committing to either pivot.
4. Add the missing baselines from the original plan: e-prop, truncated BPTT, FA-only. Record gradient-alignment-cosine vs. BPTT.
5. Multi-seed runs and a small hyperparameter sweep ($\rho, \alpha, \beta$) on whichever pivot survives Step 4.
6. If none of the pivots succeed: the evidence against H5 is substantial. Fall back to H6 (E/I-balance recurrent) or H7 (metaplasticity as credit carrier).

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
