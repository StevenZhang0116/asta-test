# Research State: Biologically Plausible Learning Rule for RNNs

**Last updated**: 2026-05-10 (after Step 3 — BPTT baseline)

## 1. Research Question & Scope

**Primary question**: What new learning rule can we design for recurrent systems that addresses known biological implausibilities of BPTT (weight transport, non-locality, non-causal credit assignment, global error signals, memory requirements, offline operation, separate forward/backward phases) while achieving competitive performance on short-sequence supervised ML tasks?

**Deliverable**: A novel online, local learning algorithm for vanilla RNNs (later extended to GRU/LSTM), grounded in concrete biological mechanisms.

**Scope anchors** (per `mission.md`, NOT a baseline list): Lillicrap 2016 (FA), Murray 2019 (RFLO), Liu 2022 (ModProp). Related work was built from scratch at Step 1.

## 2. Operational Definitions

Unchanged from initialization.

- **Biological plausibility** (checklist): no weight transport (WT), locality (Loc), online / bounded memory (Onl), no separate forward/backward phase (NoP), causality (Cau).
- **Competitive performance**: ≤ ~2× BPTT error on same architecture at short supervised tasks.
- **Novelty** and **surprise**: per `mission.md`.

## 3. Related Work (summary — see `literature_report_step1.md` for the full report)

Step 1 surveyed 68 papers across six clusters; full analysis is in `literature_report_step1.md` (68 citations rendered cleanly via pandoc+citeproc, all resolving to `references.bib`).

**High-level structure of the field**:
- **Forward-mode online credit**: RTRL → UORO → SnAp → RFLO → e-prop. RFLO and e-prop are the two biologically-plausible options (local, online, causal). e-prop has BPTT-comparable performance on TIMIT; RFLO trails BPTT but is the cleanest local rule.
- **Random feedback**: FA / DFA / sign-concordant / weight mirrors address weight transport. In RNN settings, RFLO (random feedback + eligibility trace) and ModProp (cell-type-specific multiplicative modulators that collectively reconstruct the recurrent Jacobian) are the key implementations.
- **Predictive coding & target propagation**: PC approximates BP under an infinitesimal-inference limit (Whittington/Bogacz/Millidge/Song); local Hebbian updates on prediction errors; typically not online (iterates to convergence). TP (Lee 2015, Meulemans 2020, Ernoult 2022) replaces gradients with learned inverse targets; two-phase.
- **Dendritic compartments**: Guerguiev 2017 / Sacramento 2018 / Burstprop (Payeur 2021) / Greedy 2022 single-phase / Max 2022. Mostly static inputs; temporal extensions are open.
- **Synaptic tagging & capture (STC)**: Clopath 2008, Benna-Fusi 2016, Luboeinski 2020, Zenke 2024. Two-timescale consolidation (fast tag + slow capture). **Largely unused as a credit-assignment mechanism — mostly deployed for continual-learning / memory consolidation.**
- **Node / weight perturbation**: Miconi 2017 trains RNNs with R·ξ·r node-perturbation three-factor rule; satisfies every plausibility criterion but has high variance. Fernandez 2025 revisits with variance-reduction.
- **Three-factor / neuromodulation / synthetic gradients / EqProp**: cross-cutting glue that the above methods fit into.

**Strongest current baselines** for Step 2 to beat (or match while being more biologically plausible):
- **e-prop** — BPTT-competitive on short supervised tasks, online, local, needs global broadcast.
- **RFLO** — fully local, simple, biased gradient; trails BPTT.
- **ModProp** — cell-type-specific modulators reconstruct multi-step Jacobian; hand-designed gains.

## 4. Hypotheses

**H1 (primary candidate, now specified as STC-Credit)** — *confidence: ~40%, up from 35% post-Step-1* — A two-timescale **eligibility-tag + surprise-thresholded capture** rule (full spec in `design_step2.md`). Per-synapse fast tag integrates RFLO-style φ'(u_i)·x_j with decay α_τ. A random-feedback teaching signal ℓ_i = B_i·e is z-scored into a per-neuron surprise s_i; a threshold θ gates capture; Δw_{ij} = -η · g_i · ℓ_i · τ_{ij}. Biological metaphor: fast tag = AMPAR/CaMKII priming; capture gate = PRP/late-LTP consolidation driven by salient neuromodulatory events; random feedback = divergent teaching projection. **Novelty audit verdict (Step 2): novel.** Closest predecessor is Izhikevich 2007 (continuous dopamine × STDP eligibility for distal reward, RL task). STC-Credit differs in (i) thresholded rather than continuous capture, (ii) supervised-RNN temporal-credit evaluation, (iii) explicit RFLO/e-prop-style eligibility. Residual risk: an unsurfaced SNN-literature precedent (different vocabulary); follow-up targeted search scheduled.

**H2 (alternative)** — *confidence: ~15%* — A dendritic-compartment RNN where each hidden unit's apical compartment receives a *temporally lagged* predicted hidden state from a slow-timescale neuromodulatory broadcaster. The apical–basal mismatch gates plasticity. Ties dendritic and STC traditions together but heavier biological machinery than H1. Deferred unless H1 is refuted.

**H3 (fallback)** — *confidence: ~10%* — Node perturbation with **eligibility-gated accumulation**: the standard Miconi rule but where each per-step contribution ξ_i·r_j is multiplied by a local "informativeness" factor to reduce variance. Simpler but possibly not novel.

## 5. Experimental Designs

**E1 (Step 3, next)** — Establish BPTT baseline on a short supervised task.
- Architecture: vanilla RNN, hidden size 64–128, tanh activations.
- Task: pick one of {copy task (length 10, 8 symbols), delayed-match-to-sample, adding problem (length 20)}. Leaning toward **copy task length 10** — simplest unambiguous temporal credit assignment, used by e-prop and RFLO for parity comparisons.
- Metric: final loss, steps-to-convergence, wall-clock, accuracy.
- Device: GPU if available, CPU fallback.
- Tool: `run-experiment` (Panda), not ad-hoc Python (per CLAUDE.md).

**E2** — Reproduce RFLO and e-prop baselines on the same task — gives a "biologically plausible reference" we need to beat.

**E3** — Implement H1 (STC-inspired two-timescale rule) and compare to E1/E2.

**E4** — Ablations: remove the gating threshold (should reduce to RFLO-style continuous update); remove the tag (should degenerate to instantaneous Hebbian); remove the broadcaster (should destroy credit assignment).

## 6. Results Summary

- **Step 1 (2026-05-10)**: Literature survey covering 68 papers across six clusters. Key outputs: `literature_report_step1.md`, `references.bib` (both in run5/; rendered cleanly via pandoc+citeproc). The single most promising gap identified is **two-timescale STC-style consolidation as a temporal-credit-assignment mechanism**, which is underexplored relative to continuous eligibility-trace decay.
- **Step 2 (2026-05-10)**: Specified STC-Credit in `design_step2.md`. Full update equations, biological metaphor, novelty audit against 7 targeted queries, plausibility checklist (passes all 5 desiderata), falsifiability criteria, and default hyperparameters. Audit outcome: novel; closest predecessor is Izhikevich 2007 (continuous, not thresholded; RL not supervised). Residual audit gap: SNN-literature vocabulary check deferred to Step 4. `references.bib` now contains 73 entries (5 added in Step 2 for near-matches: izhikevich2007, luboeinski2021, rombouts2015, huertas2014, tsurumi2025, fernandez2025).
- **Step 3 (2026-05-10)**: BPTT baseline on copy task via `run-experiment` (Panda). Vanilla Elman RNN with explicit recurrence (128 hidden, tanh, 18,824 params), Adam lr=1e-3, batch 64, 3000 steps, evaluated every 100 steps on 1000 held-out sequences. Ran on A100 GPU (total 24.6s, ~0.82s per 100 steps). **Result: 100% per-symbol accuracy and 100% sequence-level accuracy by step 200; 90% threshold crossed at step 100 (first evaluation).** Loss trajectory: 2.16 → 0.18 (step 100) → 0.04 (step 200) → 2e-4 (step 3000). Artifacts in `.asta/experiment/2026-05-10-bptt-copy-baseline/`. Single seed, single run (caveat). This establishes the reference: biologically-plausible rules on this task should reach >95% per-symbol accuracy within 3000 steps to be "within reasonable range of BPTT"; the comparable-speed landmark is "crosses 90% by step 100".

## 7. Open Questions & Confusions

- **Novelty of H1 vs. Luboeinski & Tetzlaff 2020**: that paper puts STC into *recurrent* networks for *memory consolidation*; I need to verify in Step 2 that using STC for *gradient credit assignment* in supervised RNN learning is genuinely unexplored. Targeted S2 lookup planned.
- **Novelty of H1 vs. e-prop with surprise-weighting**: does any e-prop variant already gate the eligibility-trace update by a local prediction error? A Step-2 targeted lookup.
- **Surprise gating vs. continuous modulation**: is a thresholded capture step really more expressive than a smoothly-modulated trace? An ablation (E4) will tell, but a theoretical argument would strengthen the design.
- **Continuous-time vs. discrete-time formulation**: discrete-time is easier for ML benchmarks; continuous-time maps better to biology. Step 2 will pick and justify.
- **Choice of short task for Step 3**: copy task is my current leaning, but sequence length 10 may be too easy to meaningfully discriminate BPTT from local rules. May need to run at length 20 or use adding problem if copy-10 saturates both.

## 8. Next Steps

1. **[Step 4, next]** SNN-vocabulary novelty audit (residual risk from Step 2 §4.3). Target queries: "event-driven plasticity SNN", "protein-synthesis threshold plasticity SNN", "gated STDP recurrent", "threshold-triggered consolidation spiking network". Goal: close the main residual uncertainty in H1's novelty claim before investing in the implementation. If a near-match is found, revise H1 (either distinguish, or pivot to H2/H3).
2. **[Step 5]** Reproduce **RFLO** and **e-prop** baselines on the same copy task (length-15, 5 symbols, alphabet-8, same architecture/hyperparameters as the BPTT baseline). Gives a "biologically-plausible reference" — STC-Credit needs to at least match these to be interesting.
3. **[Step 6]** First STC-Credit implementation on the copy task. Metrics: same as Steps 3/5. Ablations: F2 (θ = −∞ ⇒ continuous), F3 (θ forced low ⇒ dense gating), plus an additional ablation removing the tag (pure delta rule).

**Performance targets established at Step 3** (to be used in Steps 5–6):
- Minimum acceptable per-symbol accuracy: **>95%** (BPTT reaches 100%).
- Minimum acceptable sequence-level accuracy: **>95%** (BPTT reaches 100%).
- Speed landmark: **step 100** (BPTT crosses 90% at first eval).

**Limitations to keep in mind**: Copy task with T=15 is deliberately easy; BPTT saturates quickly, so the gap between BPTT and local rules may be small here but much larger on longer tasks. A Step-7+ extension should re-run on seq length 30 and on an adding problem to test robustness.
