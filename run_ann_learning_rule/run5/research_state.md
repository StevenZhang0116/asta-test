# Research State: Biologically Plausible Learning Rule for RNNs

**Last updated**: 2026-05-10 (after Step 5 — RFLO / rate e-prop baselines)

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

**H1 (primary candidate, now specified as STC-Credit)** — *confidence: ~45% after Step 4; up from ~40% post-Step-2* — A two-timescale **eligibility-tag + surprise-thresholded capture** rule (full spec in `design_step2.md`). Per-synapse fast tag integrates RFLO-style φ'(u_i)·x_j with decay α_τ. A random-feedback teaching signal ℓ_i = B_i·e is z-scored into a per-neuron surprise s_i; a threshold θ gates capture; Δw_{ij} = -η · g_i · ℓ_i · τ_{ij}. Biological metaphor: fast tag = AMPAR/CaMKII priming; capture gate = PRP/late-LTP consolidation driven by salient neuromodulatory events; random feedback = divergent teaching projection. **Novelty audit verdicts:** Step 2 = novel vs. three-factor / e-prop / RFLO / ModProp; Step 4 = novel vs. SNN-literature adjacencies (Lehr 2022 — STC for memory consolidation; Yamada 2025 — attention-gated reservoir; TESS 2025 — synchronization-based SNN rule; AGMP 2026 — activity-gated astrocyte rule for continual learning). Narrower margin after Step 4, but the specific combination *(i) binary/thresholded capture + (ii) error-driven surprise + (iii) supervised-RNN credit assignment* remains unclaimed. A new Step-6 ablation (activity-gated baseline, AGMP-style) is added to isolate the *error-driven* claim.

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
- **Step 4 (2026-05-10)**: SNN-vocabulary novelty audit for STC-Credit (§4.4 of `design_step2.md`). Seven targeted queries. Surfaced four adjacencies requiring careful reading: Lehr 2022 (STC in SNNs for memory, continuous neuromodulator), Yamada 2025 (attention-gated reservoir), TESS 2025 (synchronization-based SNN rule — strongest local-SNN competitor), AGMP 2026 (astrocyte-gated multi-timescale plasticity for continual learning). **Verdict: novel — the specific combination of binary-thresholded capture, error-driven surprise, and supervised-RNN credit-assignment is unclaimed.** Added 4 entries to `references.bib` (77 total). Updated falsifiability plan: Step 6 now includes an AGMP-style activity-gated baseline, not just continuous (θ=−∞) ablations, to isolate the error-driven aspect of STC-Credit's gate.
- **Step 5 (2026-05-10)**: RFLO / rate-e-prop baseline on the same copy task via `run-experiment` (Panda). Implemented with no autograd: explicit eligibility trace p^W_{ij} = (1-α)p^W_{ij} + α·φ'(u_i)·h_j, fixed random feedback B ~ N(0, 1/√8), teaching signal ℓ = B·e active on output timesteps only, plain-SGD update W += -η·outer(ℓ, p^W). Key clarification: for a rate RNN with linear readout, RFLO and rate e-prop produce identical updates, so one experiment covers both. Swept η ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}; **best at η=1e-4, reaching 55.3% per-symbol accuracy, 2.7% sequence-level accuracy, never crossing 90% in 3000 steps**. Default η=3e-3 yielded only 34.1% — lower LR needed. Wall-clock 0.85 s / 100 steps on A100 (vs 0.82 s for BPTT; ~4% overhead, essentially free). **The BPTT↔RFLO gap is 44.7 pp per-symbol accuracy.** This is the credit-assignment floor STC-Credit must close. Artifacts in `.asta/experiment/2026-05-10-rflo-eprop-copy/`. Single-seed caveat; best config had not fully converged at step 3000 (asymptote unknown).

## 7. Open Questions & Confusions

- **Novelty of H1 vs. Luboeinski & Tetzlaff 2020**: that paper puts STC into *recurrent* networks for *memory consolidation*; I need to verify in Step 2 that using STC for *gradient credit assignment* in supervised RNN learning is genuinely unexplored. Targeted S2 lookup planned.
- **Novelty of H1 vs. e-prop with surprise-weighting**: does any e-prop variant already gate the eligibility-trace update by a local prediction error? A Step-2 targeted lookup.
- **Surprise gating vs. continuous modulation**: is a thresholded capture step really more expressive than a smoothly-modulated trace? An ablation (E4) will tell, but a theoretical argument would strengthen the design.
- **Continuous-time vs. discrete-time formulation**: discrete-time is easier for ML benchmarks; continuous-time maps better to biology. Step 2 will pick and justify.
- **Choice of short task for Step 3**: copy task is my current leaning, but sequence length 10 may be too easy to meaningfully discriminate BPTT from local rules. May need to run at length 20 or use adding problem if copy-10 saturates both.

## 8. Next Steps

1. **[Step 6, next]** First STC-Credit implementation on the copy task. Same architecture as Steps 3 and 5 (vanilla Elman RNN, N=128). Use η=1e-4 as starting point (learned from Step-5 sweep). Ablations in one experiment:
   - **Main**: STC-Credit with surprise threshold θ tuned so capture rate is ~5–20%.
   - **F2** (ablation): θ = −∞ (continuous modulation) — this IS RFLO; should replicate Step-5 (55.3%).
   - **F3** (ablation): θ forced low so capture rate is ≥80% — should also approximate RFLO.
   - **AGMP-style** (ablation from Step 4): gate by EMA of |h_i| (activity-driven), not by error-surprise — tests whether *error-driven* gating is what matters.
   - **No-tag** (ablation): pure delta rule (no eligibility).
   
   Report the same metrics as Steps 3 & 5. Key question: does the error-surprise gating close any of the 44.7 pp gap?

2. **[Step 7+]** If STC-Credit shows promise, scale to T=30 and adding problem; multiple seeds; characterize what fraction of the gap the capture gate closes vs. what comes from other sources.

**Performance landscape (from Steps 3 & 5)**:
| Rule | Per-symbol acc | Seq acc | 90%-at-step | Time/100 steps |
|---|---|---|---|---|
| BPTT (Step 3) | 100.0% | 100.0% | 100 | 0.82 s |
| RFLO / rate e-prop (Step 5, η=1e-4) | 55.3% | 2.7% | never | 0.85 s |
| STC-Credit (Step 6, pending) | ? | ? | ? | ~0.85 s (expected) |

STC-Credit needs to show statistically-meaningful improvement over the RFLO/e-prop floor to be worth pursuing further. Mission's "within reasonable range of BPTT" (>95%) is a stretch target; closing even half the 44.7 pp gap would be a real result.

**Performance targets established at Step 3** (to be used in Steps 5–6):
- Minimum acceptable per-symbol accuracy: **>95%** (BPTT reaches 100%).
- Minimum acceptable sequence-level accuracy: **>95%** (BPTT reaches 100%).
- Speed landmark: **step 100** (BPTT crosses 90% at first eval).

**Limitations to keep in mind**: Copy task with T=15 is deliberately easy; BPTT saturates quickly, so the gap between BPTT and local rules may be small here but much larger on longer tasks. A Step-7+ extension should re-run on seq length 30 and on an adding problem to test robustness.
