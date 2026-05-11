# Research State: Biologically Plausible Learning Rule for RNNs

**Last updated**: 2026-05-10 (after Step 1 — literature survey)

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

**H1 (primary candidate for Step 2)** — *confidence: ~35%, up from 25% at init* — A two-timescale **eligibility-tag + surprise-gated consolidation** rule: each synapse keeps a fast decaying eligibility tag (as in RFLO/e-prop); a *separate*, slow modulatory broadcaster monitors local prediction error; only when the broadcaster crosses a "surprise threshold" does the accumulated tag commit into a weight change. This is a concrete STC-inspired credit-assignment mechanism — STC has been used for continual learning and memory consolidation, but **not** as a temporal-credit-assignment mechanism for RNNs. Specifically:

    tag_{ij}(t) = (1 − α_fast) · tag_{ij}(t−1) + α_fast · φ'(u_i(t)) · x_j(t)
    surprise(t)  = f( local prediction error )
    Δw_{ij}(t)   = η · B_i · surprise(t) · tag_{ij}(t) · 1{surprise(t) > θ}

    plus a slow decay of the tag when no capture happens.

    **What's potentially surprising**: replacing continuous eligibility-trace decay with a *gated* consolidation step directly analogous to the late-LTP / PRP biology; plus the "surprise" broadcast is computed from a *local* prediction error (à la PC) rather than from a global supervised loss.

**H2 (alternative)** — *confidence: ~15%* — A dendritic-compartment RNN where each hidden unit's apical compartment receives a *temporally lagged* predicted hidden state from a slow-timescale neuromodulatory broadcaster. The apical–basal mismatch gates plasticity. This ties the dendritic and STC traditions together but requires substantially more biological machinery than H1.

**H3 (fallback)** — *confidence: ~10%* — Node perturbation with **eligibility-gated accumulation**: the standard Miconi node-perturbation rule but where the per-step contribution ξ_i · r_j is multiplied by a local "informativeness" factor (e.g., local prediction error) to reduce variance. Simpler but possibly not novel.

*Old H2/H3 from init are subsumed into H1/H2/H3 above.*

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

No experimental results yet.

## 7. Open Questions & Confusions

- **Novelty of H1 vs. Luboeinski & Tetzlaff 2020**: that paper puts STC into *recurrent* networks for *memory consolidation*; I need to verify in Step 2 that using STC for *gradient credit assignment* in supervised RNN learning is genuinely unexplored. Targeted S2 lookup planned.
- **Novelty of H1 vs. e-prop with surprise-weighting**: does any e-prop variant already gate the eligibility-trace update by a local prediction error? A Step-2 targeted lookup.
- **Surprise gating vs. continuous modulation**: is a thresholded capture step really more expressive than a smoothly-modulated trace? An ablation (E4) will tell, but a theoretical argument would strengthen the design.
- **Continuous-time vs. discrete-time formulation**: discrete-time is easier for ML benchmarks; continuous-time maps better to biology. Step 2 will pick and justify.
- **Choice of short task for Step 3**: copy task is my current leaning, but sequence length 10 may be too easy to meaningfully discriminate BPTT from local rules. May need to run at length 20 or use adding problem if copy-10 saturates both.

## 8. Next Steps

1. **[Step 2, next]** Novelty-informed design. Write `design_step2.md` specifying H1's update equations, biological metaphor, and a novelty audit. The audit must include targeted Semantic Scholar lookups against the closest near-matches: (a) "synaptic tagging capture eligibility RNN temporal credit", (b) "e-prop surprise gated", (c) "predictive coding modulated eligibility trace".
2. **[Step 3]** BPTT baseline experiment via `run-experiment` (Panda). Vanilla RNN on copy task (length 10 or 20). Gives the reference number for H1 to beat in Step 4.
3. *[future]* E2 (RFLO/e-prop baselines) and E3 (H1 implementation) in Steps 4–5.
