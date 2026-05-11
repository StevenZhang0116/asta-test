# Research State: Biologically Plausible Learning Rule for RNNs

**Last updated**: 2026-05-10 (Step 0 — initialization)

## 1. Research Question & Scope

**Primary question**: What new learning rule can we design for recurrent systems that addresses known biological implausibilities of BPTT — weight transport, non-locality, non-causal credit assignment, global error signals, memory requirements, offline operation, separate forward/backward phases, etc. — while achieving competitive performance on short-sequence supervised ML tasks?

**Deliverable**: A novel online, local learning algorithm for vanilla RNNs (later extended to GRU/LSTM), grounded in concrete biological mechanisms (eligibility traces, neuromodulation, dendritic compartments, synaptic tagging, Hebbian/anti-Hebbian, etc.).

**Scope anchors (per `mission.md`, NOT a baseline list)**: random feedback alignment (Lillicrap 2016), RFLO (Murray 2019), ModProp (Liu 2022). Related work must be built from scratch — these three only disambiguate the subfield.

**Out of scope for now**: RL and unsupervised variants (come later); long sequences (start at length 10–50); gated architectures (start with vanilla RNN).

## 2. Operational Definitions

- **Biological plausibility** (operationalized as a checklist):
  1. No weight transport (feedback pathway must not share weights with forward pathway).
  2. Local updates: Δw_{ij} depends only on pre-synaptic activity x_j, post-synaptic activity y_i (and its local derivatives), and at most one diffuse modulatory scalar / low-dim vector per layer or neuron.
  3. Online or near-online: memory footprint bounded by O(N²) state (the recurrent weight matrix size), NOT O(T·N) activity history.
  4. No separate backward phase: updates happen during or immediately after each forward step.
  5. Causal: updates at time t use only information available at time ≤ t (possibly delayed by a few steps for neuromodulation).
- **Competitive performance**: within some reasonable multiplicative factor of BPTT on short supervised tasks (copy, delayed-match, adding problem, sMNIST at length 50). Target: ≤ 2× the BPTT error on the same architecture/capacity, or equal-accuracy with modest extra training time.
- **Novelty**: the *mechanism* (update equation + biological metaphor) is not previously described. Re-checked whenever the design materially changes.
- **Surprise**: the design reflects a conceptual insight, not a trivial mash-up of two existing techniques.

## 3. Related Work

*To be filled in at Step 1.* A dedicated literature survey is the first research action; the anchor citations (Lillicrap, Murray, Liu) will be included only alongside the full picture surfaced by that search.

## 4. Hypotheses

(Initial, low-confidence seeds — to be refined after the literature survey.)

- **H1 (working hypothesis for a candidate mechanism)** — *confidence: 25%* — An online RNN learning rule that combines (a) per-synapse eligibility traces (forward-mode credit), (b) random fixed feedback projections of a local prediction error, and (c) a slow-timescale modulatory gate that controls when eligibility traces consolidate, can reach BPTT-competitive performance on short tasks while satisfying all locality criteria. The novelty would be in the gating rule, not in (a) or (b) individually.
- **H2** — *confidence: 15%* — A dendritic-compartment formulation, where each hidden unit computes a local "prediction" in an apical compartment and the mismatch between apical and basal activity drives plasticity, can replace the need for a global error broadcast entirely.
- **H3** — *confidence: 10%* — A "synaptic tagging and capture" formulation — a two-timescale update where fast tags mark recently active synapses and a slow consolidation step (triggered by a sparse modulatory event) commits them — may produce better temporal credit assignment than continuous eligibility decay.

Open space: we have not yet identified the *surprising* ingredient the mission explicitly demands. Finding it is itself a research subgoal.

## 5. Experimental Designs

(None yet. Planned once a first design is specified.)

Rough plan once we have a candidate rule:
- **E1**: Baseline parity check. Vanilla RNN trained with BPTT on a copy task (seq len 10, 5 distinct symbols). Establish the target error and wall-clock.
- **E2**: Same task/architecture with the candidate rule. Compare.
- **E3**: Ablations (remove eligibility, remove neuromodulation, etc.) to isolate which ingredient matters.

## 6. Results Summary

(None yet.)

## 7. Open Questions & Confusions

- What precisely is the "surprise" beyond known ingredients (eligibility + random feedback + neuromodulation)? This is the hardest open question and should shape Step 1's literature search.
- Should the first design target vanilla RNN forward dynamics h_t = tanh(W h_{t-1} + U x_t), or a rate-based continuous-time formulation dh/dt = -h + f(Wh + Ux)? The continuous-time version may map more cleanly onto biology but slightly complicates the ML comparison.
- What short supervised task gives the sharpest BPTT-vs-local-rule signal? Copy task vs. adding problem vs. sMNIST — need to pick 1–2.
- For novelty checks: how broadly should the search cover (e-prop, REINFORCE-style noise perturbation, target propagation variants, predictive coding networks)?

## 8. Next Steps

1. **[Step 1, next]** Literature survey — survey recent work on biologically plausible RNN learning rules to build a real Related Work section and identify where the genuine novelty gap is. Use the `literature-report` skill.
2. **[Step 2]** Novelty-informed design — propose a concrete candidate learning rule, specifying update equations, biological metaphor, and a novelty audit against the literature found in Step 1.
3. **[Step 3]** Sanity experiment — implement vanilla RNN + BPTT baseline on a short task (copy/adding, seq len ≤ 20) via the `run-experiment` skill, to establish the comparison target.
