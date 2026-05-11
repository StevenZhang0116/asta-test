# Research State — Biologically Plausible RNN Learning Rule (run4, independent)

*Independent replicate — do not reference sibling `runN/` directories.*

---

## 1. Research Question & Scope

**Primary question.** What new learning rule can we design for recurrent neural networks that addresses known biological implausibilities of BPTT — weight transport, non-locality, non-causality, global error signals, memory cost, offline-ness, separate forward/backward phases, continuous-time incompatibility — while remaining competitive on ML tasks, and what biological principles should it be grounded in?

**Scope.**
- Architectures: vanilla RNN first, then GRU/LSTM.
- Tasks: short supervised temporal tasks (T ~10–50) first, then longer; later extend to RL / unsupervised.
- Deliverable: a *new* algorithm — not a survey. Must satisfy a novelty + "surprise" bar.

**Non-scope (for now).**
- Transformers, deep feed-forward only models, continuous-time spiking networks as the *primary* target (we may draw on them).
- Reproducing any specific paper verbatim.

---

## 2. Operational Definitions

- **Biologically plausible (working definition, to be refined):** a learning rule that satisfies, at minimum, (a) no weight transport (backward path ≠ transpose of forward path), (b) locality (each synaptic update uses only pre-synaptic activity, post-synaptic activity, and scalar/low-dimensional modulatory signals available at the synapse), (c) online or near-online operation (bounded memory, no requirement to store the full trajectory), (d) no separate backward pass requiring frozen forward activity for the whole sequence.
- **Competitive on ML tasks (working):** within ~1.5× the error of BPTT on matched architectures for short-sequence supervised tasks (copy, delayed-match, add-task, permuted-MNIST-subset).
- **Novelty (mission):** not previously described in the literature, re-verified at every material design change.
- **Surprise (mission):** not a trivial combination of known ingredients.

---

## 3. Related Work

*To be populated by a literature search. Mission references Lillicrap 2016 (feedback alignment), Murray 2019 (RFLO), Liu 2022 (ModProp) only as *subfield disambiguators* — these are NOT the Related Work list and must not bias the novelty audit.*

Planned first-pass search axes:
1. Biologically plausible credit assignment in RNNs (eligibility traces; e-prop; RTRL approximations; RFLO; ModProp; SnAp-k; UORO; KF-RTRL).
2. Weight-transport-free feedback (feedback alignment; direct feedback alignment; sign-concordant feedback; weight-mirror dynamics).
3. Neuromodulation-gated plasticity (three-factor rules; dopamine-like TD signals; vector-valued modulation).
4. Dendritic / compartmental learning (Sacramento 2018; Guergiuev; Payeur 2021 burst-dependent plasticity).
5. Predictive coding / energy-based local learning in recurrent settings.

---

## 4. Hypotheses

Prior beliefs (rough, to be tested):

- **H1 (eligibility-trace primacy).** A local eligibility trace that tracks the sensitivity of hidden state to recent synaptic activity, combined with a low-rank / random projection of the error signal, can approximate BPTT gradients well enough for short tasks. *Prior confidence: 60%.*
- **H2 (non-random feedback beats random).** Structured but *non-symmetric* feedback (e.g., sign-concordant, learned via a Hebbian-like "mirror" rule, or gated by activity statistics) will outperform purely random feedback alignment on sequential tasks. *Prior confidence: 55%.*
- **H3 (modulatory bottleneck is sufficient).** A scalar or low-dimensional (≤10) neuromodulatory signal, broadcast globally, is sufficient to carry the task-relevant error for short supervised tasks *if* the eligibility traces are expressive enough. *Prior confidence: 40%.*
- **H4 (temporal decomposition).** Decomposing credit assignment into (i) a purely causal forward-propagated "responsibility" signal and (ii) an *anticipatory* predictive signal learned by an auxiliary network can replace BPTT's backward pass without loss. *Prior confidence: 35% — the most "surprise"-worthy if it works.*

These are initial priors; the research loop may refine, split, or kill any of them.

---

## 5. Experimental Designs

To be filled as experiments are defined. Standard harness anticipated:

- Tasks: copy task (T=10–30), delayed XOR, adding problem, sequential MNIST (first on a subset).
- Baselines: BPTT, truncated BPTT, RFLO, e-prop, feedback alignment (FA), random recurrent feedback.
- Metrics: final loss/accuracy, wall-clock per epoch, memory footprint, gradient-alignment cosine with true BPTT gradient.
- Ablations: remove modulatory signal, remove eligibility trace, freeze feedback weights vs. learn them.

---

## 6. Results Summary

*None yet — research has just started in run4.*

---

## 7. Open Questions & Confusions

- What is the *tightest* formulation of "locality" that is still useful? Some papers allow layer-wise error signals; others only allow synapse-local.
- Is the weight transport problem *the* critical issue, or is non-causality (needing future information) more binding for temporal tasks?
- Is there a regime where a purely Hebbian recurrent rule + smart modulation is competitive, or is some form of eligibility trace mandatory?
- What is the right "surprise" direction — a new biological mechanism not yet exploited (e.g., synaptic tagging with *two* timescales, astrocyte-mediated slow modulation, dendritic spikes gating plasticity), or a new mathematical decomposition of the credit assignment problem?

---

## Next Tasks (priority order)

1. **[NEXT]** Run a focused literature search to populate §3 Related Work, identify the *current frontier* of biologically plausible RNN learning rules, and surface candidate gaps for a novel design. This is the prerequisite for any novelty audit and for choosing a first concrete design to prototype.
2. Formulate a first concrete candidate learning rule (equations + pseudocode), pass a novelty audit.
3. Implement a minimal experimental harness (vanilla RNN + copy task) and compare candidate vs. BPTT + FA + RFLO baselines.
4. Iterate.
