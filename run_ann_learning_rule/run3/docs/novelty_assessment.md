---
bibliography: ../references.bib
---

# Novelty Assessment: Proposed Biologically Plausible Learning Algorithms for RNNs

## Overview

This report assesses the novelty of two proposed learning algorithms for recurrent neural networks against the existing literature:

1. **H1**: Dendritic Prediction Error for Temporal Credit Assignment
2. **H2**: Oscillatory Phase-Gated Credit Assignment

## Hypothesis 1: Dendritic Prediction Error for Temporal Credit

### Proposed Mechanism

Each neuron in a recurrent network uses multi-compartment dendrites to compute LOCAL temporal prediction errors. The apical compartment receives a prediction of the neuron's own future state, while the basal compartment integrates current bottom-up input. The mismatch between these compartments drives an eligibility trace that is consolidated into weight updates when a global "surprise" signal arrives.

Key claim: This operates WITHIN recurrent dynamics (temporal credit), not across layers (structural credit).

### Closest Existing Work

**1. Sacramento et al. (2018) — "Dendritic cortical microcircuits approximate backpropagation"** [@sacramento2018]

This is the closest prior work. Sacramento et al. use multi-compartment pyramidal neurons where apical dendrites receive top-down error signals and basal dendrites receive bottom-up input. Local dendritic prediction errors (mismatch between lateral predictions and top-down feedback) drive plasticity.

**Critical distinction**: Sacramento et al. address *structural* credit assignment in feedforward/hierarchical networks, not *temporal* credit in recurrent networks. Their prediction errors arise from mismatch between layers, not from self-prediction of future states.

**2. Rao et al. (2021) — "A normative framework for learning top-down predictions through synaptic plasticity in apical dendrites"** [@rao2021]

Rao, Legenstein, and Maass propose that apical dendrites learn to predict top-down signals. This is about learning predictions in apical compartments, but again focuses on inter-layer prediction (hierarchical predictive coding), not temporal self-prediction within recurrent dynamics.

**3. Saponati & Vinck (2023) — "Sequence anticipation and spike-timing-dependent plasticity emerge from a predictive learning rule"** [@saponati2023]

This paper proposes a predictive learning rule where neurons learn a low-rank model of synaptic input dynamics in their membrane potential, allowing them to anticipate future inputs. This is close to our idea of self-prediction but differs in that: (a) it predicts *inputs* to the neuron rather than the neuron's own future *state*, (b) it doesn't use dendritic compartments explicitly for the prediction-error computation, and (c) it doesn't incorporate a modulatory surprise signal for gating consolidation.

**4. ELiSe (Kriener et al., 2024) — "Efficient Learning of Sequences in Structured Recurrent Networks"** [@kriener2024]

ELiSe uses dendritic compartments in recurrent networks for sequence learning with local, always-on, phase-free plasticity. However, it relies on a pre-existing network scaffold and focuses on replay/sequence generation rather than general temporal credit assignment.

**5. Learning Long-Range Dependencies with Temporal Predictive Coding (2026)** [@temporalPC2026]

This very recent paper extends predictive coding to RNNs for long-range temporal dependencies. It is the most direct competitor to H1, applying PC principles to recurrent temporal processing. However, standard temporal predictive coding propagates prediction errors between time steps in a chain-like fashion, rather than having each neuron self-predict its own future state via dendritic compartments.

**6. Predictive E-prop (Noè et al., 2026)** [@noe2026]

Combines predictive coding with e-prop in recurrent spiking networks. This merges two known approaches (predictive coding + eligibility propagation) for temporal credit.

### Novelty Assessment for H1

**Verdict: PARTIALLY NOVEL but with significant overlap concerns (~55% novel)**

The *specific* combination of (a) dendritic self-prediction of a neuron's own future state, (b) within recurrent dynamics for temporal credit, (c) gated by surprise-modulated consolidation has not been explicitly proposed. However:

- The dendritic compartment idea for error computation is well-established (Sacramento et al.)
- Predictive learning rules for temporal sequences exist (Saponati & Vinck)
- The "surprise gates eligibility trace" motif is essentially what ModProp and e-prop do
- Temporal predictive coding in RNNs is now actively being explored (2026 papers)

The novelty lies in the specific *self-prediction* mechanism (predicting own future state rather than inputs or outputs of other layers), but this may be considered a natural extension of existing work rather than a surprising insight.

## Hypothesis 2: Oscillatory Phase-Gated Credit Assignment

### Proposed Mechanism

Neural oscillations (theta/gamma) temporally multiplex forward computation and credit assignment into different phases. During "forward" phases, the network processes input normally through recurrent dynamics. During "credit" phases, the same physical connections propagate local error estimates backward in time. The key insight is that the SAME weights serve dual roles depending on oscillatory phase, eliminating the need for separate backward pathways.

### Closest Existing Work

**1. Hasselmo et al. (2002, 2024) — SPEAR model: "Separate Phases of Encoding and Retrieval"** [@hasselmo2002; @hasselmo2024]

Hasselmo's highly influential SPEAR model proposes that theta oscillation phases separate encoding from retrieval in the hippocampus. During one phase, new associations are encoded; during another, stored associations are retrieved. This is conceptually analogous to our proposal of separating forward computation from credit assignment.

**Critical distinction**: SPEAR separates *memory encoding* vs *retrieval*, not *forward computation* vs *gradient/credit propagation*. It does not address the credit assignment problem or propose that error signals travel through the same connections during a different phase.

**2. Multiplexing Neural Oscillations for Sequence Learning (2019)** [@maes2019]

Maes et al. show that multiple oscillatory signals can enable recurrent networks to generate complex temporal sequences. However, this is about using oscillations as a *scaffold for sequence representation*, not for multiplexing forward/backward information flow.

**3. Sacramento et al. (2018) with "no separate phases"** [@sacramento2018]

Sacramento et al. explicitly state their model "does not require separate phases" — learning is continuous. Some earlier models (like Boltzmann machines and contrastive Hebbian learning) DO use alternating phases, but these alternate between "free" and "clamped" states, not between forward and credit phases in the temporal sense.

**4. Contrastive Hebbian Learning / Equilibrium Propagation**

These methods alternate between a "free phase" and a "nudged phase" to compute gradients. This is the most structurally similar prior work to H2 — using different dynamical modes of the same network for computation vs. credit. However, these methods: (a) operate at equilibrium, not in sequential/temporal processing, (b) don't use biological oscillations as the phase-switching mechanism, and (c) don't specifically address temporal credit assignment in recurrent sequences.

**5. Wake-Sleep Algorithm**

Uses separate phases (wake for recognition, sleep for generation) but requires separate networks for the two directions, unlike our proposal where the same weights are reused.

### Novelty Assessment for H2

**Verdict: MODERATELY NOVEL (~65% novel) but with key challenges**

The specific proposal — using oscillatory phases to multiplex forward temporal processing and temporal credit assignment through the SAME physical connections — has not been explicitly proposed in the literature. The closest work is:

- Hasselmo's SPEAR (phase separation, but for encoding/retrieval, not credit assignment)
- Equilibrium propagation (dual-mode dynamics, but at equilibrium, not sequential)
- Contrastive Hebbian learning (alternating phases, but separate networks or equilibrium)

The novelty lies in: (a) applying phase-multiplexing specifically to *temporal* credit assignment in recurrent processing, (b) using the same weights bidirectionally by phase, and (c) grounding it in theta/gamma oscillations.

**Concern**: The mathematical feasibility is unclear. For the same weights to serve both forward processing and backward credit propagation, specific constraints on the weight matrices would be needed (e.g., near-symmetry or some form of alignment). This is a known hard problem — it's essentially the weight transport problem in temporal form.

## Comparative Summary

| Criterion | H1 (Dendritic Self-Prediction) | H2 (Oscillatory Phase-Gating) |
|-----------|-------------------------------|------------------------------|
| Novelty estimate | ~55% | ~65% |
| Biological grounding | Strong (dendritic compartments well-studied) | Moderate (theta phases exist, but dual-use unclear) |
| Mathematical feasibility | Moderate (eligibility traces + prediction are established tools) | Uncertain (weight reuse constraint is hard) |
| Closest prior work | Temporal Predictive Coding (2026), Saponati & Vinck (2023) | Hasselmo SPEAR + Equilibrium Propagation |
| Surprise factor | Low-moderate (natural extension) | Moderate-high (unexpected connection) |
| Risk of being scooped | Higher (active area, multiple 2026 papers) | Lower (less explored space) |

## Recommendations

1. **H2 is more novel and surprising** but carries higher mathematical risk. The "same weights, different phases" insight is genuinely unexpected and has not been proposed for temporal credit in RNNs.

2. **H1 is less novel** due to convergent work in temporal predictive coding (2026 papers), but is more mathematically tractable and has stronger biological grounding.

3. **Consider combining elements**: The self-prediction component of H1 could be integrated into H2's framework — oscillatory phases gate between "predict forward" and "propagate credit," with dendritic compartments as the local error computation mechanism.

4. **Before committing to H2**: A mathematical proof-of-concept is needed to show that the same weight matrix can meaningfully propagate credit backward when used in a different oscillatory phase. If this requires near-symmetric weights, it reduces to a variant of feedback alignment.
