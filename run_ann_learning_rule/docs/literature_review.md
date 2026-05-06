# Literature Review: Biologically Plausible Learning Rules for Recurrent Neural Networks

## Overview

This review surveys methods (2016-2025) that address the biological implausibilities of Backpropagation-Through-Time (BPTT) for training recurrent neural networks. We focus on four criticisms: (1) weight transport, (2) non-locality, (3) non-causality, and (4) global error signals. For each method, we describe its mechanism, biological metaphor, supported architectures, benchmark tasks, and limitations.

## 1. Feedback Alignment (Lillicrap et al., 2016)

**Paper:** "Random synaptic feedback weights support error backpropagation for deep learning" — Nature Communications

**Mechanism:** Replaces the transpose of the forward weight matrix in the backward pass with a fixed random matrix B. Errors are propagated using e = B * δ instead of e = W^T * δ. Over training, the forward weights W align with the random backward weights B, enabling effective credit assignment.

**Biological metaphor:** Distinct, non-symmetric feedback pathways in the brain (e.g., separate populations of feedback neurons with fixed random connectivity).

**What it solves:**
- Weight transport: YES (no symmetric weights needed)
- Non-locality: Partially (still needs layer-wise error propagation)
- Non-causality: Not addressed (feedforward setting)

**Architectures:** Feedforward networks. Not directly applicable to RNNs, but foundational for later recurrent extensions (RFLO, e-prop).

**Tasks tested:** MNIST, CIFAR-10, regression tasks.

**Limitations:** Performance degrades in very deep networks. Does not address temporal credit assignment. Feedforward only.

---

## 2. RFLO — Random Feedback Local Online Learning (Murray, 2019)

**Paper:** "Local online learning in recurrent networks with random feedback" — eLife

**Mechanism:** Derives an online learning rule for vanilla RNNs by taking a rank-1 approximation to the influence matrix (how past activity affects present). The weight update is:

ΔW ∝ e_t * (B * error_t)^T

where e_t is an eligibility trace that evolves as:

e_t = (1 - α)e_{t-1} + α * φ'(h_t) * x_t^T

and B is a fixed random feedback matrix replacing W^T.

**Biological metaphor:** 
- Eligibility traces = synaptic tags marking recently active synapses
- Random feedback B = distinct feedback pathways (as in feedback alignment)
- Modulation by error = neuromodulatory "now print" signal

**What it solves:**
- Weight transport: YES (random B)
- Non-locality: YES (only local pre/post-synaptic info + global error signal)
- Non-causality: YES (fully online, causal)

**Architectures:** Vanilla RNN (continuous-time formulation with tanh nonlinearity). Limited/no extension to gated architectures.

**Tasks tested:** Sine wave generation, pattern generation, delayed match-to-sample, evidence accumulation.

**Limitations:**
- Rank-1 approximation to influence matrix loses information about complex temporal dependencies
- Performance degrades for long sequences where multi-step credit assignment is needed
- Not extended to LSTM/GRU — unclear how to handle gating
- Still requires a global error signal (though this is less biologically problematic than weight transport)

---

## 3. e-prop — Eligibility Propagation (Bellec et al., 2020)

**Paper:** "A solution to the learning dilemma for recurrent networks of spiking neurons" — Nature Communications

**Mechanism:** Factors the BPTT gradient into:
- An eligibility trace (forward-computed, local): tracks how each synapse's recent activity would affect the postsynaptic neuron
- A learning signal (backward-projected): carries top-down error information

The gradient is approximated as: ΔW ∝ eligibility_trace * learning_signal

Three variants:
- e-prop1: symmetric (uses W^T for learning signal, not biologically plausible)
- e-prop2: random feedback (uses random B)
- e-prop3: broadcast alignment (single global learning signal)

**Biological metaphor:**
- Eligibility traces = synaptic tagging mechanisms
- Learning signals = neuromodulatory signals (dopamine-like)
- Aligns with "three-factor learning rules" from neuroscience (pre × post × modulator)

**What it solves:**
- Weight transport: YES (in e-prop2 and e-prop3 variants)
- Non-locality: YES (eligibility is local; learning signal is global but simple)
- Non-causality: YES (forward-computed traces, no future information)

**Architectures:** Leaky Integrate-and-Fire (LIF) spiking neurons, ALIF (adaptive LIF). Designed for spiking networks specifically. Applicable to rate-based networks with adaptation.

**Tasks tested:** Temporal XOR, speech recognition (TIMIT), evidence accumulation, store-recall tasks, reinforcement learning (ATARI-like).

**Limitations:**
- Designed for spiking neural networks — not directly for standard RNNs (tanh, ReLU)
- The factored approximation drops "symmetric e-prop" terms needed for exact gradients
- Performance gap with BPTT grows on tasks requiring precise long-range dependencies
- Gated architecture extension not addressed (no LSTM/GRU formulation)
- Computational overhead of maintaining per-synapse eligibility traces

---

## 4. KeRNL — Kernel-based Recurrent Neural Learning (Roth et al., 2019)

**Paper:** "Kernel RNN Learning (KeRNL)" — ICLR 2019

**Mechanism:** Approximates the temporal Jacobian (how hidden states affect future states) using a learned temporal kernel. Instead of backpropagating through time, the influence of past states on present loss is approximated by a parameterized kernel function k(t-s) that decays over time. The kernel parameters are learned alongside network weights.

ΔW ∝ Σ_s k(t-s) * local_gradient_s

**Biological metaphor:**
- Temporal kernel = synaptic trace with learned timescale
- Avoids unrolling = no storage of full history (online-compatible)
- Can be interpreted as a learned eligibility trace timescale

**What it solves:**
- Weight transport: Partially (still uses some form of error backprop locally)
- Non-locality: Partially (kernel is local in time but weight update still uses error)
- Non-causality: YES (kernel looks backward only)

**Architectures:** Vanilla RNN, with extensions possible to gated architectures.

**Tasks tested:** Copy task, adding problem, sequential MNIST, Penn Treebank language modeling.

**Limitations:**
- Kernel parameters add overhead and must be learned
- Performance gap with BPTT is significant on longer sequences
- Not fully local (error signal is still non-local)
- Weight transport not fully resolved
- Biological metaphor is weaker than e-prop or RFLO (kernel is mathematical convenience, not clear neural mechanism)

---

## 5. OSTL — Online Spatio-Temporal Learning (Bohnstingl et al., 2022)

**Paper:** "Online spatio-temporal learning in deep neural networks" — IEEE TNNLS / arXiv

**Mechanism:** Combines spatial and temporal credit assignment in an online fashion. Uses:
- Forward-mode differentiation for temporal gradients (avoiding BPTT)
- Random feedback for spatial gradients (avoiding weight transport)
- Online updates at each timestep

The key insight is that forward-mode differentiation (computing dh/dW forward in time) avoids non-causality, while random feedback avoids weight transport.

**Biological metaphor:**
- Forward-mode = running average of synaptic sensitivity (eligibility-like)
- Random feedback = distinct feedback pathways
- Online updates = immediate plasticity

**What it solves:**
- Weight transport: YES
- Non-locality: Partially (forward-mode requires per-neuron Jacobian tracking)
- Non-causality: YES

**Architectures:** Spiking networks, vanilla RNNs.

**Tasks tested:** DVS gesture recognition, speech commands, sequential MNIST.

**Limitations:**
- Forward-mode differentiation has O(n²) memory per layer (Jacobian storage)
- Computational cost can be high for large networks
- Not tested on gated architectures

---

## 6. ModProp (Liu et al., 2022)

**Paper:** "Biologically-plausible backpropagation through arbitrary timespans via local neuromodulators" — NeurIPS 2022

**Mechanism:** Introduces synapse-specific modulatory filters that convolve with local activity to approximate temporal credit assignment. Each synapse has its own temporal filter that determines how past eligibility traces are weighted when a modulatory signal arrives.

ΔW_ij ∝ (filter_ij * eligibility_ij) * modulator

The filters are learned, allowing each synapse to develop its own temporal credit assignment window.

**Biological metaphor:**
- Synapse-specific filters = diversity of synaptic time constants (observed in biology)
- Modulator = neuromodulatory signal (dopamine, norepinephrine)
- Local computation = all information available at the synapse
- Inspired by synaptic tagging and capture (STC) theory

**What it solves:**
- Weight transport: YES
- Non-locality: YES (fully local with modulatory signal)
- Non-causality: YES (filters operate causally)

**Architectures:** Vanilla RNN, continuous-time networks.

**Tasks tested:** Evidence accumulation, pattern generation, temporal XOR, tasks requiring long temporal credit.

**Limitations:**
- Synapse-specific filters add significant parameterization (more parameters to learn)
- Computational overhead of per-synapse convolutions
- Not tested on gated architectures (GRU/LSTM)
- Convergence can be slower than BPTT
- Filter learning dynamics not fully understood

---

## 7. Additional Recent Work (2023-2025)

### DECOLLE / Deep Continuous Local Learning (Kaiser et al., 2020)
- Local losses at each layer, online learning for spiking networks
- Avoids inter-layer backpropagation entirely
- Limited temporal credit assignment ability

### Predictive Coding Networks for Temporal Processing (Millidge et al., 2022-2023)
- Uses predictive coding framework where each layer predicts the next
- Errors are local (prediction errors)
- Temporal extensions exist but are less mature
- Promising biological metaphor (predictive coding is well-supported in neuroscience)

### RTRL Approximations (Tallec & Ollivier, 2017; Menick et al., 2021)
- Real-Time Recurrent Learning (RTRL) is online and causal but O(n⁴) cost
- Various approximations: Unbiased Online Recurrent Optimization (UORO), sparse RTRL
- SnAp (Menick et al., 2021): sparse n-step approximation to RTRL
- Biologically plausible variants exist but computational cost remains high

### Three-Factor Learning Rules (Gerstner et al., 2018 framework)
- General framework: ΔW = pre × post × modulator
- Encompasses e-prop, RFLO, and many others
- Theoretical unification rather than a specific new algorithm

---

## Comparative Table

| Method | Weight Transport | Locality | Causality/Online | Gated Arch. | Bio Metaphor Strength |
|--------|:---:|:---:|:---:|:---:|:---:|
| BPTT | No | No | No | Yes | None |
| Feedback Alignment | Yes | Partial | N/A (FF) | N/A | Moderate |
| RFLO | Yes | Yes | Yes | No | Strong |
| e-prop | Yes (v2,v3) | Yes | Yes | No | Strong |
| KeRNL | Partial | Partial | Yes | Possible | Weak |
| OSTL | Yes | Partial | Yes | No | Moderate |
| ModProp | Yes | Yes | Yes | No | Strong |
| RTRL approx | Yes (some) | No (O(n²)) | Yes | Possible | Weak |

---

## Identified Gaps and Opportunities

### Gap 1: Gated Architecture Support
**No existing biologically plausible method has been convincingly extended to GRU or LSTM.** All strong methods (RFLO, e-prop, ModProp) target vanilla RNNs or spiking networks. This is a clear opportunity. The challenge: gating introduces multiplicative interactions that complicate local gradient approximation.

### Gap 2: Bridging Supervised and RL
e-prop has some RL results, but most methods are demonstrated only in supervised settings. A method that naturally supports both (e.g., through a modifiable "reward prediction error" as the modulatory signal) would be novel and practically useful.

### Gap 3: Simplicity vs. Expressiveness Tradeoff
- RFLO is simple but limited (rank-1 approximation, short temporal credit)
- ModProp is expressive but complex (per-synapse learned filters)
- **There may be a sweet spot**: a method more expressive than RFLO but simpler than ModProp

### Gap 4: Computational Practicality
Most methods are tested on small-scale tasks. A method that scales to moderate-size problems (e.g., language modeling, moderate-length sequences) while remaining biologically plausible would be impactful.

### Gap 5: Unified Biological Narrative
Most methods pick one or two biological mechanisms. A method that coherently integrates multiple biological principles (e.g., eligibility traces + neuromodulation + dendritic computation + gating-as-neuromodulation) into a unified story could be more compelling for the neuroscience audience.

---

## Recommendations for Novel Algorithm Design

Based on the gaps identified:

1. **Target gated architectures** as the primary novelty — this is the clearest gap
2. **Start from RFLO** as the simplest working baseline, then ask: what minimal additions enable gating?
3. **Biological metaphor for gates**: interpret gate activations as local neuromodulatory signals that control plasticity at individual synapses (GRU reset gate ≈ eligibility trace decay; GRU update gate ≈ modulator strength)
4. **Keep it simpler than ModProp** — avoid per-synapse learned filters unless clearly necessary
5. **Test on standard benchmarks** (copy task, adding problem, sequential MNIST) to enable direct comparison with RFLO, e-prop, and KeRNL
6. **Design with RL extension in mind** — if the modulatory signal can be swapped from "supervised error" to "reward prediction error," the method naturally extends

---

## References

1. Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J. (2016). Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7, 13276.
2. Murray, J. M. (2019). Local online learning in recurrent networks with random feedback. eLife, 8, e43299.
3. Liu, Y. H., et al. (2022). Biologically-plausible backpropagation through arbitrary timespans via local neuromodulators. NeurIPS 2022.
4. Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. Nature Communications, 11, 3625.
5. Roth, C., Bhatt, I., & Bhatt, P. (2019). Kernel RNN Learning (KeRNL). ICLR 2019.
6. Bohnstingl, T., et al. (2022). Online spatio-temporal learning in deep neural networks. IEEE TNNLS.
7. Tallec, C., & Ollivier, Y. (2017). Unbiased Online Recurrent Optimization. arXiv:1702.05043.
8. Menick, J., et al. (2021). A practical sparse approximation for real time recurrent learning. arXiv:2006.07232.
9. Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D., & Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales. Frontiers in Neural Circuits, 12, 53.
