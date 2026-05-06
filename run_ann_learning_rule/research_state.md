# Research State: Biologically Plausible Learning Rules for RNNs

## 1. Research Question & Scope

**Core question**: Can we design a learning rule for recurrent neural networks that (a) avoids the biological implausibilities of BPTT (weight transport, non-locality, non-causality), (b) has a clear mapping to known biological mechanisms, and (c) remains computationally effective on standard ML benchmarks?

**Scope**:
- Architecture: vanilla RNN → GRU → LSTM (progressive complexity)
- Learning paradigms: supervised → reinforcement → unsupervised
- Biological plausibility axes: locality, causality, no weight transport, biologically interpretable signals
- Benchmark difficulty: toy tasks → sequential MNIST → more complex temporal tasks

**Compute environment**:
- Python/PyTorch experiments: use conda env at `/opt/homebrew/Caskroom/miniforge/base/envs/uwzihan` (PyTorch 2.2.1)
- Run via: `/opt/homebrew/Caskroom/miniforge/base/envs/uwzihan/bin/python script.py`

## 2. Operational Definitions

- **Locality**: A weight update Δw_ij depends only on quantities available at synapse ij (pre/post activities, local modulatory signals), not on activities or weights of distant neurons
- **Causality**: The update at time t depends only on information available at or before time t (no backward pass through time)
- **Weight transport**: The requirement that the feedback pathway uses the exact transpose of the forward weight matrix
- **Eligibility trace**: A local synaptic variable that records recent co-activity of pre- and post-synaptic neurons, gating plasticity when a modulatory signal arrives
- **Neuromodulatory signal**: A diffuse, extra-synaptic signal (e.g., neuropeptides, dopamine) that can broadcast information to populations of neurons without precise synaptic targeting

## 3. Related Work

### 3.1 ModProp (Liu et al., NeurIPS 2022)
- Proposes that extra-synaptic diffusion of neuromodulators (e.g., neuropeptides) can propagate credit through arbitrary timespans
- The key idea: modulatory signals convolve eligibility traces via causal, time-invariant, synapse-type-specific filter taps
- Goes beyond temporal truncation (e-prop, RTRL truncations) by maintaining credit over longer horizons
- Demonstrates advantage over existing bio-plausible temporal credit assignment rules on benchmark tasks
- Provides a low-complexity, causal alternative to BPTT

### 3.2 RFLO (Murray, eLife 2019)
- Derives a local, online learning rule for RNNs using random feedback projections
- Weight updates depend only on local pre/post-synaptic activities + a random feedback projection of output error
- Addresses both locality and weight transport simultaneously
- Proposes an augmented circuit architecture for concatenating short patterns into longer sequences
- Limitation: may struggle with very long temporal dependencies without the augmented architecture

### 3.3 Feedback Alignment (Lillicrap et al., Nature Comms 2016)
- Demonstrates that random, fixed feedback weights can replace symmetric backpropagation
- The forward weights gradually align with the random feedback weights during learning
- Resolves the weight transport problem for feedforward networks
- Foundation for subsequent work extending feedback alignment to recurrent settings

### 3.4 SuperSpike (Zenke & Ganguli, 2018)
- Supervised learning in multilayer spiking neural networks via surrogate gradients
- Key contribution: a three-factor learning rule (pre × post × error) that works in temporal/spiking systems
- Uses surrogate gradients to handle the non-differentiability of spikes
- Demonstrates that local eligibility traces combined with a global error signal suffice for multi-layer credit assignment in spiking networks
- Relevant to our work: provides a template for three-factor rules in temporal networks and shows eligibility traces can bridge the locality gap

### 3.5 Predictive Coding as Approximate Backprop (Whittington & Bogacz, 2017)
- Shows that predictive coding networks with local Hebbian plasticity approximate the backpropagation algorithm
- Each layer maintains a prediction of the layer below; errors propagate locally through prediction mismatches
- Weight updates are purely local (Hebbian) once the network reaches equilibrium
- Canonical paper linking the predictive coding framework to gradient-based learning
- Relevant to our work: suggests predictive coding could be the biological substrate for error propagation in recurrent/temporal settings — Open Question #7

### 3.6 Dendritic Cortical Microcircuits (Sacramento et al., 2018)
- Proposes that cortical pyramidal neurons with segregated dendritic compartments can implement backpropagation
- Apical dendrites carry top-down error signals; basal dendrites carry feedforward activations
- Credit assignment emerges from the interaction between compartments without weight transport
- Specific interneuron types (SST, VIP, PV) play defined roles in routing error vs. activation signals
- Relevant to our work: provides a concrete biological circuit motif for spatial credit assignment that could complement temporal mechanisms (eligibility traces, neuromodulation)

### 3.7 Equilibrium Propagation (Scellier & Bengio, 2017)
- Energy-based alternative to backpropagation where error information propagates through network dynamics
- After a small perturbation (nudge) at the output, the network relaxes to a new equilibrium; the difference between free and nudged equilibria gives the gradient
- Fully local: weight updates depend only on pre/post activities in the two phases
- No separate backward pass or error pathway required — the same connections carry both inference and learning signals
- Relevant to our work: offers a physics-inspired framework that avoids weight transport entirely; however, extension to temporal/recurrent processing (beyond equilibrium systems) remains an open challenge

### 3.8 Temporal Predictive Coding (Rao & Ballard, 1999; Millidge et al., 2022)
- Rao & Ballard (1999) introduced predictive coding in visual cortex; Millidge et al. (2022, "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs") extends the framework to arbitrary computation graphs including recurrent/temporal ones
- In temporal predictive coding, each neuron maintains a prediction of its own next-state; the prediction error drives local plasticity
- Learning reduces to minimizing a sequence of local prediction errors propagated in time — no explicit backward pass through time is needed
- Weight updates are Hebbian: Δw ∝ (prediction error) × (presynaptic activity), entirely local
- The temporal hierarchy naturally handles multi-timescale dependencies: higher layers predict slower dynamics, lower layers predict faster dynamics
- Key limitation: convergence requires iterative inference (settling dynamics) at each timestep, which is computationally expensive and biologically debated
- Relevant to our work: provides an alternative spatial credit mechanism to random feedback — prediction errors propagated through a hierarchy could replace feedback alignment with a more biologically grounded signal, and the temporal hierarchy offers a natural way to extend the credit horizon beyond single eligibility trace timescales

### 3.9 e-prop (Bellec et al., Nature Communications 2020)
- Three-factor learning rule for spiking RNNs: dw_ij/dt = e_ij(t) * L_i(t)
- Eligibility trace e_ij(t) is local, online, O(1) per synapse; learning signal L_i(t) is projected top-down
- Two variants: symmetric (uses weight transpose — not bio-plausible) and random (feedback alignment — bio-plausible)
- Benchmarks: TIMIT ~91%, store-and-recall ~89%; symmetric closely matches BPTT, random shows some degradation
- Limitation: temporal credit horizon limited by eligibility trace decay (tau_e ~ hundreds of ms); struggles with very long dependencies
- Complexity: O(N) total per timestep — only bio-plausible method at this scale
- Directly validates that eligibility traces + feedback alignment work for recurrent spiking networks

### 3.10 RTRL Approximations (UORO, SnAp, KF-RTRL)
- Full RTRL: O(n^4) time, O(n^3) memory — exact but impractical
- **UORO** (Tallec & Ollivier, 2017): rank-one stochastic approximation, O(n^2) time, O(n) memory, unbiased but high variance, NOT biologically local
- **SnAp** (Menick et al., 2020): sparse n-step approximation, O(n^2) for SnAp-1, deterministic, outperforms UORO substantially, NOT biologically local
- **KF-RTRL** (Mujika et al., NeurIPS 2018): Kronecker factored, O(n^2), lower variance than UORO, "almost matches TBPTT" on PTB
- Key insight: ALL RTRL approximations achieve O(n^2) but NONE are biologically local — they require global information flow
- Our approach (if H2 works) would be unique: O(N) AND local AND extended temporal credit

### 3.11 Three-Factor Hebbian Rules (General Framework)
- Framework: dw_ij/dt = e_ij * M(t) where e_ij = f(pre) * g(post), M = neuromodulator
- Key papers: Izhikevich (2007, ~2000 cit.), Gerstner et al. (2018, ~473 cit.), Kusmierz et al. (2017)
- Biological evidence: striatum (dopamine, 1s window), cortex (NE/serotonin, 3-10s), hippocampus (dopamine, up to 1 min)
- Proven achievable: reward-based learning, surprise-driven plasticity, behavioral timescale learning (tau_e 200ms-2s)
- The third factor (modulator) can encode: reward prediction error, supervised error, attention, novelty

### 3.12 Combining Eligibility Traces with Neuromodulatory Temporal Convolutions
- **No published paper explicitly unifies e-prop + ModProp** — this is a genuine research gap
- Closest: Barretto-Bittar et al. (2026) extends e-prop with diffusion-based neuromodulatory credit; improved learning on three benchmarks
- ModProp itself can be viewed as applying learned causal filter taps to eligibility traces
- The formal comparison/synthesis remains open → directly supports novelty of H2

### 3.13 Other Relevant Work (to be surveyed)
- Dendritic computation theories beyond Sacramento (e.g., Guerguiev et al., 2017)
- Marschall, Cho & Savin (JMLR 2020): unified framework for online RNN learning

## 4. Hypotheses

**H1** (Confidence: 60%): A combination of eligibility traces (for temporal credit) and random feedback projections (for spatial credit) can yield a fully local, causal learning rule for vanilla RNNs that performs within 80% of BPTT accuracy on sequential MNIST.

**H2** (Confidence: 45%): The neuromodulatory convolution idea from ModProp can be unified with random feedback (from RFLO/feedback alignment) into a single framework that handles both temporal and spatial credit assignment without weight transport.

**H3** (Confidence: 50%): Such a unified rule can be naturally extended to gated architectures (GRU/LSTM) by interpreting gating operations as dendritic computations or local neuromodulatory gating, without requiring fundamentally different mechanisms.

**H4** (Confidence: 35%): The same rule, with minimal modification, can operate in reinforcement learning settings by replacing the supervised error signal with a reward prediction error (analogous to dopamine signaling).

**H5** (Confidence: 40%): The computational overhead of the bio-plausible rule is O(N^2) per timestep (similar to RFLO), making it practical for moderate-sized networks.

**H6** (Confidence: 55%): The effective temporal credit horizon of the combined rule is governed by the eligibility trace time constant τ, predicting that (a) networks trained on tasks requiring T-step dependencies will develop eligibility traces with τ ∝ T, and (b) performance will degrade sharply for dependencies beyond ~2τ — mirroring the biological observation that neuromodulator diffusion timescales (tens of ms to seconds) set a natural limit on credit assignment range, and suggesting that organisms requiring longer credit horizons must recruit additional mechanisms (e.g., hippocampal replay, working memory buffers).

**H7** (Confidence: 35%): Replacing random feedback projections with temporal predictive coding — where spatial credit is carried by local prediction errors rather than fixed random weights — will (a) improve learning stability and final accuracy (by 5–15% on sMNIST) compared to feedback alignment, and (b) yield representations that spontaneously organize into a temporal hierarchy (higher layers tracking slower features, lower layers tracking faster features), matching the hierarchical timescale organization observed in primate cortex (Murray et al., J. Neurosci 2014). This would provide a falsifiable neuroscience prediction: if the model is correct, ablating top-down predictive connections in cortex should selectively impair learning of long-range but not short-range temporal dependencies.

## 5. Experimental Designs

### Phase 1: Supervised Learning (vanilla RNN)
- **Tasks**: Copy task, adding problem, sequential MNIST, sequential CIFAR-10
- **Baselines**: BPTT, truncated BPTT, RTRL, e-prop, ModProp, RFLO
- **Metrics**: accuracy/loss, convergence speed, memory footprint, wall-clock time
- **Ablations**: effect of feedback alignment vs. symmetric feedback; effect of eligibility trace time constants; effect of modulatory filter shape

### Phase 2: Extension to Gated Architectures
- **Tasks**: same as Phase 1 + language modeling (Penn Treebank char-level)
- **Architectures**: GRU, LSTM with bio-plausible rule vs. BPTT
- **Key question**: does gating help the bio-plausible rule as much as it helps BPTT?

### Phase 3: Beyond Supervised Learning
- **RL tasks**: CartPole, Acrobot, simple Atari games with recurrent policies
- **Unsupervised**: next-step prediction, sequence completion
- **Key question**: can the same local learning rule operate with reward-modulated or prediction-error-driven signals?

## 6. Results Summary

*No computational experiments conducted yet.*

## 7. Open Questions & Confusions

1. **How to handle gating in a bio-plausible way?** Gates in GRU/LSTM involve multiplicative interactions — what is the biological analogue? Dendritic gating? Shunting inhibition? Local neuromodulators controlling ion channel conductances?

2. **Reconciling ModProp and RFLO**: ModProp focuses on temporal credit (replacing BPTT's backward pass through time) while RFLO focuses on spatial credit (replacing weight transport). Can these be cleanly composed, or do they interfere?

3. **Scalability**: RTRL is O(N^4) which is impractical. RFLO is O(N^2). Can we maintain O(N^2) or better while incorporating ModProp-like temporal propagation?

4. **Biological metaphor coherence**: We need a single, coherent biological story — not a patchwork of tricks. What neural circuit motif implements the full rule? (e.g., a cortical microcircuit with specific interneuron types?)

5. **Benchmarking fairness**: How to fairly compare bio-plausible rules against BPTT given that BPTT has access to exact gradients? Should we compare at equal parameter count, equal compute budget, or equal architectural complexity?

6. **Stability**: Bio-plausible rules often suffer from instability in long sequences. What stabilization mechanisms (analogous to gradient clipping in BPTT) are biologically plausible?

7. **Connection to predictive coding**: Several recent papers frame biological learning as predictive coding in hierarchical networks. Is there a natural extension of this to the temporal/recurrent domain that subsumes our approach?
