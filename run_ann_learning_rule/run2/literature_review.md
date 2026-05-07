# Literature Review: Biologically Plausible Learning Algorithms for Recurrent Neural Networks

## 1. Overview of the Landscape

The problem of training recurrent neural networks (RNNs) in a biologically plausible manner has generated a rich body of work spanning computational neuroscience and machine learning. The central challenge is temporal credit assignment: how can a synapse that was active at time $t$ be appropriately modified based on an error signal that arrives at time $t + \Delta t$, using only locally available information and causal (forward-in-time) computations?

This review systematically covers known approaches, the biological mechanisms they exploit, their limitations, and identifies gaps where genuinely novel algorithms might be developed.

---

## 2. Known Biologically Plausible RNN Learning Rules

### 2.1 Real-Time Recurrent Learning (RTRL) and Approximations

**RTRL (Williams & Zipser, 1989)** computes exact gradients online by maintaining a matrix of partial derivatives (the "influence matrix") that is updated at each timestep. While RTRL is online and causal, it has $O(n^4)$ computational complexity and requires non-local information (the full influence matrix), making it biologically implausible.

**Approximations to RTRL:**
- **SnAp (Menick et al., 2020)**: Sparse n-step Approximation that restricts the influence matrix to sparse interactions
- **UORO (Tallec & Ollivier, 2017)**: Unbiased Online Recurrent Optimization using rank-one approximations
- **KF-RTRL (Mujika et al., 2018)**: Kronecker-factored approximation reducing complexity to $O(n^3)$
- **Marschall et al. (2020)**: Unified framework showing many online rules are RTRL approximations with different sparsity patterns

**Biological plausibility**: RTRL approximations are generally more plausible than BPTT (online, causal) but still require non-local information for the influence matrix updates.

### 2.2 RFLO — Random Feedback Local Online Learning (Murray, 2019)

**Key idea**: Derives a local, online learning rule by (1) dropping non-local terms from the RTRL gradient, retaining only terms depending on pre/post-synaptic activity, and (2) replacing exact feedback weights with random fixed projections.

**Mechanism**: 
- Eligibility traces accumulate pre/post-synaptic correlations
- Random feedback alignment transmits error signals
- Weight updates: $\Delta W_{ij} \propto e_{ij} \cdot (\text{random projection of error})$
- Where $e_{ij}$ is the eligibility trace for synapse $i \to j$

**Properties**: Online, causal, local, no weight transport. Performance degrades on long temporal dependencies.

**Limitations**: The local approximation discards important temporal gradient information; struggles with tasks requiring credit assignment over many timesteps.

### 2.3 e-prop — Eligibility Propagation (Bellec et al., 2020)

**Key idea**: Decomposes the loss gradient into a product of (1) a learning signal broadcast from the output and (2) eligibility traces computed locally at each synapse.

**Mechanism**:
- Each synapse maintains an eligibility trace $e_{ij}(t)$ that is a filtered version of how past pre-synaptic activity affected current post-synaptic activity
- A top-down learning signal $L_j(t)$ is broadcast to neuron $j$
- Weight update: $\Delta W_{ij} \propto \sum_t L_j(t) \cdot e_{ij}(t)$

**Three variants**:
- e-prop (symmetric): uses exact transpose weights (not biologically plausible)
- e-prop (random): uses random feedback (biologically plausible, like feedback alignment)
- e-prop (adaptive): uses learned feedback weights

**Properties**: Online, local synapse computation, demonstrated on spiking networks. Proven to approximate BPTT under certain conditions.

**Limitations**: Still requires a globally broadcast error signal (though delivered via random projection). Performance gap with BPTT grows for longer temporal dependencies.

### 2.4 ModProp (Liu et al., 2022)

**Key idea**: Uses local neuromodulatory signals (analogous to neuropeptide diffusion) to propagate credit through time via synapse-type-specific filters applied to eligibility traces.

**Mechanism**:
- Modulatory signals diffuse extra-synaptically and convolve with eligibility traces
- Filter taps are causal and time-invariant
- Credit propagates through arbitrary timespans without truncation
- Different synapse types can have different filter characteristics

**Biological basis**: Neuropeptide diffusion, volume transmission, neuromodulatory signaling

**Properties**: Local, causal, handles longer temporal dependencies than e-prop/RFLO. Key advance is propagating credit through time without BPTT-like backward passes.

**Limitations**: Requires specifying the filter structure; assumes particular neuromodulatory connectivity patterns.

### 2.5 Feedback Alignment and Variants (Lillicrap et al., 2016)

**Key idea**: Replace the transpose of the forward weights in backpropagation with fixed random matrices. Surprisingly, learning still works because forward weights align with the random feedback over training.

**Variants for recurrent networks**:
- Direct Feedback Alignment (Nøkland, 2016): error projected directly to each layer
- Broadcast Alignment: single random projection shared across layers
- Sign concordance variants

**Properties**: Solves weight transport problem. Originally for feedforward; applied to recurrent settings in combination with other rules (e.g., RFLO, e-prop).

**Limitations**: Alone, does not solve temporal credit assignment. Must be combined with eligibility traces or online learning rules for RNNs.

### 2.6 Equilibrium Propagation (Scellier & Bengio, 2017; Ernoult et al., 2019)

**Key idea**: In a convergent RNN (one that settles to a fixed point), learning can be achieved by comparing the free-phase equilibrium with a weakly-clamped equilibrium where the output is nudged toward the target.

**Mechanism**:
- Two-phase learning: free phase (settle to equilibrium) and nudged phase (small perturbation toward target)
- Weight update proportional to difference in correlations between phases
- Proven to compute exact gradients in the limit of small nudging

**Properties**: Contrastive Hebbian-like, local computation, biologically motivated (resembles experimental protocols).

**Limitations**: Requires convergence to equilibrium (not suitable for online temporal processing); restricted to energy-based/convergent architectures; slow due to equilibration requirement.

### 2.7 Predictive Coding for Temporal Learning

**Key idea**: Hierarchical generative models where each layer predicts the activity of the layer below; prediction errors drive learning.

**Papers**:
- Rao & Ballard (1999): Original predictive coding
- Millidge et al. (2022): Shows predictive coding approximates backprop
- Temporal predictive coding (various): Extend to sequences by predicting future states

**Mechanism**:
- Each neuron maintains a prediction of its input
- Mismatch between prediction and actual input drives local plasticity
- Temporal extension: predictions span time, so temporal prediction errors drive temporal credit assignment

**Properties**: Local error computation, biologically grounded in hierarchical cortical processing.

**Limitations**: Standard predictive coding requires convergence (iterative inference); temporal extensions are less well-developed for arbitrary RNN tasks; unclear how to scale to complex temporal problems.

### 2.8 Dendritic Error Models (Sacramento et al., 2018; Payeur et al., 2021)

**Sacramento et al. (2018)**: "Dendritic cortical microcircuits approximate the backpropagation algorithm"
- Separate apical and basal dendritic compartments
- Apical dendrites receive top-down predictions; mismatches create local error signals
- Demonstrated that this architecture mathematically approximates backprop

**Payeur et al. (2021)**: "Burst-dependent synaptic plasticity"
- Multiplexing of signals through burst/single-spike coding
- Bursts carry error information while single spikes carry feedforward signals
- Provides a temporal mechanism for credit assignment without explicit backward passes

**Properties**: Biologically grounded in dendritic physiology; local error computation.

**Limitations**: Primarily demonstrated for feedforward networks; extension to temporal processing in RNNs is not straightforward.

### 2.9 Forward-Forward Algorithm (Hinton, 2022)

**Key idea**: Replace forward-backward passes with two forward passes — one with real data (positive) and one with negative/generated data. Each layer learns to have high "goodness" for positive data and low for negative.

**Properties**: No backward pass, potentially local.

**Limitations**: Designed for feedforward networks; no clear temporal extension for RNNs; performance significantly below backprop on standard benchmarks.

### 2.10 FORCE Learning (Sussillo & Abbott, 2009)

**Key idea**: Train only the readout weights of a chaotic reservoir using recursive least squares, shaping the dynamics by feeding back the trained output.

**Properties**: Extremely fast learning; biologically motivated by the reservoir computing paradigm.

**Limitations**: Not a true learning rule for recurrent weights; the recurrent connections are not trained.

### 2.11 Miconi (2017) — Biologically Plausible RNN Learning via Reward-Modulated Hebbian Plasticity

**Key idea**: Combines Hebbian traces with a reward-modulated global signal (analogous to dopamine) to train RNNs on cognitive tasks.

**Properties**: Local Hebbian computation, global reward signal; demonstrated on working memory and decision tasks.

**Limitations**: Relies on a global reward signal; limited to RL settings; struggles with complex temporal credit assignment.

### 2.12 Asabuki & Clopath (2024) — Predictive Alignment in RNNs

Recent work proposing a predictive alignment rule that "tames the chaos gently" in RNNs, combining predictive coding with recurrent dynamics. Uses local prediction errors to stabilize and train recurrent networks.

### 2.13 Soo, Goudar & Wang (2023) — Training Bioplausible RNNs on Long Dependencies

Addresses the specific challenge of training biologically plausible RNNs on tasks with long-term dependencies, bridging the gap between short-timescale plasticity and long-timescale task requirements.

---

## 3. Biological Mechanisms Exploited

| Mechanism | Used By | How It's Used |
|-----------|---------|---------------|
| **Eligibility traces** | e-prop, RFLO, ModProp | Mark recently active synapses for later modification |
| **Neuromodulation** | ModProp, reward-modulated Hebbian | Gate plasticity with diffuse signals |
| **Random feedback** | RFLO, e-prop (random), FA variants | Solve weight transport without symmetric weights |
| **Hebbian/anti-Hebbian** | Miconi (2017), contrastive Hebbian | Local correlation-based updates |
| **Dendritic computation** | Sacramento et al. (2018), Payeur et al. (2021) | Compartmentalized error computation |
| **Burst coding** | Payeur et al. (2021) | Multiplex error and activation signals |
| **Predictive coding** | Temporal predictive coding, Asabuki & Clopath (2024) | Local prediction errors drive plasticity |
| **Equilibrium dynamics** | Equilibrium propagation | Contrastive phases encode gradient |
| **Synaptic tagging** | Conceptually referenced but rarely formalized in RNN rules | Multi-timescale consolidation |
| **Oscillatory dynamics** | Rarely exploited for learning rules | Phase coding, cross-frequency coupling |

---

## 4. Gaps and Limitations in Current Approaches

### 4.1 Common Limitations

1. **Performance-plausibility tradeoff**: More biologically plausible rules (RFLO) perform significantly worse than less plausible ones (e-prop symmetric) which in turn lag behind BPTT
2. **Long temporal dependencies**: All local rules struggle when credit must be assigned over many timesteps (>50-100)
3. **Scalability**: Most rules demonstrated only on simple tasks or small networks
4. **Gated architectures**: Few rules have been extended to LSTMs/GRUs (which themselves have biological interpretations via gating)
5. **Unsupervised/self-supervised**: Most rules assume supervised error signals

### 4.2 Fundamental Tradeoffs

- **Locality vs. credit depth**: More local rules can't propagate credit as far back
- **Online vs. accuracy**: Online rules use stale/approximate gradient information
- **Biological fidelity vs. task performance**: Stricter biological constraints generally reduce learning capacity

---

## 5. Assessment of Proposed Novel Directions

### 5.1 Oscillatory Phase for Temporal Credit Assignment

**Has it been done?** 

*Partially explored but NOT as a complete RNN learning rule:*
- Oscillatory dynamics are well-studied in neuroscience (theta-gamma coupling, phase precession)
- Phase coding has been used for information representation (O'Keefe & Recce, 1993; Lisman & Jensen, 2013)
- Some work on oscillatory gating of plasticity (Fell & Axmacher, 2011)
- Phasic modulation of STDP windows (Lengyel et al., 2005)

**Gap identified**: No existing work uses oscillatory phase relationships as the *primary mechanism* for temporal credit assignment in RNNs. Phase has been used for representation and for gating plasticity windows, but not as a structured method to route credit assignment signals through time in a recurrent network.

**Assessment**: This direction appears **genuinely novel** as a complete learning rule framework. The key insight would be using phase relationships to create "temporal routing" of credit — different phase offsets could selectively connect different temporal delays in the eligibility trace.

### 5.2 Multi-Timescale Dendritic Compartments for Temporal Credit

**Has it been done?**

*Partially:*
- Sacramento et al. (2018) uses dendritic compartments but for feedforward (spatial) credit assignment
- Multi-timescale processing in dendrites is biologically established (Poirazi et al., 2003)
- Some models use dendritic compartments with different time constants (Urbanczik & Senn, 2014)
- Guerguiev et al. (2017): Multi-compartment neurons for deep learning

**Gap identified**: Using different dendritic compartments with different temporal integration timescales *specifically for temporal credit assignment in RNNs* appears unexplored. The idea that a single neuron could use its dendritic tree to simultaneously track credit at multiple timescales is not well-developed.

**Assessment**: **Partially novel**. The components exist separately (dendritic compartments for error, multi-timescale integration) but their combination for temporal credit in RNNs has not been formalized. A key challenge: how do compartments with different timescales cooperate to assign credit at the correct temporal offset?

### 5.3 Synaptic Resource Competition for Learning

**Has it been done?**

*Related but distinct:*
- Synaptic tagging and capture (Frey & Morris, 1997) — biological mechanism exists
- Cascade models of synaptic plasticity (Fusi et al., 2005) — multi-state synapses
- Metaplasticity (Abraham, 2008) — activity-dependent changes in plasticity rules
- Resource-rational theories of cognition (Lieder & Griffiths, 2020) — different framework

**Gap identified**: Framing learning *itself* as resource competition — where synapses compete for a limited plasticity resource rather than following a gradient — has **not** been developed as a complete RNN learning rule. This is a fundamentally different computational paradigm from gradient-based approaches.

**Assessment**: **Novel and surprising**. This would be a departure from the standard framework where learning = gradient descent. The key question: can competitive resource dynamics implement something functionally equivalent to temporal credit assignment? This requires theoretical work to show under what conditions resource competition converges to useful solutions.

---

## 6. Under-Explored Directions for Novel Algorithms

Based on this survey, the following directions appear genuinely under-explored:

1. **Oscillatory phase as a credit routing mechanism** — Using neural oscillations not just for gating but as a structured temporal address system for credit assignment

2. **Synaptic competition as implicit optimization** — Moving away from gradient computation entirely; learning through competitive dynamics where behaviorally relevant synapses outcompete others for limited resources

3. **Topological/structural learning** — Instead of modifying weights, growing/pruning connections based on local activity statistics (structural plasticity as learning)

4. **Cross-frequency coupling for multi-timescale credit** — Nested oscillations (theta containing gamma cycles) as a natural hierarchy for assigning credit at multiple temporal scales simultaneously

5. **Retrograde signaling for temporal credit** — Using biological retrograde messengers (endocannabinoids, nitric oxide) that travel backwards from post- to pre-synaptic neurons as a temporal credit signal

6. **Energy-based temporal models** — Extending equilibrium propagation to non-equilibrium temporal settings using dissipative dynamics

7. **Astrocytic integration for slow credit assignment** — Astrocytes integrate neural activity over seconds-to-minutes timescales and modulate synaptic transmission; they could implement slow temporal credit assignment that complements fast Hebbian learning

---

## 7. Summary Table: Comparison of Approaches

| Method | Year | Weight Transport | Local | Online | Temporal Credit | Performance vs BPTT |
|--------|------|-----------------|-------|--------|-----------------|-------------------|
| BPTT | 1990 | ✗ | ✗ | ✗ | Exact | Reference |
| RTRL | 1989 | ✓ | ✗ | ✓ | Exact | Equal |
| RFLO | 2019 | ✓ | ✓ | ✓ | Weak | ~60-80% |
| e-prop | 2020 | ✓* | ✓ | ✓ | Moderate | ~80-95% |
| ModProp | 2022 | ✓ | ✓ | ✓ | Strong | ~85-95% |
| Equil. Prop. | 2017 | ✓ | ✓ | ✗ | Static only | ~90-95% (static) |
| Feedback Align. | 2016 | ✓ | ✗ | ✗ | Via combo | ~85-95% |
| Predictive Coding | Various | ✓ | ✓ | ✗† | Emerging | ~80-90% |

*e-prop with random feedback; ✓* = partially solved
†Requires iterative inference convergence

---

## 8. Key References

1. Bellec, G., et al. (2020). "A solution to the learning dilemma for recurrent networks of spiking neurons." *Nature Communications*, 11, 3625.
2. Murray, J.M. (2019). "Local online learning in recurrent networks with random feedback." *eLife*, 8, e43299.
3. Liu, Y.H., et al. (2022). "Biologically-plausible backpropagation through arbitrary timespans via local neuromodulators." *NeurIPS 2022*.
4. Lillicrap, T.P., et al. (2016). "Random synaptic feedback weights support error backpropagation for deep learning." *Nature Communications*, 7, 13276.
5. Sacramento, J., et al. (2018). "Dendritic cortical microcircuits approximate the backpropagation algorithm." *NeurIPS 2018*.
6. Payeur, A., et al. (2021). "Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits." *Nature Neuroscience*, 24, 1010-1019.
7. Scellier, B. & Bengio, Y. (2017). "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation." *Frontiers in Computational Neuroscience*, 11, 24.
8. Williams, R.J. & Zipser, D. (1989). "A learning algorithm for continually running fully recurrent neural networks." *Neural Computation*, 1(2), 270-280.
9. Marschall, O., Cho, K., & Savin, C. (2020). "A unified framework of online learning algorithms for training recurrent neural networks." *JMLR*, 21(135), 1-34.
10. Sussillo, D. & Abbott, L.F. (2009). "Generating coherent patterns of activity from chaotic neural networks." *Neuron*, 63(4), 544-557.
11. Miconi, T. (2017). "Biologically plausible learning in recurrent neural networks reproduces neural dynamics observed during cognitive tasks." *eLife*, 6, e20899.
12. Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." *arXiv:2212.13345*.
13. Tallec, C. & Ollivier, Y. (2017). "Unbiased Online Recurrent Optimization." *arXiv:1702.05043*.
14. Ernoult, M., et al. (2019). "Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input." *NeurIPS 2019*.
15. Asabuki, T. & Clopath, C. (2024). "Taming the chaos gently: a predictive alignment learning rule in recurrent neural networks." *bioRxiv*.
16. Soo, W., Goudar, V., & Wang, X.J. (2023). "Training biologically plausible recurrent neural networks on cognitive tasks with long-term dependencies." *bioRxiv*.

---

## 9. Conclusions and Recommendations

### Most Promising Novel Direction: Oscillatory Phase-Based Credit Routing

The direction with the best combination of novelty, biological grounding, and potential surprise is **using oscillatory phase relationships as a temporal credit routing mechanism**. This is because:

1. **Genuinely novel**: No existing RNN learning rule uses phase as the primary credit assignment mechanism
2. **Biologically grounded**: Theta-gamma coupling, phase precession, and cross-frequency coupling are well-established phenomena
3. **Surprising**: The connection between oscillatory phase coding and temporal credit assignment is non-obvious
4. **Mechanistically distinct**: Unlike eligibility traces (exponential decay) or neuromodulation (scalar gating), phase provides a *structured* temporal reference frame

### Second Most Promising: Synaptic Competition

The resource competition framing is highly novel and conceptually surprising, but faces a higher theoretical bar — it requires demonstrating that non-gradient dynamics can solve temporal credit assignment. This could be transformative if successful but carries more risk.

### Recommendation

Begin with formalizing the oscillatory phase credit routing idea, while keeping synaptic competition as a backup direction. The first step should be to work out the mathematical framework: how do phase relationships between neurons encode temporal credit assignment information, and what update rule follows from this?
