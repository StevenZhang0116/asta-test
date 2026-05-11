---
title: "Step 2 Design: STC-Credit — Synaptic Tagging-and-Capture as a Temporal Credit-Assignment Rule for RNNs"
bibliography: references.bib
---

# 1. Summary

We propose **STC-Credit**: a biologically plausible online learning rule for recurrent neural networks in which temporal credit assignment is performed by a two-timescale *synaptic tagging and capture* (STC) process. A fast, decaying **eligibility tag** integrates pre × post-synaptic co-activity at each synapse; a slow, threshold-crossing **capture event**, driven by a surprise / prediction-error modulator broadcast through fixed random feedback weights, commits the accumulated tag into a weight change. Between capture events, tags decay but do not commit.

STC-Credit sits in the three-factor rule family ([@gerstner2018; @fremaux2016]) but differs from all existing three-factor RNN rules (RFLO, e-prop, ModProp, Miconi node-perturbation) in one specific way: *the modulator acts as a threshold gate, not a continuous multiplier*. This maps onto the known biology of protein-synthesis-dependent consolidation ([@clopath2008; @ziegler2015; @benna2016]) that none of those rules explicitly implement as a credit-assignment mechanism.

# 2. The Rule (STC-Credit)

## 2.1 Architecture

A vanilla RNN with hidden state $h_t \in \mathbb{R}^N$, input $x_t \in \mathbb{R}^{n_x}$, readout $\hat y_t \in \mathbb{R}^{n_y}$:

$$
u_t = W h_{t-1} + U x_t + b, \qquad
h_t = \phi(u_t), \qquad
\hat y_t = V h_t.
$$

Trainable parameters: $\theta = \{W, U, V, b\}$. Output target $y_t$; instantaneous error $e_t = \hat y_t - y_t$ (supervised, for Step 3/4 comparability to BPTT).

Fixed random **feedback matrix** $B \in \mathbb{R}^{N \times n_y}$ (never updated; Gaussian entries). It maps readout error back into a per-hidden-unit teaching signal:

$$
\ell_t := B \, e_t \quad \in \mathbb{R}^N .
$$

This is the same feedback pathway used by RFLO ([@murray2019]) and avoids weight transport.

## 2.2 Fast eligibility tag (per synapse)

For each recurrent synapse $W_{ij}$, maintain a scalar tag $\tau^W_{ij}(t)$ that integrates the RFLO-style pre × post-synaptic coincidence:

$$
\tau^W_{ij}(t) = (1 - \alpha_\tau)\,\tau^W_{ij}(t-1) \;+\; \alpha_\tau \, \phi'(u_i(t)) \, h_j(t-1).
$$

Similarly for input synapses $U_{ij}$ and readout $V_{ij}$:

$$
\tau^U_{ij}(t) = (1 - \alpha_\tau)\,\tau^U_{ij}(t-1) \;+\; \alpha_\tau \, \phi'(u_i(t)) \, x_j(t),
$$
$$
\tau^V_{ij}(t) = (1 - \alpha_\tau)\,\tau^V_{ij}(t-1) \;+\; \alpha_\tau \, h_j(t).
$$

$\alpha_\tau \in (0, 1]$ sets the tag decay. We choose $\alpha_\tau$ so the effective tag window is $\sim 1/\alpha_\tau$ timesteps, matching the range of temporal dependencies in the task (e.g. $\alpha_\tau \approx 0.1$ for sequences of length $\sim 10$). $\phi'(u_i)$ is the post-synaptic gain — the standard e-prop / RFLO post-synaptic factor.

## 2.3 Surprise signal and capture gate

Rather than multiply the tag by $\ell_t$ on every step, STC-Credit computes a scalar **surprise** $s_t$ per post-synaptic unit $i$:

$$
s_i(t) = \tfrac{|\ell_i(t)|^2}{\text{EMA}_i(|\ell|^2) + \varepsilon},
$$

i.e. the instantaneous squared feedback-driven error, normalized by its exponentially-moving average. (This is Welford-style z-scoring of the feedback magnitude, giving scale invariance.) An alternative simpler choice is $s_i(t) = |\ell_i(t)|$. The capture gate at neuron $i$:

$$
g_i(t) = \mathbb{1}\{s_i(t) > \theta\}.
$$

$\theta$ is a **surprise threshold**: only when the feedback signal is anomalously large does capture occur. We initialize $\theta$ so the average firing rate of $g$ is $\sim 5\text{–}10\%$ of timesteps (tunable; a high-level hyperparameter is "average capture rate").

## 2.4 Weight update (capture step)

When $g_i(t) = 1$, the accumulated tag at neuron $i$'s incoming synapses is committed:

$$
\Delta W_{ij}(t) = - \eta \, g_i(t) \, \ell_i(t) \, \tau^W_{ij}(t), \qquad
\Delta U_{ij}(t) = - \eta \, g_i(t) \, \ell_i(t) \, \tau^U_{ij}(t).
$$

The readout $V$ is trained with the standard local delta rule (no tagging needed since $\hat y$ is static in $h$):

$$
\Delta V_{ij}(t) = - \eta_V \, e_i(t) \, h_j(t).
$$

No tag reset after capture is needed in the basic rule: tags continue to decay naturally. An optional variant resets $\tau_{ij} \leftarrow 0$ after capture (analogous to PRP-driven tag reset); this becomes an ablation in E4.

## 2.5 Pseudocode

```
Initialize W, U, V, b  (small Gaussian)
Initialize B (fixed random Gaussian, never updated)
Initialize tags τ^W, τ^U, τ^V = 0
Initialize running mean m_i = 0, variance v_i = 1

for each timestep t in sequence:
    u = W h + U x + b
    h = φ(u)
    ŷ = V h

    # Teaching signal
    e = ŷ - y_t                  # readout error
    ℓ = B e                      # per-neuron feedback

    # Update tags
    τ^W ← (1-α_τ) τ^W + α_τ · φ'(u) ⊗ h_prev
    τ^U ← (1-α_τ) τ^U + α_τ · φ'(u) ⊗ x
    τ^V ← (1-α_τ) τ^V + α_τ · h

    # Surprise and capture
    s_i = |ℓ_i|² / (m_i + ε)
    g_i = 1 if s_i > θ else 0
    update m_i via EMA on |ℓ_i|²

    # Commit (capture)
    ΔW_{ij} = -η · g_i · ℓ_i · τ^W_{ij}
    ΔU_{ij} = -η · g_i · ℓ_i · τ^U_{ij}
    ΔV_{ij} = -η_V · e_i · h_j     # direct delta
    W += ΔW; U += ΔU; V += ΔV

    h_prev ← h
```

## 2.6 Extensions (not in the Step-3/4 first run)

- **Two-level tag state**: early tag (seconds) + late tag (minutes); capture converts early → late. Matches the biological STC two-step protocol ([@ziegler2015]).
- **Predictive-coding surprise**: replace the supervised-error-derived $s_i$ with a *locally-computed* prediction error from a dendritic compartment, removing the remaining global broadcast.
- **Cell-type-specific $\theta$**: different populations have different capture thresholds (analogous to cell-type-specific dopamine sensitivity).

# 3. Biological Mapping

| Model quantity | Biological interpretation |
|---|---|
| $\tau^W_{ij}$ (fast tag) | Synaptic tag: AMPA receptor phosphorylation, CaMKII activation, actin-cytoskeleton priming at the spine. Window ~minutes. |
| $\phi'(u_i) \cdot h_j$ | Pre × post coincidence detector — the STDP-/Hebbian-style activity correlate that sets the tag. |
| $\alpha_\tau$ | Tag decay rate (dephosphorylation / protein turnover). |
| $B$ | Fixed divergent projection from the teaching population (random feedback, as in FA/RFLO). |
| $\ell_i = B_i e$ | Synaptically-local teaching current arriving at neuron $i$ (climbing-fibre-like or thalamic; a broadcast signal, but per-neuron via random feedback). |
| $s_i > \theta$ | Release of plasticity-related proteins (PRPs) triggered by a salient / surprising event: a dopaminergic burst, a novelty signal, or a neuromodulatory pulse exceeding threshold. |
| $g_i(t) = 1$ (capture event) | Protein-synthesis-dependent late-LTP consolidation: the tagged synapses are *captured* into a long-lasting weight change. |
| $\Delta W_{ij} = -\eta \, g_i \, \ell_i \, \tau^W_{ij}$ | Tagged, captured synapses update in the direction of the local teaching current; un-tagged or uncaptured synapses do not. |

The headline metaphor: **weight changes happen only at "important" moments, but what gets changed is determined by activity over the preceding seconds**. This is a direct operationalization of Frey & Morris (1997)'s original STC hypothesis as the primary credit-assignment mechanism — not an incidental consolidation step.

# 4. Novelty Audit

This audit was run on 2026-05-10 against the *current* design (not the initial H1 sketch). It targets the seven closest prior mechanisms.

## 4.1 Queries

- **Q1**: "synaptic tagging capture recurrent neural network credit assignment supervised learning"
- **Q2**: "threshold gated eligibility trace e-prop recurrent learning surprise"
- **Q3**: "predictive coding error modulates eligibility trace biologically plausible"
- **Q4**: "two timescale synaptic plasticity tag consolidation supervised learning recurrent"
- **Q5**: "reward prediction error gated eligibility trace learning rule neural network"
- **Q6**: "surprise threshold dopamine eligibility late LTP recurrent neural network learning"
- **Q7**: "Izhikevich solving distal reward problem dopamine eligibility"

## 4.2 Top hits considered and verdicts

**Q1 hits.** The only on-topic result was [@huertas2014] (competing LTP/LTD traces for stable RL — relevant but not STC-capture; no thresholded gate, no recurrent credit-assignment rule). **Verdict: no near-match.** [@luboeinski2020], surfaced in Step 1, uses STC in RNNs for *memory consolidation* of input patterns — not as the update rule driving a supervised temporal-credit-assignment learner. **Verdict: near-domain, different task and update structure.**

**Q2 hits.** No threshold-gated e-prop or RFLO variant in the literature; surface results (recent self-prediction-enhanced SNN rules and SuperSpike-style rules [@zenke2017]) use continuous surrogate gradients with no capture step. **Verdict: no near-match.**

**Q3 hits.** The closest match is **Rombouts et al. 2015** [@rombouts2015], which combines an attention gate with RL-style reward-modulated eligibility traces for cortical networks. It uses *attention-gated* rather than *surprise-thresholded* capture, and the task is reinforcement learning of attention, not supervised RNN training. Mechanism overlaps qualitatively but the functional form ($s_i > \theta$ capture of a pre × post tag with random feedback) is not the same. **Verdict: adjacent, not a near-match.**

**Q4 hits.** [@luboeinski2021] extends Luboeinski-Tetzlaff to priming/organization of LTM; still memory-storage-focused, no supervised learning rule. **Verdict: no near-match.**

**Q5 hits.** Most relevant are [@fernandez2025] (node-perturbation with variance reduction — different mechanism), [@tsurumi2025] (online RL with random feedback — overlaps on random-feedback ingredient but continuous, not gated), and [@rombouts2015] (discussed above). [@liu2022modprop] (ModProp) uses continuous cell-type-specific multiplicative gains — not thresholded. **Verdict: no near-match.**

**Q6 hits.** [@izhikevich2007] ("Solving the distal reward problem through linkage of STDP and dopamine signaling") is the closest conceptual predecessor: STDP sets an eligibility trace, and dopamine release *multiplies* it to determine plasticity. Izhikevich's rule **does use** a dopamine signal that is sparse and event-driven in realistic settings — but the mathematical form multiplies continuously without an explicit threshold, and there is no supervised RNN credit-assignment evaluation. **Verdict: qualitative predecessor, different mathematical form (continuous modulation vs thresholded capture) and different task regime (RL with delayed reward vs supervised temporal credit for prediction tasks).** This is the most important near-match to discuss.

**Q7 hits.** Just confirmed Izhikevich 2007 is the same predecessor as in Q6.

## 4.3 Verdict

**Overall: novel.** STC-Credit's combination of (i) RFLO-style random-feedback ell, (ii) e-prop/RFLO-style eligibility-tag dynamics, and (iii) an explicit *thresholded* capture event is not claimed in any surfaced prior work. The closest predecessors are:

- **Izhikevich 2007** [@izhikevich2007]: continuous dopamine × STDP trace for RL, not thresholded capture, not for supervised RNN credit.
- **Luboeinski & Tetzlaff 2020** [@luboeinski2020]: STC in RNNs for memory consolidation, not credit assignment.
- **Rombouts et al. 2015** [@rombouts2015]: attention-gated eligibility in cortical RL, mechanistically adjacent but with a continuous attention signal and an RL task.

None of these claim that STC's thresholded capture is the *update-triggering mechanism for gradient-descent-style supervised learning in a recurrent network*. STC-Credit makes that specific claim.

**Residual risk.** Q3 and Q5 returned zero direct hits on "threshold-gated eligibility trace"; this is thin evidence and should be treated as *absence-of-evidence*. The most likely place where the idea could be hidden is a spiking-neural-network paper under a different vocabulary ("gated plasticity", "event-driven weight update"). Before committing resources beyond Step 3, one additional search in the SNN literature is warranted (Step 4 task).

## 4.4 SNN-vocabulary audit (Step 4, 2026-05-10)

This audit closes the residual risk from §4.3 by re-running the novelty check in SNN / neuromorphic vocabulary. Six new queries (Q8–Q13) were run, plus one follow-up (Q14).

### Queries

- **Q8**: "event-driven plasticity spiking neural network recurrent learning rule"
- **Q9**: "gated STDP threshold consolidation spiking network learning rule"
- **Q10**: "protein synthesis dependent plasticity spiking neural network supervised"
- **Q11**: "sparse plasticity events dopamine threshold spiking recurrent network"
- **Q12**: "synaptic tagging capture spiking neural network temporal credit assignment"
- **Q13**: "surprise gated plasticity reward modulated spiking recurrent"
- **Q14**: "threshold crossing plasticity gate eligibility trace SNN"

### Adjacencies found

Four papers required individual reading to classify:

**Lehr, Luboeinski & Tetzlaff 2022** [@lehr2022] — "Neuromodulator-dependent synaptic tagging and capture retroactively controls neural coding in spiking neural networks" — implements full biophysical STC (tag + late-phase capture) in a recurrent spiking network with a *continuous* neuromodulator concentration. The level of neuromodulation selects *which* learned patterns consolidate (rate-coded vs timing-coded). **Not** a supervised-learning rule, **not** thresholded, **not** driven by prediction-error surprise. Most vocabulary-similar paper surfaced but fundamentally a memory-consolidation study, as with its 2020 predecessor [@luboeinski2020]. **Verdict: adjacent, not near-match.**

**Yamada & Chao 2025** [@yamada2025] — "Joint encoding of 'what' and 'when' predictions through error-modulated plasticity in biologically-plausible spiking networks" — uses an error-modulated **attention-gated** three-factor Hebbian rule in an Izhikevich spiking reservoir with multiplexed readouts. Gating is *attention-based* (continuous multiplicative) rather than surprise-thresholded. Underlying network is a *fixed reservoir*, not a trained RNN. Similar in spirit to Rombouts 2015 [@rombouts2015]. **Verdict: adjacent, not near-match.**

**Apolinario, Roy & Frenkel 2025 (TESS)** [@tess2025] — "A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks" — eligibility traces + STDP + neural activity synchronization, trained SNN, matches BPTT within ~1.4 accuracy points on DVS-Gesture / CIFAR10-DVS. No thresholded capture; synchronization plays the role of a spatial modulator, not an event trigger. **Verdict: different mechanism in the same family; the strongest local-SNN-rule *baseline* we should be aware of, but not a near-match for STC-Credit.**

**Dong & He 2026 (AGMP)** [@agmp2026] — "Astrocyte-gated multi-timescale plasticity for online continual learning in deep spiking neural networks" — eligibility traces + broadcast teaching signal + astrocyte-mediated gate, trained SNN. The closest SNN-vocabulary analog of STC-Credit's two-timescale gating. *Critically*, the gate here is a slow astrocytic variable that **integrates neuronal activity** — *stability* oriented (suppresses updates in stable regimes, opens during distribution shifts) — rather than an error-driven **surprise** threshold. Their goal is continual-learning stability-plasticity, not credit assignment. **Verdict: adjacent with partial overlap; must be cited as the primary SNN-side prior on multi-timescale gated plasticity, and distinguished carefully.**

### Revised verdict

Narrower than the Step-2 verdict, but still **novel**. Specifically:

- STC-Credit is the first rule I have been able to surface that uses **(i) a binary/thresholded capture gate (ii) driven by prediction-error surprise (iii) as the supervised-RNN credit-assignment mechanism** — the combination of all three.
- Prior work has each ingredient individually:
  - Thresholded consolidation in SNNs → Lehr 2022 (memory, not credit).
  - Error-driven plasticity gate → Yamada 2025 (continuous, reservoir).
  - Multi-timescale gated plasticity in trained SNNs → Dong & He 2026 (stability-driven, not error-driven).
  - STC as credit-assignment → Izhikevich 2007 (continuous dopamine, RL).
- No paper combines all three. Confidence in H1 novelty updated from ~40% (Step 2) to **~45%** after Step 4 — the novelty claim survives, but the margin is tighter.

### Implications for the design

1. **Must-cite additions** to Related Work in any future paper: Lehr 2022, Yamada 2025, TESS 2025, AGMP 2026. Step-2 design doc should now reference these (done — this subsection adds them).
2. **Distinction to emphasize**: STC-Credit's gate is *error-driven* and *binary*; AGMP's is *activity-driven* and *continuous*; Lehr's is *neuromodulator-level-driven* and *continuous*; Yamada's is *attention-driven* and *continuous*.
3. **Additional falsifiability**: a cleaner Step-6 comparison should include AGMP-style activity-integrated gating as a baseline (not just θ=−∞ and small-θ), to test whether the *error-driven* aspect of STC-Credit matters over and above *any* gating.

# 5. Biological-Plausibility Checklist

| Desideratum | STC-Credit? | Note |
|---|---|---|
| **No weight transport (WT)** | ✓ | $B$ is a fixed random matrix, never updated; no transpose of $V$ is used. |
| **Locality (Loc)** | ✓ | $\Delta W_{ij}$ depends on pre ($h_j$), post ($\phi'(u_i)$), a per-neuron broadcast $\ell_i$, a scalar gate $g_i$ — all locally available at the synapse of neuron $i$. |
| **Online (Onl)** | ✓ | Per-step memory is $\mathcal{O}(N^2 + N_\text{params})$; no activity history is stored. |
| **No separate phase (NoP)** | ✓ | All updates occur during the forward pass; no nominal backward sweep or clamped phase. |
| **Causality (Cau)** | ✓ | At time $t$, the tag $\tau_{ij}(t)$ depends only on history up to $t$, and the capture gate $g_i(t)$ uses only information available at $t$. |
| **"Surprise" requirement** | ✓ | Thresholded capture is the mission's *conceptual-insight* ingredient: a gradient update that only fires at surprising moments, committing the tag that integrated over the preceding window. Not a trivial extension of any single prior rule. |

# 6. Falsifiability

Three signatures that would refute STC-Credit:

1. **F1 (core learning claim)**. If STC-Credit fails to drive any learning on a copy task of length 10 — specifically, if the test error is ≥ chance after 10k gradient-step equivalents on a vanilla RNN of hidden size 64 — then the thresholded capture step is preventing signal propagation entirely, and the rule is broken. This is a binary check.

2. **F2 (gating vs continuous modulation)**. If the ablation "$\theta = -\infty$" (no threshold, i.e. continuous updates) strictly dominates the gated version across all hyperparameters, then the capture step offers no benefit and the rule reduces to a random-feedback eligibility-trace rule (essentially RFLO) — that is, *the novel ingredient is not doing anything*.

3. **F3 (biological-metaphor claim)**. If $\theta$ has to be tuned so low that $g_i = 1$ on >80% of timesteps to get competitive performance, then the "sparse capture event" metaphor is empirically wrong — the rule is really a continuous modulator, just labeled differently. Empirically we want $g_i = 1$ on ~5–20% of timesteps for the rule to be meaningfully STC-flavored.

F2 and F3 are exactly the ablations built into E4.

# 7. Hyperparameters & defaults (Step 3/4 starting point)

| Name | Symbol | Default |
|---|---|---|
| Hidden size | $N$ | 64 |
| Input dim | $n_x$ | $n_\text{symbols} + 1$ (+1 for task cue, task-dependent) |
| Activation | $\phi$ | $\tanh$ |
| Tag decay | $\alpha_\tau$ | 0.1 (giving $\sim 10$-step window) |
| Feedback matrix | $B$ | $\mathcal{N}(0, 1/\sqrt{n_y})$ |
| Surprise threshold | $\theta$ | 2.5 (z-score; fires on $\sim 5\%$ of steps) |
| EMA constant | — | 0.01 |
| Learning rate | $\eta$ | $3\times10^{-3}$ |
| Readout LR | $\eta_V$ | $1\times10^{-3}$ |

# 8. Exit criteria / handoff to Step 3

Step 3 (next) does **not** implement STC-Credit yet. Step 3 establishes the BPTT baseline on the copy task so that STC-Credit's quality can be judged later. Specifically Step 3 will:

- Run a vanilla RNN (N=64, tanh) with BPTT on a copy task of length 10 (5 distinct input symbols, output target = delayed copy).
- Report per-epoch test accuracy, final cross-entropy, and wall-clock / memory footprint.
- Provide the reference number for subsequent STC-Credit / RFLO / e-prop comparisons.

A follow-up Step 4 will target an SNN-vocabulary novelty audit (see Residual risk in §4.3). Step 5 is the first STC-Credit implementation experiment.

# 9. References

The full bibliography is in `references.bib`. Key citations in this design: [@murray2019] (RFLO, the closest ancestor for the tag dynamics), [@bellec2020] (e-prop, the closest ancestor for post-synaptic factor choice), [@liu2022modprop] (ModProp, the main continuous-modulation alternative), [@izhikevich2007] (distal-reward dopamine-eligibility predecessor), [@clopath2008; @ziegler2015; @benna2016] (STC biology), [@luboeinski2020; @luboeinski2021] (STC in RNNs for memory), [@rombouts2015] (attention-gated eligibility, the nearest mechanism-level adjacency), [@gerstner2018; @fremaux2016] (three-factor rule framework).
