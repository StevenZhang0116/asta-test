---
bibliography: references.bib
---

# Candidate Learning Rule — STC-Prop: Sparse Capture with Two-Timescale Synaptic Tagging

**Step 2 design document.** This document specifies the first concrete candidate rule for the mission, along with a novelty audit. The choice is `H5` from `research_state.md` — two-timescale synaptic tagging with discrete, local capture events — motivated by the synaptic tagging-and-capture (STC) hypothesis in neuroscience [@liu2020].

---

## 1. Chosen Hypothesis and Motivation

### 1.1 H5 restated

> A two-timescale synaptic tagging mechanism — a fast decaying tag + a slow capturable tag that only updates on *discrete capture events* triggered by local surprise — can provide a low-variance credit-assignment estimator for recurrent networks that addresses multiple BPTT implausibilities (CAUS, MEM, and partially GLOB) at once.

### 1.2 Why this choice (vs. H6–H10)

| Hypothesis | Verdict | Reason |
|---|---|---|
| H5 (sparse-capture STC) | **CHOSEN** | Concrete math; clear biology; sharp surprise claim |
| H6 (E/I-balance in RNN) | Held in reserve | Core idea was just published (2025 [@rossbroich2025]); FF→RNN extension risks reading as incremental |
| H7 (metaplasticity as credit) | Held in reserve | The precise mathematical claim (metaplasticity absorbs non-local Jacobian) needs more theoretical work before specification |
| H8 (self-supervised PC core) | Held in reserve | Sits close to Bio-Mamba 2024 [@qin2024] and EchoSpike 2024 [@graf2024] |
| H9 (three-compartment dendritic) | Held in reserve | Third compartment is not universally supported biologically |
| H10 (astrocytic slow-modulation) | Held in reserve | Related recent work found: AGMP 2026 (see §4 audit) |

### 1.3 Biological substrate — Synaptic Tagging and Capture (STC)

STC is a well-established neuroscience hypothesis [Frey & Morris 1997; Redondo & Morris 2011]. Its computational structure:

1. Recent pre/post-synaptic activity at a synapse sets a **tag** — a local biochemical marker that is silent w.r.t. the synapse's strength and decays over minutes.
2. Independently, the neuron may receive a **capture signal** (plasticity-related-protein, or PRP, synthesis) triggered by salient events: novelty, strong modulatory input, or behavioral outcomes. This capture signal is *local* to the neuron (somatodendritic), not globally broadcast.
3. **Only tagged synapses consolidate.** Untagged synapses ignore the capture. The capture is effectively a discrete event from the synapse's perspective.

Crucially for our design: the biological capture event is *sparse in time* (seconds to minutes apart) and *local to the neuron*. Existing three-factor rules in ML (e-prop [@bellec2019a; @bellec2020], ModProp [@liu2022], RFLO [@murray2019]) use *continuous* per-timestep modulation and do not exploit the asymmetry between the tag and the capture.

Recent papers [@luboeinski2020] and neuromodulator-dependent STC [@gyorgyi2022] have used STC for **memory consolidation** (preserving already-formed memories), not as a **learning rule for solving ML tasks**. This preserves the novelty window.

---

## 2. Mathematical Formulation

### 2.1 Architecture

Vanilla RNN (to start):

$$
h_t = \sigma(W_{\mathrm{rec}}\, h_{t-1} + W_{\mathrm{in}}\, x_t + b),\qquad \hat{y}_t = W_{\mathrm{out}}\, h_t
$$

with loss $\mathcal{L} = \sum_t \ell(\hat{y}_t, y_t)$.

### 2.2 STC-Prop update rule (per synapse $(i,j)$)

**(A) Fast tag** — continuous, RFLO-style eligibility trace:

$$
e_{ij}(t) = \alpha\, e_{ij}(t-1) + \sigma'\!\big(u_i(t)\big)\; h_j(t-1)
$$

where $u_i(t)$ is the pre-activation of unit $i$ at time $t$ and $\alpha \in [0,1)$ sets the fast-tag decay. For $W_{\mathrm{in}}$ the second factor is $x_j(t)$; for $W_{\mathrm{out}}$ the trace is the standard online output-weight term.

**(B) Surprise signal** — per-neuron, local:

$$
\sigma_i(t) \;=\; \big|\,h_i(t) - \hat{h}_i(t)\,\big|
$$

where $\hat{h}_i(t)$ is a locally maintained short-horizon predictor:

$$
\hat{h}_i(t) = \gamma\, \hat{h}_i(t-1) + (1-\gamma)\, h_i(t-1),\qquad \gamma\in[0,1)
$$

Alternative surprise definitions (task-dependent, to be ablated): gradient-magnitude surrogate $|W^\top_{\mathrm{out}} (\hat{y}_t - y_t)|_i$, or a separately-learned predictor network.

**(C) Capture event** — discrete, per-neuron, local:

$$
c_i(t) = \mathbb{1}\!\left[\sigma_i(t) > \theta_i(t)\right]
$$

with a homeostatic adaptive threshold that keeps the event rate $\rho$ per neuron approximately constant:

$$
\theta_i(t+1) = \theta_i(t) + \eta_\theta\,\big(c_i(t) - \rho\big)
$$

Typical $\rho \in [0.05, 0.2]$ — most timesteps produce no update.

**(D) Slow commit trace** — integrates tag at captures only:

$$
s_{ij}(t) = \beta\, s_{ij}(t-1) + c_i(t)\, e_{ij}(t)
$$

**(E) Weight update** — sparse, gated by capture:

$$
\Delta W_{ij}(t) = -\eta\, c_i(t)\, M_{ij}(t)\, s_{ij}(t)
$$

where $M_{ij}(t)$ is a *non-symmetric* feedback signal. The natural choice, to address WT, is a random-feedback projection of the output error:

$$
M_i(t) = \big[B\, (\hat{y}_t - y_t)\big]_i
$$

where $B\in\mathbb{R}^{N\times k}$ is a fixed random matrix (FA-style). $M_{ij}(t) = M_i(t)$ (broadcast per row) to keep the rule per-neuron local.

### 2.3 Pseudocode for one training step

```
for each t in sequence:
    # --- forward pass ---
    u = W_rec @ h + W_in @ x[t] + b
    h_new = sigma(u)
    yhat = W_out @ h_new

    # --- local per-neuron surprise ---
    sigma_i = abs(h_new - h_hat)              # neuron-local
    c = (sigma_i > theta).astype(float)       # discrete capture
    theta += eta_theta * (c - rho)            # homeostasis

    # --- fast tag (RFLO-style) ---
    e_rec = alpha*e_rec + outer(sigma_prime(u), h)     # shape [N,N]
    e_in  = alpha*e_in  + outer(sigma_prime(u), x[t])  # shape [N,d_in]

    # --- slow commit trace ---
    s_rec = beta*s_rec + c[:,None] * e_rec
    s_in  = beta*s_in  + c[:,None] * e_in

    # --- feedback (random, non-symmetric) ---
    err   = yhat - y[t]
    M     = B @ err   # shape [N]   (random feedback projection)

    # --- sparse updates gated by capture ---
    dW_rec = -eta * (c * M)[:,None] * s_rec
    dW_in  = -eta * (c * M)[:,None] * s_in
    dW_out = -eta * outer(err, h_new)   # output is trained normally

    W_rec += dW_rec; W_in += dW_in; W_out += dW_out

    h = h_new
```

### 2.4 Complexity

| Quantity | Cost per step | Memory |
|---|---|---|
| Fast tag $e_{ij}$ | $O(N\cdot (N+d_{\mathrm{in}}))$ | Same |
| Slow commit $s_{ij}$ | $O(N\cdot (N+d_{\mathrm{in}}))$ | Same |
| Surprise $\sigma_i$, threshold $\theta_i$ | $O(N)$ | $O(N)$ |
| Weight update | $O(N\cdot (N+d_{\mathrm{in}}))$ | — |

No full-trajectory storage. Matches RFLO / e-prop asymptotically; adds one extra $N\times N$ matrix ($s_{ij}$) relative to e-prop.

### 2.5 Biological-primitive mapping

| Rule element | Biological interpretation |
|---|---|
| $e_{ij}(t)$ | Fast synaptic tag — short-lived biochemical marker at active synapses (minutes timescale in biology, timesteps in model). |
| $s_{ij}(t)$ | Slow commit trace — tag-that-has-been-captured but not yet consolidated into weight; bridges to the PRP-capture event timing. |
| $c_i(t)$ | Local PRP-synthesis / capture event at neuron $i$. Discrete, sparse. |
| $\sigma_i(t)$ | Somatodendritic surprise signal (novelty, local prediction error, strong modulatory input). |
| $\theta_i(t)$ | Homeostatic threshold — intrinsic excitability adaptation. |
| $M_i(t)$ | Extra-synaptic / random-feedback error signal — neuromodulatory or cortical feedback carrying task-relevant direction, without weight transport. |
| $B$ | Fixed feedback projection — FA-style random weights. |

### 2.6 BPTT implausibilities addressed

| Code | Addressed? | How |
|---|---|---|
| WT (weight transport) | ✅ | Feedback via random $B$, not $W^\top$. |
| LOC (locality) | ✅ | All quantities at synapse $(i,j)$ depend only on neuron-$i$ and neuron-$j$ activity (+ scalar/broadcast $M$). |
| CAUS (causality) | ✅ | Strictly causal (forward) computation; no future info needed. |
| GLOB (global error) | ⚠️ | $M_i$ still uses the output error vector, but only via a *low-rank non-symmetric* projection. Capture is local and drives the *timing* of updates, so the effective global signal is gated down to sparse, informative moments. |
| MEM (memory) | ✅ | Constant per-synapse state. |
| PHASE (phases) | ✅ | Single forward pass; updates computed online. |
| CONT (continuous-time) | ⚠️ | Threshold $c_i(t)$ is discrete. Continuous-time variant would replace $c_i(t)$ with a smooth surprise-weighted capture function. |

### 2.7 Design decisions and trade-offs

1. **Why a threshold on surprise rather than a smooth sigmoid?** Biology's capture mechanism is effectively all-or-none at the synapse level. A threshold gives sparsity — most weights update rarely — which is our central claim: sparse, informative updates yield variance reduction. A smooth gate would blur the "capture event" concept and reduce to a continuous three-factor rule.
2. **Why a per-neuron (not per-synapse) capture event?** The STC biology has PRP synthesis at the neuron level (soma/dendrites) but tags at the synapse level. Per-neuron capture is both biologically grounded and computationally tractable.
3. **Why keep a slow commit trace $s_{ij}$ rather than applying the fast tag immediately at capture?** This allows multiple nearby capture events to sum over the same tagged credit — closer to the biological picture where PRPs remain available for capture over a long window — and it dampens the variance of using a single tag sample.
4. **Why random feedback $B$ rather than sign-concordant or learned feedback?** Start simple: random $B$ is the cleanest way to decouple the rule's novelty from the feedback mechanism. Once the core claim is established, we can swap in sign-concordant or weight-mirror feedback as an ablation / improvement.
5. **Homeostatic threshold:** Needed to keep the capture rate stable across training (otherwise the magnitude of $\sigma_i$ drifts with training phase). This is itself biologically plausible (intrinsic excitability homeostasis).

---

## 3. Differences from the closest prior work

| Prior rule | How STC-Prop differs |
|---|---|
| **RFLO** [@murray2019] | RFLO uses continuous eligibility + continuous scalar random feedback. STC-Prop adds a discrete per-neuron capture gate and a slow commit trace. |
| **e-prop** [@bellec2019a; @bellec2020] | e-prop truncates temporal dependencies and uses continuous modulation. STC-Prop preserves the eligibility substrate but replaces continuous modulation with sparse per-neuron capture. |
| **ModProp** [@liu2022] | ModProp uses continuous cell-type-specific diffusive modulators to carry credit. STC-Prop carries credit via the slow commit trace $s_{ij}$, which is local to the synapse and only consolidated at discrete neuron-local capture events. |
| **Meta-learned three-factor rules with sparse feedback** (2025, found in audit) | They meta-learn the plasticity rule under *sparse reward*. STC-Prop is *not* meta-learned; sparseness is in the *per-neuron capture gate*, not in the feedback schedule. |
| **AGMP** (2026, found in audit) | AGMP uses an astrocyte-like slow state gating eligibility traces for *continual* learning. STC-Prop uses per-neuron *discrete* capture for *supervised* learning, motivated by STC rather than tripartite synapses. The gate is discrete+neuron-local, not continuous+global astrocyte-like. |
| **STC memory consolidation** [@luboeinski2020; @gyorgyi2022] | Both use STC for **retention of already-formed memories**, not as a **learning rule** for solving ML tasks. STC-Prop flips the purpose: STC *is* the credit-assignment mechanism during training. |

---

## 4. Novelty Audit

### 4.1 Queries executed

Search backend: Semantic Scholar via `asta papers search` (2026-05-10).

1. `"synaptic tagging capture neural network learning rule"` (10 hits)
2. `"gated eligibility trace online learning recurrent"` (10 hits)
3. `"event-driven sparse weight update recurrent biologically plausible"` (10 hits)
4. `"two-timescale synaptic plasticity recurrent neural network gradient"` (10 hits)
5. `"surprise-driven synaptic plasticity recurrent learning rule"` (10 hits)
6. `"threshold-gated credit assignment recurrent neural network"` (10 hits)
7. `"discrete capture event plasticity eligibility trace"` (10 hits)
8. `"local surprise signal gates synaptic plasticity recurrent network gradient"` (10 hits)
9. `"variance reduction eligibility trace sparse discrete weight update"` (10 hits)
10. `"synaptic consolidation discrete events gradient descent recurrent"` (10 hits)
11. `"stop-and-commit plasticity rule neural network"` (5 hits)
12. Drill-downs on the highest-overlap hits: AGMP 2026; Meta-learning three-factor with sparse feedback 2025; Luboeinski 2020 STC memory consolidation; Györgyi 2022 neuromodulator-dependent STC.

### 4.2 Candidate overlaps

**Near-matches found:**

- **AGMP (Astrocyte-Gated Multi-Timescale Plasticity, 2026).** Uses a slow astrocytic variable to gate plasticity of an eligibility trace in SNNs for continual learning. *Overlap:* (a) eligibility trace, (b) slow gating variable, (c) online BPTT-alternative. *Differences:* (i) gating in AGMP is *continuous* (astrocyte variable modulates a scalar gate), not discrete per-neuron capture; (ii) AGMP's motivation is stability-plasticity trade-off in continual learning; STC-Prop targets supervised-learning credit assignment; (iii) biological substrate: tripartite synapse (glia) vs. STC (PRP synthesis); (iv) AGMP's gate suppresses updates in *stable* regimes, STC-Prop's gate *triggers* updates at *surprising* moments — conceptually opposite polarity.

- **Meta-learning three-factor with sparse feedback (2025).** Meta-learns plasticity parameters under *sparse reward feedback*. *Overlap:* sparse feedback + three-factor + recurrent. *Differences:* (i) their sparseness is in the *reward schedule* (environment-determined); STC-Prop's sparseness is in the *per-neuron capture gate* (dynamics-determined). (ii) Meta-learning vs. hand-designed rule.

- **Luboeinski & Tetzlaff 2020 "Memory consolidation and improvement by STC in recurrent networks"** and **Györgyi 2022 "Neuromodulator-dependent STC".** Both use STC in RNNs but for *memory consolidation* (stabilizing already-formed memories), not as a *learning rule* for supervised or RL tasks. Their weight update is not gradient-like; ours is a gradient estimator.

**Overlap with no prior work:** the specific combination of (i) per-neuron discrete capture gate driven by local surprise, (ii) slow commit trace bridging tag-and-capture timescales, (iii) random-feedback low-rank error projection, (iv) applied as a supervised-learning rule for RNNs on temporal tasks — is not present in any retrieved paper.

### 4.3 Verdict

**NOVEL with near-matches to document.** The combination (discrete local capture + two-timescale trace + STC-motivated purpose as credit assignment) does not appear in the retrieved literature. Near-matches (AGMP 2026; meta-learned sparse-feedback 2025; Luboeinski 2020) will be cited and explicitly differentiated in any written output. The rule meets the mission's "surprise" bar because it proposes a *discrete, sparse* update regime for a problem the field has attacked only with continuous-modulation rules.

Re-audit trigger: any time the design changes materially (swap in learned feedback, change the capture trigger, combine with dendritic compartments, etc.).

---

## 5. Open questions for Step 3 (implementation + first experiments)

1. **Capture rate $\rho$ tuning.** What range of $\rho$ gives best copy-task performance? Hypothesis: $\rho \approx 0.1$ (sparse but not starved).
2. **Surprise signal choice.** Local-prediction-error ($|h - \hat h|$) vs. feedback-magnitude ($|M_i|$) vs. a learned predictor. Likely the first is the safer / simpler / more biology-grounded choice for the first experiment.
3. **Fast vs. slow decay balance.** $\alpha$ (tag) and $\beta$ (commit) jointly determine the effective temporal horizon. Expect $\beta > \alpha$ so captures consolidate credit across multiple tag windows.
4. **Does sparse capture reduce gradient variance relative to e-prop?** The central theoretical claim. Measure with gradient-alignment cosine against BPTT.
5. **Does the rule benefit from structured feedback $B$?** Random $B$ first; then sign-concordant or weight-mirror as ablation.
6. **Failure mode: if $\sigma_i$ is uninformative about credit, capture becomes random and the rule degenerates.** Need a baseline where $c_i(t)$ is drawn i.i.d. Bernoulli($\rho$) — if matching performance, the surprise signal adds nothing and the design must be revisited.

---

## 6. Pointer to next step

Step 3 will:
- Implement the minimal harness (vanilla RNN + copy task in PyTorch, `panda` conda env, GPU auto-detect per `mission.md`);
- Implement baselines: BPTT, truncated BPTT, RFLO, e-prop, FA;
- Implement STC-Prop;
- Run a first comparison and collect results (loss, wall-clock, memory, gradient-alignment cosine against BPTT, variance of updates).

The novelty-audit verdict above is the *pre-commitment* required by the mission's novelty clause. If any major architectural or mechanistic change is made during Step 3, the audit must be re-run.
