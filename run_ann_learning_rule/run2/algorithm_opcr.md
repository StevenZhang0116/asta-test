# OPCR: Oscillatory Phase Credit Routing

## A Novel Biologically Plausible Learning Rule for Recurrent Neural Networks

---

## 1. Overview

**OPCR** (Oscillatory Phase Credit Routing) is a learning algorithm for recurrent neural networks that uses oscillatory phase relationships between neurons to route temporal credit assignment signals. Unlike standard eligibility trace methods (which provide exponentially decaying, unstructured temporal credit), OPCR uses phase as a *temporal address system* — enabling selective, targeted credit assignment to specific temporal offsets.

**Key Insight**: In biological neural circuits, oscillatory rhythms (theta, gamma) provide a temporal reference frame. Phase precession in hippocampal place cells demonstrates that phase encodes temporal/spatial position. OPCR exploits this principle: the phase relationship between two neurons at the time of activity *encodes* the temporal context, and credit signals are routed to synapses whose phase-encoded temporal context matches the delay of the causal effect being credited.

---

## 2. Network Architecture

### 2.1 Base RNN with Oscillatory Modulation

Consider a vanilla RNN with $N$ hidden neurons. Each neuron $i$ has:
- A **state** $h_i(t)$ (standard RNN hidden state)
- An **intrinsic phase** $\phi_i(t) \in [0, 2\pi)$ (oscillatory reference)

**Forward dynamics:**

$$h_i(t) = \sigma\left(\sum_j W_{ij} h_j(t-1) + \sum_k U_{ik} x_k(t) + b_i\right)$$

where $\sigma$ is tanh (or other activation), $W$ is recurrent weight matrix, $U$ is input weight matrix, $x(t)$ is input.

### 2.2 Phase Dynamics

Each neuron has an intrinsic oscillator with learnable frequency:

$$\phi_i(t) = \phi_i(t-1) + \omega_i \cdot \Delta t + \epsilon_i(t)$$

where:
- $\omega_i$ is the intrinsic frequency of neuron $i$ (learnable parameter)
- $\Delta t$ is the discrete timestep
- $\epsilon_i(t)$ is a small activity-dependent phase perturbation: $\epsilon_i(t) = \alpha \cdot h_i(t)$

The phase perturbation $\epsilon_i(t)$ couples the oscillator to neural activity, providing a mechanism for phase precession (active neurons shift phase relative to quiescent ones).

**Frequency distribution**: Neurons have frequencies drawn from a range $[\omega_{min}, \omega_{max}]$, analogous to the theta band. We use $K$ discrete frequency bands (analogous to theta-gamma hierarchy):
- Band 0 (theta): $\omega \in [0.1, 0.3]$ radians/timestep (slow, ~10-30 timestep period)
- Band 1 (beta): $\omega \in [0.3, 0.8]$ radians/timestep
- Band 2 (gamma): $\omega \in [0.8, 2.0]$ radians/timestep (fast, ~3-8 timestep period)

---

## 3. Phase-Dependent Eligibility Traces

### 3.1 Standard Eligibility (for comparison)

In e-prop/RFLO, the eligibility trace is:
$$e_{ij}(t) = \lambda \cdot e_{ij}(t-1) + \frac{\partial h_i(t)}{\partial W_{ij}}$$

This decays exponentially with rate $\lambda$ — it has no temporal structure beyond decay.

### 3.2 OPCR Phase-Structured Eligibility

In OPCR, we maintain a **phase-indexed eligibility bank** for each synapse. Instead of a single scalar trace, each synapse $W_{ij}$ maintains a set of eligibility components indexed by relative phase:

$$e_{ij}^{(m)}(t) = \lambda_m \cdot e_{ij}^{(m)}(t-1) + g_m(\Delta\phi_{ij}(t)) \cdot \text{pre}_j(t) \cdot \text{post}'_i(t)$$

where:
- $m \in \{1, ..., M\}$ indexes the phase bins (M = number of phase slots, e.g., 8)
- $\Delta\phi_{ij}(t) = \phi_i(t) - \phi_j(t) \mod 2\pi$ is the relative phase between post- and pre-synaptic neurons
- $g_m(\Delta\phi) = \exp\left(-\frac{(\Delta\phi - \theta_m)^2}{2\kappa^2}\right)$ is a von Mises-like kernel centered at phase bin $m$'s preferred phase $\theta_m = \frac{2\pi m}{M}$
- $\text{pre}_j(t) = h_j(t-1)$ is the pre-synaptic activity
- $\text{post}'_i(t) = \sigma'(z_i(t))$ is the derivative of the post-synaptic activation (locally available)
- $\lambda_m$ is a decay rate that can differ per phase bin (allowing different timescales)

**Key property**: The eligibility for synapse $ij$ at time $t$ is deposited into the phase bin $m$ that best matches the current phase relationship $\Delta\phi_{ij}(t)$. This creates a *phase-stamped record* of when activity occurred.

### 3.3 Intuition

Imagine the phase relationship as a clock hand. When neuron $j$ activates neuron $i$, the "clock hand" (their phase difference) marks WHEN this happened in the oscillatory cycle. Later, when a credit signal arrives, it specifies WHICH clock position should be credited. Only eligibility stored at the matching phase bin receives the update.

---

## 4. Credit Routing via Phase-Aligned Learning Signals

### 4.1 Error Signal Generation

At output neurons, the error is computed locally:
$$\delta_k(t) = y_k(t) - \hat{y}_k(t)$$

where $y_k(t)$ is the target and $\hat{y}_k(t)$ is the network output.

### 4.2 Feedback Projection (solving weight transport)

The error is projected to hidden neurons via fixed random feedback weights $B$ (as in feedback alignment):
$$L_i(t) = \sum_k B_{ik} \cdot \delta_k(t)$$

### 4.3 Phase-Selective Credit Signal

The critical innovation: the learning signal $L_i(t)$ is modulated by the current phase of neuron $i$ to create **phase-selective credit**:

$$C_i^{(m)}(t) = L_i(t) \cdot g_m(\phi_i(t) - \phi_{\text{ref}}(t))$$

where $\phi_{\text{ref}}(t)$ is a global reference phase (analogous to the theta rhythm that provides a common temporal reference, like a "conductor" for the neural orchestra).

**Interpretation**: The credit signal $C_i^{(m)}(t)$ activates most strongly for the phase bin $m$ that matches neuron $i$'s current position in the oscillatory cycle. This means credit is *tagged with a phase address*.

### 4.4 Alternative: Self-Referencing Phase Credit

For greater biological plausibility (avoiding a global reference), we can use the neuron's own phase history:

$$C_i^{(m)}(t) = L_i(t) \cdot g_m(\phi_i(t))$$

The neuron's own phase at the time of the error acts as the "temporal selector."

---

## 5. Weight Update Rule

### 5.1 The OPCR Update

The weight update combines phase-selective credit with phase-indexed eligibility:

$$\Delta W_{ij}(t) = \eta \sum_{m=1}^{M} C_i^{(m)}(t) \cdot e_{ij}^{(m)}(t)$$

**Expanding:**

$$\Delta W_{ij}(t) = \eta \sum_{m=1}^{M} \left[ L_i(t) \cdot g_m(\phi_i(t)) \right] \cdot e_{ij}^{(m)}(t)$$

### 5.2 Interpretation

This update has a beautiful temporal matching property:
1. Eligibility $e_{ij}^{(m)}$ accumulates activity evidence tagged with phase bin $m$
2. Credit $C_i^{(m)}$ is strongest for the phase bin matching the neuron's current phase
3. The dot product $\sum_m C_i^{(m)} \cdot e_{ij}^{(m)}$ performs **temporal pattern matching**: credit is assigned to historical activity that occurred at the same relative phase position

Because phases cycle with a known period, this means:
- If neuron $i$ is at phase $\theta_m$ when error arrives, credit goes to past activity that occurred when the $i$-$j$ phase relationship was also at $\theta_m$
- Since phase advances at rate $\omega$, this selectively credits activity from $\Delta t = \frac{\theta_m}{\omega}$ timesteps ago

**This implements temporal credit assignment with O(1) complexity per synapse per timestep** (just M multiplications), compared to RTRL's O(n^2).

### 5.3 Frequency Update Rule

The oscillator frequencies $\omega_i$ are also learned to adapt temporal credit ranges:

$$\Delta \omega_i(t) = \eta_\omega \cdot \frac{\partial}{\partial \omega_i} \sum_m C_i^{(m)}(t) \cdot e_{ij}^{(m)}(t)$$

In practice, this simplifies to:

$$\Delta \omega_i(t) = \eta_\omega \cdot L_i(t) \cdot \sum_m \frac{\partial g_m(\phi_i(t))}{\partial \omega_i} \cdot e_{ij}^{(m)}(t)$$

This allows the network to learn the appropriate timescale for each neuron's temporal credit window.

---

## 6. Multi-Scale Credit via Cross-Frequency Coupling

### 6.1 Nested Oscillations

To handle multiple temporal scales simultaneously (short and long dependencies in the same sequence), OPCR uses nested oscillations analogous to theta-gamma coupling:

- **Slow oscillation** (theta, period ~$T_\theta$): provides coarse temporal addressing over long sequences
- **Fast oscillation** (gamma, period ~$T_\gamma < T_\theta$): provides fine temporal addressing within each theta cycle

Each neuron has both a slow phase $\phi_i^{slow}$ and a fast phase $\phi_i^{fast}$:

$$\phi_i^{slow}(t) = \phi_i^{slow}(t-1) + \omega_i^{slow} + \alpha^{slow} h_i(t)$$
$$\phi_i^{fast}(t) = \phi_i^{fast}(t-1) + \omega_i^{fast} + \alpha^{fast} h_i(t) + \beta \sin(\phi_i^{slow}(t))$$

The $\beta \sin(\phi_i^{slow})$ term couples fast to slow: the fast oscillation frequency is modulated by the slow phase (cross-frequency coupling).

### 6.2 Multi-Scale Eligibility

$$e_{ij}^{(m_s, m_f)}(t) = \lambda \cdot e_{ij}^{(m_s, m_f)}(t-1) + g_{m_s}(\Delta\phi_{ij}^{slow}) \cdot g_{m_f}(\Delta\phi_{ij}^{fast}) \cdot \text{pre}_j \cdot \text{post}'_i$$

This creates a 2D temporal address: $(m_s, m_f)$ specifying both coarse and fine temporal position.

### 6.3 Multi-Scale Update

$$\Delta W_{ij}(t) = \eta \sum_{m_s, m_f} C_i^{(m_s, m_f)}(t) \cdot e_{ij}^{(m_s, m_f)}(t)$$

---

## 7. Biological Plausibility Analysis

### 7.1 Properties Satisfied

| Property | How OPCR satisfies it |
|----------|----------------------|
| **No weight transport** | Uses random feedback weights $B$ (feedback alignment) |
| **Locality** | Update uses only: pre-synaptic activity $h_j$, post-synaptic derivative $\sigma'$, local phase $\phi_i, \phi_j$, and local learning signal $L_i$ |
| **Online/causal** | All computations are forward-in-time; eligibility traces update causally |
| **No global error broadcast** | Error projected via random $B$; each neuron receives its own projected signal |
| **Biological correspondence** | Phase = neural oscillation; eligibility bank = synaptic tagging at specific phases; credit routing = phase-dependent plasticity |

### 7.2 Biological Correspondence Table

| OPCR Component | Biological Analog |
|----------------|-------------------|
| Phase $\phi_i(t)$ | Theta/gamma oscillatory rhythm of neuron |
| Phase bins $m$ | Discrete gamma cycles within theta |
| Eligibility bank $e_{ij}^{(m)}$ | Synaptic tags set at specific oscillatory phases |
| Phase-selective credit $C_i^{(m)}$ | Phase-dependent LTP/LTD windows |
| Cross-frequency coupling | Theta-gamma coupling in hippocampus |
| Frequency adaptation $\Delta\omega$ | Experience-dependent changes in oscillatory frequency |
| Reference phase $\phi_{ref}$ | Global theta rhythm (hippocampal theta pacemaker) |
| Random feedback $B$ | Non-specific neuromodulatory projections |

### 7.3 Experimental Predictions

If OPCR captures real biological computation, it predicts:
1. Disrupting oscillatory coherence should selectively impair temporal credit assignment (not just memory)
2. The temporal range of learnable associations should scale with oscillatory period
3. Phase-locked stimulation during learning should be able to artificially extend/restrict credit assignment range
4. Multi-timescale learning should require intact cross-frequency coupling

---

## 8. Computational Complexity

| Operation | Complexity per timestep |
|-----------|------------------------|
| Forward pass | $O(N^2)$ (standard RNN) |
| Phase update | $O(N)$ |
| Eligibility update | $O(N^2 \cdot M)$ for single-scale; $O(N^2 \cdot M_s \cdot M_f)$ for multi-scale |
| Credit computation | $O(N \cdot M)$ |
| Weight update | $O(N^2 \cdot M)$ |
| **Total** | $O(N^2 \cdot M)$ |

Compare:
- BPTT: $O(N^2 \cdot T)$ where $T$ is sequence length (plus non-causal)
- RTRL: $O(N^4)$
- e-prop: $O(N^2)$
- OPCR: $O(N^2 \cdot M)$ where $M$ is number of phase bins (typically 4-16)

OPCR adds a factor of $M$ over e-prop but gains structured temporal credit. With $M=8$, this is a modest constant factor.

---

## 9. Theoretical Analysis

### 9.1 Connection to RTRL

OPCR can be viewed as a structured low-rank approximation to RTRL. The full RTRL influence matrix $\frac{\partial h_i(t)}{\partial W_{jk}}$ captures all temporal dependencies. OPCR approximates this with $M$ rank-1 components, each corresponding to a specific temporal offset encoded by phase:

$$\frac{\partial h_i(t)}{\partial W_{jk}} \approx \sum_{m=1}^{M} g_m(\phi_i(t)) \cdot e_{jk}^{(m)}(t)$$

This approximation is exact when:
1. The temporal dependencies are aligned with the oscillatory period
2. The phase relationships are stable enough to provide consistent temporal addressing

### 9.2 When OPCR Outperforms Simple Eligibility Traces

OPCR should outperform e-prop/RFLO when:
- The task requires assigning credit at specific temporal delays (not just recent past)
- Multiple temporal dependencies coexist in the same sequence
- The relevant delay is longer than the eligibility trace decay constant

OPCR should perform similarly to e-prop when:
- Only short-term dependencies matter
- A single exponential decay is sufficient to capture the credit structure

### 9.3 Failure Modes

OPCR may struggle when:
- The required temporal credit delays are not commensurate with any oscillatory period
- Phase is disrupted by chaotic dynamics (mitigation: use robust phase estimation)
- The number of phase bins $M$ is too small relative to the temporal complexity

---

## 10. Simplified Single-Scale Algorithm (for initial implementation)

For initial experiments, we use the simplified single-scale version:

```
OPCR Algorithm (Single-Scale):
─────────────────────────────

Initialize:
  W[N,N] ~ Normal(0, 1/sqrt(N))     # Recurrent weights
  U[N,D] ~ Normal(0, 1/sqrt(D))     # Input weights  
  B[N,K] ~ Normal(0, 1/sqrt(K))     # Random feedback (fixed)
  omega[N] ~ Uniform(omega_min, omega_max)  # Frequencies
  phi[N] = Uniform(0, 2*pi)         # Initial phases
  e[N,N,M] = 0                      # Eligibility bank
  
  theta_m = 2*pi*m/M for m=1..M     # Phase bin centers
  kappa = 2*pi/(2*M)                # Phase bin width

For each timestep t:
  # 1. Forward pass (standard RNN)
  z = W @ h_prev + U @ x[t] + bias
  h = tanh(z)
  y_hat = V @ h                      # Output (V = readout weights)
  
  # 2. Phase update
  phi = phi + omega + alpha * h      # Phase advances + activity coupling
  phi = phi mod (2*pi)
  
  # 3. Compute phase kernels
  delta_phi[i,j] = phi[i] - phi[j]  # Relative phase matrix
  For m = 1..M:
    G_m[i,j] = exp(-(delta_phi[i,j] - theta_m)^2 / (2*kappa^2))
  
  # 4. Update eligibility bank
  pre = h_prev                       # Pre-synaptic activity
  post_deriv = 1 - h^2              # tanh derivative (local)
  For m = 1..M:
    e[:,:,m] = lambda * e[:,:,m] + G_m * outer(post_deriv, pre)
  
  # 5. Compute error and learning signal
  delta = y[t] - y_hat              # Output error
  L = B @ delta                      # Feedback-aligned signal (local to each neuron)
  
  # 6. Phase-selective credit
  For m = 1..M:
    C_m[i] = L[i] * exp(-(phi[i] - theta_m)^2 / (2*kappa^2))
  
  # 7. Weight update
  For m = 1..M:
    dW += eta * outer(C_m, ones) * e[:,:,m]
  W += dW
  
  # 8. (Optional) Frequency update
  # omega += eta_omega * frequency_gradient
  
  h_prev = h
```

---

## 11. Summary of Novel Contributions

1. **Phase as temporal address**: First use of oscillatory phase as a *structured temporal addressing system* for credit assignment in RNN learning rules

2. **Phase-indexed eligibility bank**: Instead of a single decaying trace, maintains multiple traces indexed by phase — enabling selective temporal credit

3. **Phase-selective credit routing**: Error signals are routed to specific temporal offsets by matching the phase at error time with the phase at activity time

4. **Cross-frequency coupling for multi-scale credit**: Nested oscillations naturally provide hierarchical temporal addressing

5. **Theoretical insight**: OPCR is a structured low-rank approximation to RTRL where the structure comes from oscillatory dynamics

---

## 12. Next Steps

1. **Implement** in PyTorch (vanilla RNN + OPCR update rule)
2. **Test** on copy task (explicit temporal delay), adding problem, sequential MNIST
3. **Compare** against BPTT, e-prop, RFLO baselines
4. **Ablate** key components: phase gating (vs. standard eligibility), number of phase bins M, frequency distribution
5. **Analyze** what temporal structure the phase representations learn
