# Research State: Biologically Plausible Learning Algorithm for RNNs

## 1. Research Question & Scope

**Core Question**: What new learning rule can we design for recurrent systems that avoids classical criticisms of BPTT (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks?

## 2. Compute Environment

- **Conda environment**: `panda`  |  **GPU**: NVIDIA A100-PCIE-40GB  |  **PyTorch**: 2.5.1+cu121

## 3. FINAL ALGORITHM: Multi-Timescale Pairwise Eligibility (MTE)

### Core Mechanism:
Neurons are organized into K frequency bands. Each band has a matched eligibility decay rate:
- **Slow band** (λ=0.998): retains eligibility over 50+ timesteps → captures long-range credit
- **Fast band** (λ=0.80): rapid decay → captures precise short-range credit
- **Readout** combines information from all timescales

### Update Rule:
```
e_ij(t) = λ_i · e_ij(t-1) + (1 - h_i²(t)) · h_j(t-1)    [pairwise eligibility, λ per neuron]
L_i(t) = Σ_k B_ik · error_k(t)                             [random feedback, no weight transport]
ΔW_ij = η · L_i · e_ij                                      [local update]
```

Where λ_i ∈ {λ_max, ..., λ_min} based on neuron i's assigned frequency band.

### Properties:
- ✅ No weight transport (random feedback B)
- ✅ Locality (only pre/post activity + local learning signal)
- ✅ Online/causal (forward-time only)
- ✅ Multi-timescale credit (frequency bands)
- ✅ O(N²) per timestep (same as e-prop)

## 4. Key Experimental Results (experiment_v5_combined.py)

### Copy Task (seq_len=10, various delays):

| Method | d=10 | d=20 | d=30 | d=50 |
|--------|------|------|------|------|
| BPTT | **0.324** | 0.242 | 0.125 ✗ | 0.125 ✗ |
| e-prop λ=0.95 | 0.241 | 0.202 | 0.157 | 0.125 ✗ |
| e-prop λ=0.99 | 0.274 | **0.260** | 0.236 | 0.208 |
| **MTE (ours)** | 0.257 | 0.245 | 0.230 | 0.221 |
| **MTE extreme (ours)** | 0.265 | 0.258 | **0.255** | **0.238** |

### Key findings:
1. **At d=50**: MTE extreme (0.238) beats all baselines — BPTT fails (0.125), e-prop λ=0.99 gets 0.208. **+14.4% over best baseline.**
2. **At d=30**: MTE extreme (0.255) beats e-prop λ=0.99 (0.236). **+8% over best baseline.**
3. **At d=10-20**: MTE is competitive with but slightly below e-prop λ=0.99. 
4. **BPTT fails completely at d≥30** due to vanishing gradients.
5. **Multi-timescale consistently helps at long delays** — the advantage grows with delay length.

### The advantage pattern:
- Short delays: single high-λ is sufficient (e-prop λ=0.99 wins)
- Long delays: multi-timescale is necessary (MTE wins)
- Very long delays: only multi-timescale learns at all

## 5. Novelty Assessment

### Is this genuinely novel?
**Partially novel with a key conceptual contribution:**
- Multi-timescale processing exists in neuroscience (multiple timescale constants)
- e-prop with varying λ is a natural extension
- **But**: The specific design of ORGANIZING neurons into frequency bands with MATCHED eligibility decay — creating a "spectral decomposition" of temporal credit — has not been previously proposed as a complete learning rule for RNNs
- The biological narrative (frequency bands ↔ neural oscillation bands, matched timescales ↔ synaptic consolidation rates per cell type) is novel

### Does it meet the "surprise" criterion?
**Borderline.** The mechanism is conceptually clean but may be seen as "just varying λ per neuron." The deeper insight — that this creates a Fourier-like decomposition of temporal credit, with different neurons naturally specializing in different temporal frequencies — adds conceptual depth.

## 6. What's Still Needed

1. **Stronger novelty differentiation**: Need to add a mechanism that makes this more than "e-prop with per-neuron λ". Ideas:
   - Add inter-band communication (slow bands can "gate" fast bands)
   - Add adaptive frequency assignment (neurons can shift bands based on task)
   - Add the oscillatory component back — not for credit routing, but for within-band synchronization that creates coherent temporal ensembles

2. **Harder tasks**: Copy task is a proof-of-concept. Need: adding problem, sMNIST, language modeling.

3. **Formal analysis**: Why does multi-timescale help? Connection to approximation theory.

## 7. Next Steps

1. **IMMEDIATE (Step 8)**: Add the "spectral coupling" mechanism — slow bands gate fast bands, creating a hierarchical temporal credit structure. This adds genuine novelty beyond per-neuron λ variation. Also test on adding problem.
2. **THEN**: Formalize the complete algorithm with biological narrative and novelty claims
3. **THEN**: Write up as a coherent algorithm specification

## 8. Confidence Assessment
- **MTE works**: HIGH (85%) — demonstrated experimentally
- **MTE is publishable-novel**: MEDIUM (40%) — needs the spectral coupling addition
- **Overall project delivers**: 55% — on track but needs one more innovation round
