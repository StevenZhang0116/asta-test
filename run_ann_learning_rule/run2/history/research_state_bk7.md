# Research State: Biologically Plausible Learning Algorithm for RNNs

## 1. Research Question & Scope

**Core Question**: What new learning rule can we design for recurrent systems that avoids classical criticisms of BPTT (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks?

## 2. Compute Environment

- **Conda environment**: `panda`
- **GPU**: NVIDIA A100-PCIE-40GB
- **PyTorch**: 2.5.1+cu121

## 3. Full Experimental Results

### v4 Results — Copy Task, seq_len=10 (experiment_v4_pfsc.py):

| Method | d=10 | d=20 | d=30 | d=50 |
|--------|------|------|------|------|
| BPTT | 0.324 | 0.210 | **0.125** (fails) | **0.125** (fails) |
| e-prop | 0.271 | **0.257** | **0.211** | **0.124** (fails) |
| PFSC | 0.195 | 0.165 | 0.146 | **0.156** ✓ |
| PFSC-noSync | 0.198 | 0.169 | 0.161 | 0.153 |
| Random | 0.125 | 0.125 | 0.125 | 0.125 |

### KEY BREAKTHROUGH FINDING:
**At delay=50, PFSC is the ONLY method that learns** (0.156 vs 0.125 random). BPTT and e-prop both completely fail. The frequency-stratified neuron pools with matched eligibility decay rates successfully carry credit over 50 timesteps where all other methods lose the signal.

### Nuanced picture:
- d=10-30: e-prop > PFSC (the factored eligibility trace in PFSC is weaker than full pairwise in e-prop)
- d=50: PFSC > e-prop > BPTT (PFSC's multi-timescale structure shines at very long delays)
- Phase synchronization doesn't help much (PFSC ≈ PFSC-noSync)

## 4. Algorithm Status

### OPCR (phase-as-address): ❌ FAILED — Phase bin mechanism hurts, doesn't help
### PRE (phase-resonant decay): ❌ FAILED — No improvement over baseline
### PFSC (frequency-stratified pools): ✅ PARTIAL SUCCESS
- Works uniquely well at very long delays (d=50)
- Underperforms at short/medium delays due to factored eligibility approximation
- Phase synchronization component adds no value

## 5. Diagnosis and Next Direction

### Why PFSC works at d=50 but not d=10-20:
1. **Multi-timescale structure**: The slow band (λ=0.995) retains 0.995^50 = 78% of eligibility at t-50, while uniform λ=0.95 retains only 0.95^50 = 7.7%. This explains why PFSC succeeds where e-prop fails.
2. **Why PFSC loses at short delays**: The factored eligibility (trace_i * pre_j) is a rank-1 approximation of the true pairwise eligibility matrix. At short delays where pairwise structure matters, this loses information. At long delays, the dominant signal is just "was there activity back then?" which the factored form captures.

### Improved algorithm: Combine strengths of e-prop (pairwise eligibility) with PFSC (multi-timescale)

**PFSC-v2: Multi-Timescale Pairwise Eligibility**
- Use FULL pairwise eligibility e_ij(t) (not factored)
- BUT assign each neuron to a frequency band with matched λ
- Eligibility λ_ij = λ_band(i) (decay based on POST-synaptic neuron's band)
- This gives the best of both: rich pairwise structure + multi-timescale credit

The cost: O(N²) memory for eligibility matrix (same as e-prop). But with N=64 this is fine.

## 6. Next Steps

1. **IMMEDIATE (Step 7)**: Implement PFSC-v2 with full pairwise eligibility + multi-timescale bands. Test on d=10, 20, 30, 50. Goal: beat e-prop at ALL delays while maintaining the d=50 advantage.
2. **THEN (Step 8)**: If PFSC-v2 works, formalize it properly and write the algorithm specification. Test on harder tasks (adding problem, sMNIST).
3. **THEN**: Assess novelty — is multi-timescale eligibility with frequency-stratified neurons genuinely novel enough?

## 7. Confidence Assessment

- **PFSC-v2 (multi-timescale pairwise)**: 55% — combines proven strengths, should work
- **Overall project delivers novel algorithm**: 50% — PFSC-v2 might work but novelty needs careful assessment (multi-timescale approaches exist, though not with this specific structure)
