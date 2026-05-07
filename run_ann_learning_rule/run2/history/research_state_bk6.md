# Research State: Biologically Plausible Learning Algorithm for RNNs

## 1. Research Question & Scope

**Core Question**: What new learning rule can we design for recurrent systems that avoids classical criticisms of BPTT (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks?

## 2. Compute Environment

- **Conda environment**: `panda`
- **GPU**: NVIDIA A100-PCIE-40GB
- **PyTorch**: 2.5.1+cu121

## 3. Experimental Results So Far

### Full comparison (experiment_v3_pre.py, 10K steps, hidden=64):

| Method | Delay=5 | Delay=10 | Delay=20 |
|--------|---------|----------|----------|
| BPTT | **0.873** | **0.508** | 0.125 (fails) |
| e-prop (our impl) | 0.253 | 0.235 | 0.164 |
| PRE (phase-resonant) | 0.246 | 0.215 | 0.165 |
| Random baseline | 0.125 | 0.125 | 0.125 |

### Key Findings:
1. **Phase mechanisms (OPCR v1, v2, PRE) do NOT help** — both addressing and resonant-decay variants fail to improve over basic e-prop
2. **Our e-prop is very weak vs BPTT** (~3x worse on delay=5)
3. **Interesting**: At delay=20 where BPTT fails (vanishing gradients), local rules (e-prop, PRE) actually outperform BPTT — the local rules are more robust to long delays even if weaker overall
4. **The fundamental bottleneck is NOT phase** — it's that feedback alignment + simple eligibility traces can't propagate sufficient credit through the recurrent dynamics

## 4. Diagnosis and Pivot

### Why Our e-prop is Weak:
The published e-prop paper (Bellec et al. 2020) gets much better results because:
1. They use **spiking networks** with adaptive thresholds (richer eligibility structure)
2. They use **symmetric e-prop** (non-biological variant) for best results
3. The **random feedback** variant (biologically plausible) is known to be weaker
4. Our RNN is a vanilla tanh network — the eligibility trace $e_{ij} = \lambda e_{ij} + (1-h_i^2) \cdot h_j$ contains less information than in the spiking case

### Strategic Pivot:
Rather than trying to make phase help a fundamentally weak base rule, we should:
1. **Accept that local rules are inherently weaker than BPTT** for short delays
2. **Focus on the regime where BPTT fails**: long delays (d≥20) where vanishing gradients kill BPTT
3. **Design our novel mechanism to extend the temporal reach** of local rules, not to match BPTT at short delays

### NEW ALGORITHM CONCEPT: Phase-Frequency Spectrum Credit (PFSC)

**Core insight**: The problem with exponential eligibility decay is that it creates a SINGLE timescale. The problem with our phase-bin approach was signal dilution. What if instead we use the FREQUENCY SPECTRUM of neural oscillations to create a BANK OF TIMESCALES — not by splitting the signal, but by having different neurons specialize at different temporal scales?

**Mechanism**:
- Neurons are assigned to different frequency bands (slow/medium/fast oscillators)
- Each neuron's eligibility decay rate is FIXED based on its frequency: slow oscillators = slow decay = long memory
- The readout learns to weight neurons from different timescales appropriately
- Phase is used NOT for credit routing but for SYNCHRONIZATION: neurons in the same phase band share information via lateral connections
- Credit assignment uses standard feedback alignment but with the multi-timescale structure providing natural temporal decomposition

**Why this might work**: It's like a Fourier decomposition of the temporal credit — slow neurons carry coarse/long-range credit, fast neurons carry precise/short-range credit. The readout combines them.

**Why novel**: No existing work uses frequency-stratified neuron pools with matched eligibility timescales as the primary temporal credit mechanism. ModProp uses filters on traces but not the structural assignment of neurons to frequency bands.

## 5. Next Steps

1. **IMMEDIATE (Step 6)**: Implement and test the PFSC algorithm — frequency-stratified neuron pools where eligibility decay matches oscillator frequency
2. **Test specifically at delay=20 and delay=30** where BPTT fails — our algorithm should excel here
3. **Compare against e-prop on long-delay regime**
4. **If promising**: Scale to delay=50+ and harder tasks

## 6. Confidence Assessment

- **OPCR (phase-as-address)**: 5% — conclusively failed, mechanism doesn't help
- **PRE (phase-resonant decay)**: 10% — marginal, not useful
- **PFSC (frequency-stratified pools)**: 45% — new idea, theoretically sound, untested
- **Overall project success**: 40% — multiple attempts needed, learning from failures
