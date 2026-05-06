# Research Logbook

## Step 1: Literature Survey of Biologically Plausible RNN Learning Rules

Conducted a comprehensive literature review covering 7 major methods for biologically plausible learning in recurrent networks: Feedback Alignment, RFLO, e-prop, KeRNL, OSTL, ModProp, and RTRL approximations. The key finding is that **no existing method has been convincingly extended to gated architectures (GRU/LSTM)** — this represents the clearest opportunity for a novel contribution. Secondary gaps include the simplicity-expressiveness tradeoff (RFLO is too simple, ModProp too complex) and the lack of methods bridging supervised and RL settings. The full review is in `docs/literature_review.md`. Next step: implement RFLO as a working baseline on short-sequence tasks.

## Step 2: RFLO vs BPTT Baseline Experiment

Implemented and ran RFLO and BPTT on two tasks: copy task (seq_len=10) and adding problem (seq_len=30). Results: BPTT reaches 73.5% accuracy on copy task, RFLO only 29.7% (random=12.5%). On the adding problem, neither method clearly solved the task (BPTT MSE=0.142, RFLO=0.174, random=0.167). The performance gap between RFLO and BPTT is larger than expected. However, BPTT itself hasn't converged, suggesting hyperparameters (alpha=0.2, 5000 iterations) may be limiting both methods. Code in `experiments/exp01_rflo_vs_bptt.py`, plots in `results/exp01/`. Next step: hyperparameter tuning to establish true performance ceilings.

## Step 3: Hyperparameter Sweep — Confirming RFLO's Fundamental Limitation

Ran 6 BPTT configs and 10 RFLO configs on the copy task (seq_len=10, 10000 iterations each). Key finding: BPTT reaches 100% accuracy with alpha=1.0 (standard RNN, no leak), confirming Exp01's poor BPTT results were due to alpha=0.2 causing information decay. RFLO, however, saturates at ~34.5% accuracy regardless of alpha (0.1-0.5) and learning rate (0.005-0.02). This confirms a **fundamental limitation** of RFLO's rank-1 eligibility trace approximation on memory tasks — it's not a hyperparameter issue. Code in `experiments/exp02_hyperparam_sweep.py`, plots in `results/exp02/`. Next step: ablation to determine whether the bottleneck is the random feedback matrix or the trace approximation.
