
# ----------
# The below command failed to execute (raised a No module named 'torch' exception)
# import torch

# ----------

# ----------
import subprocess
import sys
import os
print(f"Current Python: {sys.executable}")
print(f"Python version: {sys.version}")
try:
    result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True, timeout=30)
    print("\nConda environments:")
    print(result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)
except Exception as e:
    print(f"conda not found: {e}")
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'torch'], 
                          capture_output=True, text=True, timeout=30)
    print("\nTorch pip info:")
    print(result.stdout)
except Exception as e:
    print(f"pip show torch failed: {e}")
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                          capture_output=True, text=True, timeout=30)
    print("\nInstalled packages (first 50 lines):")
    lines = result.stdout.split('\n')
    for line in lines[:50]:
        print(line)
except Exception as e:
    print(f"pip list failed: {e}")
try:
    import numpy
    print(f"\nnumpy available: {numpy.__version__}")
except ImportError:
    print("\nnumpy not available")
try:
    result = subprocess.run(['conda', 'run', '-n', 'panda', 'python', '-c', 'import torch; print(torch.__version__)'],
                          capture_output=True, text=True, timeout=60)
    print(f"\nTorch in panda conda env: {result.stdout.strip()}")
    if result.stderr:
        print(f"stderr: {result.stderr[:500]}")
except Exception as e:
    print(f"conda run failed: {e}")

# ----------

# ----------
import subprocess
import os
import tempfile
script = '''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

INPUT_DIM = 10
HIDDEN_DIM = 128
OUTPUT_DIM = 8
ALPHABET = 8
SEQ_LEN = 15
N_SYMBOLS = 5
BATCH_SIZE = 64
LR = 1e-3
N_STEPS = 3000
EVAL_EVERY = 100
EVAL_SIZE = 1000
BLANK_CH = 8
GO_CH = 9
OUT_START = 10

def generate_batch(batch_size, device):
    symbols = torch.randint(0, ALPHABET, (batch_size, N_SYMBOLS))
    x = torch.zeros(batch_size, SEQ_LEN, INPUT_DIM)
    for i in range(N_SYMBOLS):
        x[torch.arange(batch_size), i, symbols[:, i]] = 1.0
    x[:, 5:9, BLANK_CH] = 1.0
    x[:, 9, GO_CH] = 1.0
    x[:, 9, BLANK_CH] = 1.0
    x[:, 10:15, BLANK_CH] = 1.0
    return x.to(device), symbols.to(device)

class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.U = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        self.V = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        self.b_y = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch_size, T, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        output_logits = []
        for t in range(T):
            x_t = x[:, t, :]
            h = torch.tanh(h @ self.W.T + x_t @ self.U.T + self.b_h)
            if t >= OUT_START:
                output_logits.append(h @ self.V.T + self.b_y)
        return torch.stack(output_logits, dim=1)

@torch.no_grad()
def evaluate(model):
    model.eval()
    x, targets = generate_batch(EVAL_SIZE, device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, OUTPUT_DIM), targets.reshape(-1)).item()
    preds = logits.argmax(dim=-1)
    per_sym = (preds == targets).float().mean().item()
    seq_acc = (preds == targets).all(dim=1).float().mean().item()
    model.train()
    return loss, per_sym, seq_acc

print("="*70)
print("BPTT BASELINE: Vanilla Elman RNN on Copy Task")
print("="*70)
print(f"Architecture: input={INPUT_DIM}, hidden={HIDDEN_DIM}, output={OUTPUT_DIM}")
print(f"Training: Adam lr={LR}, batch={BATCH_SIZE}, steps={N_STEPS}")
print("="*70)

model = ElmanRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

results = []
step_times = []

loss0, sym0, seq0 = evaluate(model)
results.append({"step":0,"loss":loss0,"per_symbol_acc":sym0,"seq_acc":seq0})
print(f"Step    0 | Loss: {loss0:.4f} | Per-sym: {sym0:.4f} | Seq: {seq0:.4f}")

seg_start = time.time()
for step in range(1, N_STEPS+1):
    model.train()
    x, targets = generate_batch(BATCH_SIZE, device)
    optimizer.zero_grad()
    logits = model(x)
    loss = loss_fn(logits.reshape(-1, OUTPUT_DIM), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    if step % EVAL_EVERY == 0:
        elapsed = time.time() - seg_start
        step_times.append(elapsed)
        el, sym, seq = evaluate(model)
        results.append({"step":step,"loss":el,"per_symbol_acc":sym,"seq_acc":seq,"wall_time":elapsed})
        print(f"Step {step:4d} | Loss: {el:.4f} | Per-sym: {sym:.4f} | Seq: {seq:.4f} | Time: {elapsed:.2f}s")
        seg_start = time.time()

print("="*70)
final = results[-1]
print(f"FINAL: step={final[\"step\"]}, per_sym={final[\"per_symbol_acc\"]:.4f}, seq={final[\"seq_acc\"]:.4f}, loss={final[\"loss\"]:.4f}")
print(f"Avg time/100 steps: {np.mean(step_times):.2f}s")
print(f"Device: {device}")

landmark = None
for r in results:
    if r["per_symbol_acc"] > 0.90:
        landmark = r["step"]
        break
if landmark is not None:
    print(f"LANDMARK: per-symbol acc first >90% at step {landmark}")
else:
    print("LANDMARK: per-symbol acc did NOT exceed 90% within 3000 steps")

print(f"Success (>95% per-sym): {final[\"per_symbol_acc\"] > 0.95}")
print("="*70)
print("LEARNING CURVE:")
print("step | loss | per_sym_acc | seq_acc")
for r in results:
    print(f"{r[\"step\"]:4d} | {r[\"loss\"]:.4f} | {r[\"per_symbol_acc\"]:.4f} | {r[\"seq_acc\"]:.4f}")
'''
script_path = '/tmp/bptt_experiment.py'
with open(script_path, 'w') as f:
    f.write(script)
print(f"Script written to {script_path}")
print("Running experiment with panda conda Python (PyTorch 2.5.1+cu121)...")
print("This may take a few minutes...")
conda_python = '/home/zihan.zhang/.conda/envs/panda/bin/python'
result = subprocess.run(
    [conda_python, script_path],
    capture_output=True, text=True, timeout=600  # 10 min timeout
)
print("=== STDOUT ===")
print(result.stdout)
if result.stderr:
    print("=== STDERR (last 2000 chars) ===")
    print(result.stderr[-2000:])
print(f"Return code: {result.returncode}")

# ----------

# ----------
import subprocess
lines = [
    'import torch',
    'import torch.nn as nn',
    'import torch.optim as optim',
    'import numpy as np',
    'import time',
    '',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device: " + str(device))',
    'if torch.cuda.is_available():',
    '    print("GPU: " + torch.cuda.get_device_name(0))',
    '',
    'INPUT_DIM = 10',
    'HIDDEN_DIM = 128',
    'OUTPUT_DIM = 8',
    'ALPHABET = 8',
    'SEQ_LEN = 15',
    'N_SYMBOLS = 5',
    'BATCH_SIZE = 64',
    'LR = 1e-3',
    'N_STEPS = 3000',
    'EVAL_EVERY = 100',
    'EVAL_SIZE = 1000',
    'BLANK_CH = 8',
    'GO_CH = 9',
    'OUT_START = 10',
    '',
    'def generate_batch(batch_size, dev):',
    '    symbols = torch.randint(0, ALPHABET, (batch_size, N_SYMBOLS))',
    '    x = torch.zeros(batch_size, SEQ_LEN, INPUT_DIM)',
    '    for i in range(N_SYMBOLS):',
    '        x[torch.arange(batch_size), i, symbols[:, i]] = 1.0',
    '    x[:, 5:9, BLANK_CH] = 1.0',
    '    x[:, 9, GO_CH] = 1.0',
    '    x[:, 9, BLANK_CH] = 1.0',
    '    x[:, 10:15, BLANK_CH] = 1.0',
    '    return x.to(dev), symbols.to(dev)',
    '',
    'class ElmanRNN(nn.Module):',
    '    def __init__(self, input_dim, hidden_dim, output_dim):',
    '        super().__init__()',
    '        self.hidden_dim = hidden_dim',
    '        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)',
    '        self.U = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)',
    '        self.b_h = nn.Parameter(torch.zeros(hidden_dim))',
    '        self.V = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)',
    '        self.b_y = nn.Parameter(torch.zeros(output_dim))',
    '',
    '    def forward(self, x):',
    '        batch_size, T, _ = x.shape',
    '        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)',
    '        output_logits = []',
    '        for t in range(T):',
    '            x_t = x[:, t, :]',
    '            h = torch.tanh(h @ self.W.T + x_t @ self.U.T + self.b_h)',
    '            if t >= OUT_START:',
    '                output_logits.append(h @ self.V.T + self.b_y)',
    '        return torch.stack(output_logits, dim=1)',
    '',
    '@torch.no_grad()',
    'def evaluate(model):',
    '    model.eval()',
    '    x, targets = generate_batch(EVAL_SIZE, device)',
    '    logits = model(x)',
    '    loss = nn.functional.cross_entropy(logits.reshape(-1, OUTPUT_DIM), targets.reshape(-1)).item()',
    '    preds = logits.argmax(dim=-1)',
    '    per_sym = (preds == targets).float().mean().item()',
    '    seq_acc = (preds == targets).all(dim=1).float().mean().item()',
    '    model.train()',
    '    return loss, per_sym, seq_acc',
    '',
    'print("=" * 70)',
    'print("BPTT BASELINE: Vanilla Elman RNN on Copy Task")',
    'print("=" * 70)',
    'print("Architecture: input=" + str(INPUT_DIM) + ", hidden=" + str(HIDDEN_DIM) + ", output=" + str(OUTPUT_DIM))',
    'print("Training: Adam lr=" + str(LR) + ", batch=" + str(BATCH_SIZE) + ", steps=" + str(N_STEPS))',
    'print("=" * 70)',
    '',
    'model = ElmanRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)',
    'optimizer = optim.Adam(model.parameters(), lr=LR)',
    'loss_fn = nn.CrossEntropyLoss()',
    '',
    'results = []',
    'step_times = []',
    '',
    'loss0, sym0, seq0 = evaluate(model)',
    'results.append({"step": 0, "loss": loss0, "per_symbol_acc": sym0, "seq_acc": seq0})',
    'print("Step    0 | Loss: %.4f | Per-sym: %.4f | Seq: %.4f" % (loss0, sym0, seq0))',
    '',
    'seg_start = time.time()',
    'for step in range(1, N_STEPS + 1):',
    '    model.train()',
    '    x, targets = generate_batch(BATCH_SIZE, device)',
    '    optimizer.zero_grad()',
    '    logits = model(x)',
    '    loss = loss_fn(logits.reshape(-1, OUTPUT_DIM), targets.reshape(-1))',
    '    loss.backward()',
    '    optimizer.step()',
    '    if step % EVAL_EVERY == 0:',
    '        elapsed = time.time() - seg_start',
    '        step_times.append(elapsed)',
    '        el, sym, seq = evaluate(model)',
    '        results.append({"step": step, "loss": el, "per_symbol_acc": sym, "seq_acc": seq, "wall_time": elapsed})',
    '        print("Step %4d | Loss: %.4f | Per-sym: %.4f | Seq: %.4f | Time: %.2fs" % (step, el, sym, seq, elapsed))',
    '        seg_start = time.time()',
    '',
    'print("=" * 70)',
    'final = results[-1]',
    'fs = final["step"]',
    'fp = final["per_symbol_acc"]',
    'fq = final["seq_acc"]',
    'fl = final["loss"]',
    'print("FINAL: step=%d, per_sym=%.4f, seq=%.4f, loss=%.4f" % (fs, fp, fq, fl))',
    'print("Avg time/100 steps: %.2fs" % np.mean(step_times))',
    'print("Device: " + str(device))',
    '',
    'landmark = None',
    'for r in results:',
    '    if r["per_symbol_acc"] > 0.90:',
    '        landmark = r["step"]',
    '        break',
    'if landmark is not None:',
    '    print("LANDMARK: per-symbol acc first >90%% at step %d" % landmark)',
    'else:',
    '    print("LANDMARK: per-symbol acc did NOT exceed 90%% within 3000 steps")',
    '',
    'print("Success (>95%% per-sym): " + str(final["per_symbol_acc"] > 0.95))',
    'print("=" * 70)',
    'print("LEARNING CURVE:")',
    'print("step | loss | per_sym_acc | seq_acc")',
    'for r in results:',
    '    rs = r["step"]',
    '    rl = r["loss"]',
    '    rp = r["per_symbol_acc"]',
    '    rq = r["seq_acc"]',
    '    print("%4d | %.4f | %.4f | %.4f" % (rs, rl, rp, rq))',
]
script_path = '/tmp/bptt_experiment2.py'
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
print(f"Script written to {script_path}")
print("Running experiment...")
conda_python = '/home/zihan.zhang/.conda/envs/panda/bin/python'
result = subprocess.run(
    [conda_python, script_path],
    capture_output=True, text=True, timeout=600
)
print("=== STDOUT ===")
print(result.stdout)
if result.stderr:
    print("=== STDERR (last 2000 chars) ===")
    print(result.stderr[-2000:])
print(f"Return code: {result.returncode}")

# ----------

# ----------
import numpy as np
learning_curve = [
    (0,    2.1644, 0.1118, 0.0000),
    (100,  0.1776, 0.9972, 0.9860),
    (200,  0.0417, 1.0000, 1.0000),
    (300,  0.0181, 1.0000, 1.0000),
    (400,  0.0099, 1.0000, 1.0000),
    (500,  0.0062, 1.0000, 1.0000),
    (600,  0.0045, 1.0000, 1.0000),
    (700,  0.0032, 1.0000, 1.0000),
    (800,  0.0025, 1.0000, 1.0000),
    (900,  0.0019, 1.0000, 1.0000),
    (1000, 0.0016, 1.0000, 1.0000),
    (1100, 0.0013, 1.0000, 1.0000),
    (1200, 0.0011, 1.0000, 1.0000),
    (1300, 0.0009, 1.0000, 1.0000),
    (1400, 0.0008, 1.0000, 1.0000),
    (1500, 0.0007, 1.0000, 1.0000),
    (1600, 0.0006, 1.0000, 1.0000),
    (1700, 0.0005, 1.0000, 1.0000),
    (1800, 0.0005, 1.0000, 1.0000),
    (1900, 0.0004, 1.0000, 1.0000),
    (2000, 0.0004, 1.0000, 1.0000),
    (2100, 0.0004, 1.0000, 1.0000),
    (2200, 0.0003, 1.0000, 1.0000),
    (2300, 0.0003, 1.0000, 1.0000),
    (2400, 0.0003, 1.0000, 1.0000),
    (2500, 0.0002, 1.0000, 1.0000),
    (2600, 0.0002, 1.0000, 1.0000),
    (2700, 0.0002, 1.0000, 1.0000),
    (2800, 0.0002, 1.0000, 1.0000),
    (2900, 0.0002, 1.0000, 1.0000),
    (3000, 0.0002, 1.0000, 1.0000),
]
steps      = [r[0] for r in learning_curve]
losses     = [r[1] for r in learning_curve]
per_sym    = [r[2] for r in learning_curve]
seq_accs   = [r[3] for r in learning_curve]
final_step     = steps[-1]
final_loss     = losses[-1]
final_per_sym  = per_sym[-1]
final_seq_acc  = seq_accs[-1]
landmark_step = None
landmark_per_sym = None
landmark_seq_acc = None
for step, loss, ps, sa in learning_curve:
    if ps > 0.90:
        landmark_step = step
        landmark_per_sym = ps
        landmark_seq_acc = sa
        break
full_per_sym_step = None
for step, loss, ps, sa in learning_curve:
    if ps >= 1.0000:
        full_per_sym_step = step
        break
full_seq_step = None
for step, loss, ps, sa in learning_curve:
    if sa >= 1.0000:
        full_seq_step = step
        break
print("=" * 70)
print("BPTT BASELINE ANALYSIS SUMMARY")
print("=" * 70)
print()
print("EXPERIMENT CONFIGURATION:")
print("  Model      : Vanilla Elman RNN (explicit dynamics)")
print("  Hidden dim : 128")
print("  Input dim  : 10 (8 symbols + blank + go)")
print("  Output dim : 8 (alphabet size)")
print("  Optimizer  : Adam, lr=0.001")
print("  Batch size : 64")
print("  Eval size  : 1000 held-out sequences")
print("  Task       : Copy 5 symbols from alphabet-8, T=15")
print("  Device     : NVIDIA A100-PCIE-40GB (CUDA)")
print()
print("FINAL RESULTS (step %d):" % final_step)
print("  Per-symbol accuracy : %.4f (%.2f%%)" % (final_per_sym, final_per_sym * 100))
print("  Sequence accuracy   : %.4f (%.2f%%)" % (final_seq_acc, final_seq_acc * 100))
print("  Cross-entropy loss  : %.6f" % final_loss)
print("  Wall-clock per 100  : ~0.82s (on A100)")
print()
print("LEARNING SPEED:")
print("  Landmark (>90%% per-sym) : step %d (per-sym=%.4f, seq=%.4f)" % (landmark_step, landmark_per_sym, landmark_seq_acc))
print("  100%% per-sym first at  : step %d" % full_per_sym_step)
print("  100%% seq-acc first at  : step %d" % full_seq_step)
print()
print("SUCCESS CRITERIA:")
print("  >95%% per-symbol accuracy within 3000 steps: %s" % ("MET" if final_per_sym > 0.95 else "NOT MET"))
print("  >95%% sequence accuracy within 3000 steps  : %s" % ("MET" if final_seq_acc > 0.95 else "NOT MET"))
print()
print("CONVERGENCE ANALYSIS:")
print("  Initial loss    : %.4f (random baseline ~2.08)" % losses[0])
print("  Loss at step 100: %.4f" % losses[1])
print("  Loss at step 200: %.4f" % losses[2])
print("  Final loss      : %.6f" % final_loss)
print("  Loss reduction  : %.1fx" % (losses[0] / final_loss))
print()
print("REFERENCE TARGETS FOR FUTURE EXPERIMENTS (RFLO, e-prop, STC-Credit):")
print("  - Target: >95%% per-symbol accuracy (BPTT achieves 100%%)")
print("  - Target: >95%% sequence accuracy (BPTT achieves 100%%)")
print("  - Landmark step to beat: %d (step at which BPTT first exceeds 90%%)" % landmark_step)
print("  - BPTT achieves 100%% by step %d; future rules should reach 90%% by step ~%d" % (full_per_sym_step, landmark_step))
print()
print("CONCLUSION:")
print("  BPTT on this Elman RNN / copy-task configuration is an extremely strong")
print("  baseline. It reaches perfect (100%%) per-symbol and sequence accuracy")
print("  within 200 training steps, and 99.72%% per-symbol accuracy by step 100.")
print("  The loss continues to decrease monotonically from 2.16 to 0.0002 over")
print("  3000 steps. This sets a clear upper-bound reference for biologically-")
print("  plausible learning rules: they should aim for >95%% per-symbol accuracy")
print("  on the same task, with a landmark of step 100 being the target for")
print("  crossing the 90%% threshold.")
print("=" * 70)

# ----------

# ----------
write_report()

# ----------

# ----------
print("BPTT BASELINE EXPERIMENT - COMPLETE RESULTS SUMMARY")
print("=" * 70)
print()
print("RESEARCH QUESTION:")
print("Train a vanilla Elman RNN with BPTT on a copy task (5 symbols, alphabet-8,")
print("T=15). Does it achieve >95% per-symbol accuracy within 3000 steps?")
print("At what step does it first exceed 90% per-symbol accuracy?")
print()
print("ARCHITECTURE:")
print("  Type       : Vanilla Elman RNN (explicit dynamics, NO nn.RNN module)")
print("  Dynamics   : h_t = tanh(W @ h_{t-1} + U @ x_t + b_h)")
print("  Readout    : y_t = V @ h_t + b_y")
print("  Input dim  : 10 (8 symbols + 1 blank + 1 go/delimiter)")
print("  Hidden dim : 128")
print("  Output dim : 8 (alphabet)")
print("  Activation : tanh")
print()
print("TASK (Copy Task):")
print("  t=1..5    : present 5 symbols (one-hot, channels 0-7)")
print("  t=6..9    : blank input (channel 8)")
print("  t=10      : go signal (channel 9) + blank")
print("  t=11..15  : blank input; network must reproduce the 5 symbols")
print("  Loss      : cross-entropy only at output timesteps t=11..15")
print()
print("TRAINING CONFIG:")
print("  Optimizer  : Adam, lr=0.001")
print("  Batch size : 64")
print("  Steps      : 3000")
print("  Eval freq  : every 100 steps on 1000 held-out sequences")
print("  Device     : NVIDIA A100-PCIE-40GB (CUDA 12.1)")
print()
print("RESULTS:")
print("  Step    0: loss=2.1644, per-sym=11.18%, seq=0.00%  [random init]")
print("  Step  100: loss=0.1776, per-sym=99.72%, seq=98.60% [LANDMARK: >90%]")
print("  Step  200: loss=0.0417, per-sym=100.0%, seq=100.0% [PERFECT]")
print("  Step 3000: loss=0.0002, per-sym=100.0%, seq=100.0% [FINAL]")
print()
print("KEY METRICS:")
print("  Final per-symbol accuracy : 100.00%")
print("  Final sequence accuracy   : 100.00%")
print("  Final cross-entropy loss  : 0.000200")
print("  Wall-clock per 100 steps  : 0.82s (A100 GPU)")
print("  Total training time       : ~24.6s (30 x 0.82s)")
print()
print("LANDMARK RESULTS:")
print("  Per-symbol acc first >90% : step 100 (99.72%)")
print("  Per-symbol acc = 100%     : step 200")
print("  Sequence acc = 100%       : step 200")
print()
print("FULL LEARNING CURVE (step | loss | per-sym-acc | seq-acc):")
curve = [
    (0,2.1644,0.1118,0.0),(100,0.1776,0.9972,0.986),(200,0.0417,1.0,1.0),
    (300,0.0181,1.0,1.0),(400,0.0099,1.0,1.0),(500,0.0062,1.0,1.0),
    (600,0.0045,1.0,1.0),(700,0.0032,1.0,1.0),(800,0.0025,1.0,1.0),
    (900,0.0019,1.0,1.0),(1000,0.0016,1.0,1.0),(1100,0.0013,1.0,1.0),
    (1200,0.0011,1.0,1.0),(1300,0.0009,1.0,1.0),(1400,0.0008,1.0,1.0),
    (1500,0.0007,1.0,1.0),(1600,0.0006,1.0,1.0),(1700,0.0005,1.0,1.0),
    (1800,0.0005,1.0,1.0),(1900,0.0004,1.0,1.0),(2000,0.0004,1.0,1.0),
    (2100,0.0004,1.0,1.0),(2200,0.0003,1.0,1.0),(2300,0.0003,1.0,1.0),
    (2400,0.0003,1.0,1.0),(2500,0.0002,1.0,1.0),(2600,0.0002,1.0,1.0),
    (2700,0.0002,1.0,1.0),(2800,0.0002,1.0,1.0),(2900,0.0002,1.0,1.0),
    (3000,0.0002,1.0,1.0),
]
for s,l,p,q in curve:
    print("  %4d | %7.4f | %6.4f | %6.4f" % (s,l,p,q))
print()
print("SUCCESS CRITERIA:")
print("  (a) BPTT reaches >95% per-symbol accuracy within 3000 steps: YES (100%)")
print("  (b) Landmark step where per-sym acc first exceeds 90%: step 100")
print()
print("CONCLUSIONS FOR DOWNSTREAM EXPERIMENTS (RFLO, e-prop, STC-Credit):")
print("  1. BPTT upper bound: 100% per-symbol, 100% sequence accuracy")
print("  2. Target for bio-plausible rules: >95% per-symbol accuracy")
print("  3. Speed target: reach 90% per-symbol by step 100 (as BPTT does)")
print("  4. BPTT converges extremely fast (200 steps to perfection), so")
print("     bio-plausible rules that reach >95% within 3000 steps")
print("     qualify as 'within reasonable range of BPTT'.")
print("  5. The loss floor is near zero (0.0002), indicating the architecture")
print("     has sufficient capacity for the task.")
print("=" * 70)
result = write_report()
print("Report written:", result)

# ----------
