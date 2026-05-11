
# ----------
# The below command failed to execute (raised a No module named 'torch' exception)
# import torch

# ----------

# ----------
import subprocess
import sys
result = subprocess.run(['conda', 'run', '-n', 'panda', 'python', '-c', 
                        'import sys; print(sys.executable); import torch; print("torch version:", torch.__version__)'],
                       capture_output=True, text=True)
print('stdout:', result.stdout)
print('stderr:', result.stderr[:500] if result.stderr else 'None')
print('returncode:', result.returncode)

# ----------

# ----------
import subprocess
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
setup_code = '''
import torch
import torch.nn as nn
import numpy as np
import json
import math
import os
import time
from itertools import product

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Working directory
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
os.makedirs(WORK_DIR, exist_ok=True)
print(f'Working directory: {WORK_DIR}')

# ============================================================
# TASK CONFIGURATION
# ============================================================
N_BITS = 8
DELAY = 10
INPUT_DIM = N_BITS + 1  # +1 for recall cue marker
OUTPUT_DIM = N_BITS
HIDDEN_SIZE = 128
SEQ_LEN = N_BITS + DELAY + N_BITS  # = 26

print(f"\\nTask Configuration:")
print(f"  N_BITS={N_BITS}, DELAY={DELAY}, SEQ_LEN={SEQ_LEN}")
print(f"  INPUT_DIM={INPUT_DIM}, OUTPUT_DIM={OUTPUT_DIM}, HIDDEN_SIZE={HIDDEN_SIZE}")

# ============================================================
# COPY TASK DATA GENERATOR
# ============================================================
def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, device=device):
    """
    Generate a batch of copy task sequences.
    Sequence structure:
      t=0..n_bits-1:           present input bits (marker=0)
      t=n_bits..n_bits+delay-2: blank steps (all zeros)
      t=n_bits+delay-1:        recall cue (marker bit=1)
      t=n_bits+delay..seq_len-1: output phase (target = original bits)
    """
    seq_len = n_bits + delay + n_bits
    input_dim = n_bits + 1
    
    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)
    
    inputs = torch.zeros(batch_size, seq_len, input_dim)
    targets = torch.zeros(batch_size, seq_len, n_bits)
    output_mask = torch.zeros(batch_size, seq_len)
    
    # Present input bits
    inputs[:, :n_bits, :n_bits] = patterns
    
    # Recall cue marker
    inputs[:, n_bits + delay - 1, n_bits] = 1.0
    
    # Output phase targets
    output_start = n_bits + delay
    targets[:, output_start:, :] = patterns.unsqueeze(1).expand(-1, n_bits, -1)
    output_mask[:, output_start:] = 1.0
    
    return inputs.to(device), targets.to(device), output_mask.to(device)

# Test the data generator
batch_size = 4
torch.manual_seed(SEED)
inputs, targets, output_mask = generate_copy_batch(batch_size)

print(f"\\nBatch shapes:")
print(f"  inputs: {inputs.shape}")
print(f"  targets: {targets.shape}")
print(f"  output_mask: {output_mask.shape}")

print(f"\\nSample sequence (batch item 0):")
print(f"  Input bits (t=0..7):    {inputs[0, :N_BITS, :N_BITS].cpu().numpy()}")
print(f"  Input marker (t=17):    {inputs[0, N_BITS+DELAY-1, N_BITS].item()}")
print(f"  Target bits (t=18..25): {targets[0, N_BITS+DELAY:, :].cpu().numpy()}")
print(f"  Output mask:            {output_mask[0].cpu().numpy()}")

# Verify pattern consistency
for i in range(batch_size):
    pat_in = inputs[i, :N_BITS, :N_BITS].cpu().numpy()
    pat_out = targets[i, N_BITS+DELAY:, :].cpu().numpy()
    assert np.allclose(pat_in, pat_out), f"Mismatch at batch item {i}!"
print(f"\\n[PASS] Input/target consistency check passed for all {batch_size} items")

# Test multiple batches
for trial in range(3):
    inp, tgt, msk = generate_copy_batch(32)
    assert inp.shape == (32, SEQ_LEN, INPUT_DIM)
    assert tgt.shape == (32, SEQ_LEN, OUTPUT_DIM)
    assert msk.shape == (32, SEQ_LEN)
    assert msk[:, :N_BITS+DELAY].sum() == 0  # no targets during input+delay
    assert msk[:, N_BITS+DELAY:].sum() == 32 * N_BITS  # all output steps have targets
print(f"[PASS] Batch generation tests passed")

print(f"\\n=== Step 1: Environment Setup COMPLETE ===")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  Device: {device}")
'''
setup_file = WORK_DIR + 'step1_setup.py'
with open(setup_file, 'w') as f:
    f.write(setup_code)
result = subprocess.run([PYTHON, setup_file], capture_output=True, text=True, timeout=60)
print('STDOUT:')
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:1000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
setup_code = '''
import torch
import torch.nn as nn
import numpy as np
import json
import math
import os
import time
from itertools import product

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
print(f\'Using device: {device}\')
if torch.cuda.is_available():
    print(f\'GPU: {torch.cuda.get_device_name(0)}\')

# Working directory
WORK_DIR = \'/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/\'
os.makedirs(WORK_DIR, exist_ok=True)
print(f\'Working directory: {WORK_DIR}\')

# ============================================================
# TASK CONFIGURATION
# ============================================================
N_BITS = 8
DELAY = 10
# Copy task: present all 8 bits at t=0, blank for DELAY steps, recall cue at t=N_BITS+DELAY-1,
# then reproduce 8 bits at t=N_BITS+DELAY..N_BITS+DELAY+N_BITS-1
# Total sequence length = 1 (input) + (DELAY-1) (blank) + 1 (cue) + N_BITS (output) = N_BITS + DELAY + 1
# Actually let\'s use: N_BITS input timesteps (one bit per step) + DELAY blank + N_BITS output
# Each input timestep t presents bit t of the pattern
# This is the standard sequential copy task

INPUT_DIM = N_BITS + 1  # N_BITS channels + 1 recall cue channel
OUTPUT_DIM = N_BITS
HIDDEN_SIZE = 128
SEQ_LEN = N_BITS + DELAY + N_BITS  # 8 + 10 + 8 = 26

print(f"\nTask Configuration:")
print(f"  N_BITS={N_BITS}, DELAY={DELAY}, SEQ_LEN={SEQ_LEN}")
print(f"  INPUT_DIM={INPUT_DIM}, OUTPUT_DIM={OUTPUT_DIM}, HIDDEN_SIZE={HIDDEN_SIZE}")
print(f"  Sequence: [8 input steps] + [10 blank/delay steps] + [8 output steps]")
print(f"  At each input step t: present bit t of pattern in channel t")
print(f"  At t=17 (last delay step): recall cue marker=1")
print(f"  At t=18..25: network must reproduce the pattern")

# ============================================================
# COPY TASK DATA GENERATOR
# ============================================================
def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, device=device):
    """
    Sequential copy task:
    - t=0..n_bits-1: input phase, at step t, bit t is presented in channel t
      (so at t=0: channel 0 = bit0, others 0; at t=1: channel 1 = bit1, etc.)
    - t=n_bits..n_bits+delay-2: blank (all zeros)
    - t=n_bits+delay-1: recall cue (marker channel = 1)
    - t=n_bits+delay..seq_len-1: output phase (target = original bits)
    
    Returns:
        inputs:      (batch_size, seq_len, input_dim)
        targets:     (batch_size, seq_len, n_bits)  -- nonzero only at output steps
        output_mask: (batch_size, seq_len)           -- 1 at output steps
    """
    seq_len = n_bits + delay + n_bits
    input_dim = n_bits + 1  # n_bits channels + 1 marker
    
    # Random binary patterns
    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)
    
    inputs = torch.zeros(batch_size, seq_len, input_dim)
    targets = torch.zeros(batch_size, seq_len, n_bits)
    output_mask = torch.zeros(batch_size, seq_len)
    
    # Input phase: at timestep t, present bit t in channel t
    for t in range(n_bits):
        inputs[:, t, t] = patterns[:, t]  # shape: (batch_size,)
    
    # Recall cue at t = n_bits + delay - 1
    inputs[:, n_bits + delay - 1, n_bits] = 1.0
    
    # Output phase
    output_start = n_bits + delay
    for t in range(n_bits):
        targets[:, output_start + t, :] = patterns  # all bits shown simultaneously in output
    output_mask[:, output_start:output_start + n_bits] = 1.0
    
    return inputs.to(device), targets.to(device), output_mask.to(device)

# Test the data generator
batch_size = 4
torch.manual_seed(SEED)
inputs, targets, output_mask = generate_copy_batch(batch_size)

print(f"\nBatch shapes:")
print(f"  inputs: {inputs.shape}")
print(f"  targets: {targets.shape}")
print(f"  output_mask: {output_mask.shape}")

print(f"\nSample sequence (batch item 0):")
print(f"  Pattern to copy: {targets[0, N_BITS+DELAY, :].cpu().numpy()}")
print(f"  Input at t=0 (ch0-7): {inputs[0, 0, :N_BITS].cpu().numpy()}")
print(f"  Input at t=1 (ch0-7): {inputs[0, 1, :N_BITS].cpu().numpy()}")
print(f"  Input at t=7 (ch0-7): {inputs[0, 7, :N_BITS].cpu().numpy()}")
print(f"  Recall marker at t=17: {inputs[0, N_BITS+DELAY-1, N_BITS].item()}")
print(f"  Target at t=18: {targets[0, N_BITS+DELAY, :].cpu().numpy()}")
print(f"  Target at t=25: {targets[0, N_BITS+DELAY+N_BITS-1, :].cpu().numpy()}")
print(f"  Output mask: {output_mask[0].cpu().numpy()}")

# Verify correctness
for i in range(batch_size):
    # Extract original pattern from input
    pat_from_input = torch.zeros(N_BITS)
    for t in range(N_BITS):
        pat_from_input[t] = inputs[i, t, t].cpu()
    # Check targets match
    for t in range(N_BITS):
        tgt_t = targets[i, N_BITS+DELAY+t, :].cpu()
        assert torch.allclose(tgt_t, pat_from_input), f"Mismatch at batch {i}, output step {t}!"
print(f"\n[PASS] Input/target consistency check passed for all {batch_size} items")

# Test batch dimensions
for trial in range(3):
    inp, tgt, msk = generate_copy_batch(32)
    assert inp.shape == (32, SEQ_LEN, INPUT_DIM), f"Wrong input shape: {inp.shape}"
    assert tgt.shape == (32, SEQ_LEN, OUTPUT_DIM), f"Wrong target shape: {tgt.shape}"
    assert msk.shape == (32, SEQ_LEN), f"Wrong mask shape: {msk.shape}"
    # Only output phase has nonzero mask
    assert msk[:, :N_BITS+DELAY].sum().item() == 0, "Non-zero mask in input/delay phase!"
    expected_ones = 32 * N_BITS
    actual_ones = msk[:, N_BITS+DELAY:].sum().item()
    assert actual_ones == expected_ones, f"Expected {expected_ones} mask=1, got {actual_ones}"
print(f"[PASS] All batch shape/mask tests passed")

print(f"\n=== Step 1: Environment Setup COMPLETE ===")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}, Device: {device}")
'''
setup_file = WORK_DIR + 'step1_setup.py'
with open(setup_file, 'w') as f:
    f.write(setup_code)
result = subprocess.run([PYTHON, setup_file], capture_output=True, text=True, timeout=60)
print('STDOUT:')
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
os.makedirs(WORK_DIR, exist_ok=True)
script_path = os.path.join(WORK_DIR, 'step1_setup.py')
lines = [
    'import torch',
    'import torch.nn as nn',
    'import numpy as np',
    'import json',
    'import math',
    'import os',
    'import time',
    'from itertools import product',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    '',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Using device:", device)',
    'if torch.cuda.is_available():',
    '    print("GPU:", torch.cuda.get_device_name(0))',
    '',
    'WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"',
    'os.makedirs(WORK_DIR, exist_ok=True)',
    'print("Working directory:", WORK_DIR)',
    '',
    '# Task configuration',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    '',
    'print("")',
    'print("Task Configuration:")',
    'print("  N_BITS=", N_BITS, " DELAY=", DELAY, " SEQ_LEN=", SEQ_LEN)',
    'print("  INPUT_DIM=", INPUT_DIM, " OUTPUT_DIM=", OUTPUT_DIM, " HIDDEN_SIZE=", HIDDEN_SIZE)',
    '',
    'def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):',
    '    if dev is None:',
    '        dev = device',
    '    seq_len = n_bits + delay + n_bits',
    '    input_dim = n_bits + 1',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, input_dim)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    '# Test the data generator',
    'batch_size = 4',
    'torch.manual_seed(SEED)',
    'inputs, targets, output_mask = generate_copy_batch(batch_size)',
    '',
    'print("")',
    'print("Batch shapes:")',
    'print("  inputs:", inputs.shape)',
    'print("  targets:", targets.shape)',
    'print("  output_mask:", output_mask.shape)',
    '',
    'print("")',
    'print("Sample sequence (batch item 0):")',
    'print("  Pattern to copy:", targets[0, N_BITS+DELAY, :].cpu().numpy())',
    'print("  Input at t=0:", inputs[0, 0, :N_BITS].cpu().numpy())',
    'print("  Input at t=3:", inputs[0, 3, :N_BITS].cpu().numpy())',
    'print("  Input at t=7:", inputs[0, 7, :N_BITS].cpu().numpy())',
    'print("  Recall cue at t=17:", inputs[0, N_BITS+DELAY-1, N_BITS].item())',
    'print("  Target at t=18:", targets[0, N_BITS+DELAY, :].cpu().numpy())',
    'print("  Output mask:", output_mask[0].cpu().numpy())',
    '',
    '# Verify correctness',
    'for i in range(batch_size):',
    '    pat_from_input = torch.zeros(N_BITS)',
    '    for t in range(N_BITS):',
    '        pat_from_input[t] = inputs[i, t, t].cpu()',
    '    for t in range(N_BITS):',
    '        tgt_t = targets[i, N_BITS+DELAY+t, :].cpu()',
    '        assert torch.allclose(tgt_t, pat_from_input), "Mismatch at batch " + str(i)',
    'print("[PASS] Input/target consistency check passed")',
    '',
    '# Test batch dimensions',
    'for trial in range(3):',
    '    inp, tgt, msk = generate_copy_batch(32)',
    '    assert inp.shape == (32, SEQ_LEN, INPUT_DIM)',
    '    assert tgt.shape == (32, SEQ_LEN, OUTPUT_DIM)',
    '    assert msk.shape == (32, SEQ_LEN)',
    '    assert msk[:, :N_BITS+DELAY].sum().item() == 0',
    '    assert msk[:, N_BITS+DELAY:].sum().item() == 32 * N_BITS',
    'print("[PASS] Batch shape/mask tests passed")',
    '',
    'print("")',
    'print("=== Step 1: Environment Setup COMPLETE ===")',
    'print("  PyTorch:", torch.__version__)',
    'print("  CUDA:", torch.cuda.is_available(), "Device:", device)',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
print(f'Script written to {script_path}')
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=60)
print('STDOUT:')
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step2_bptt.py')
lines = [
    'import torch',
    'import torch.nn as nn',
    'import numpy as np',
    'import json',
    'import os',
    'import math',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    '',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    '',
    '# ============================================================',
    '# COPY TASK DATA GENERATOR',
    '# ============================================================',
    'def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):',
    '    if dev is None:',
    '        dev = device',
    '    seq_len = n_bits + delay + n_bits',
    '    input_dim = n_bits + 1',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, input_dim)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    '# ============================================================',
    '# BPTT RNN MODEL',
    '# ============================================================',
    'class VanillaRNN(nn.Module):',
    '    def __init__(self, input_dim, hidden_size, output_dim):',
    '        super(VanillaRNN, self).__init__()',
    '        self.hidden_size = hidden_size',
    '        self.W_in = nn.Parameter(torch.randn(hidden_size, input_dim) * 0.1)',
    '        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)',
    '        self.b_rec = nn.Parameter(torch.zeros(hidden_size))',
    '        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_size) * 0.1)',
    '        self.b_out = nn.Parameter(torch.zeros(output_dim))',
    '',
    '    def forward(self, inputs, h0=None):',
    '        batch_size, seq_len, _ = inputs.shape',
    '        if h0 is None:',
    '            h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)',
    '        else:',
    '            h = h0',
    '        outputs = []',
    '        for t in range(seq_len):',
    '            x_t = inputs[:, t, :]',
    '            a_t = h @ self.W_rec.T + x_t @ self.W_in.T + self.b_rec',
    '            h = torch.tanh(a_t)',
    '            y_t = h @ self.W_out.T + self.b_out',
    '            outputs.append(y_t)',
    '        outputs = torch.stack(outputs, dim=1)',
    '        return outputs, h',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    pred = outputs * mask',
    '    tgt = targets * mask',
    '    mse = ((pred - tgt) ** 2).sum() / (mask.sum() * targets.shape[-1])',
    '    pred_bits = (outputs > 0.5).float() * output_mask.unsqueeze(-1)',
    '    tgt_bits = targets * output_mask.unsqueeze(-1)',
    '    correct = (pred_bits == tgt_bits).float() * output_mask.unsqueeze(-1)',
    '    n_total = output_mask.sum() * targets.shape[-1]',
    '    acc = correct.sum() / n_total if n_total > 0 else torch.tensor(0.0)',
    '    return mse.item(), acc.item()',
    '',
    '# ============================================================',
    '# BPTT TRAINING FUNCTION',
    '# ============================================================',
    'def train_bptt(n_iter=200, batch_size=32, lr=0.001, verbose=True, log_interval=50):',
    '    torch.manual_seed(SEED)',
    '    model = VanillaRNN(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM).to(device)',
    '    optimizer = torch.optim.Adam(model.parameters(), lr=lr)',
    '    losses = []',
    '    mse_history = []',
    '    acc_history = []',
    '    for it in range(n_iter):',
    '        inputs, targets, output_mask = generate_copy_batch(batch_size)',
    '        optimizer.zero_grad()',
    '        outputs, _ = model(inputs)',
    '        mask = output_mask.unsqueeze(-1)',
    '        loss = ((outputs - targets) ** 2 * mask).sum() / (mask.sum() * OUTPUT_DIM)',
    '        loss.backward()',
    '        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)',
    '        optimizer.step()',
    '        mse, acc = compute_metrics(outputs.detach(), targets, output_mask)',
    '        losses.append(loss.item())',
    '        mse_history.append(mse)',
    '        acc_history.append(acc)',
    '        if verbose and (it + 1) % log_interval == 0:',
    '            print("  Iter", it+1, "| Loss:", round(loss.item(), 5),',
    '                  "| MSE:", round(mse, 5), "| Acc:", round(acc, 4))',
    '    return model, losses, mse_history, acc_history',
    '',
    '# ============================================================',
    '# QUICK TEST: 200 iterations to verify BPTT works',
    '# ============================================================',
    'print("")',
    'print("=== Testing BPTT (200 iterations) ===")',
    'model, losses, mse_history, acc_history = train_bptt(n_iter=200, lr=0.001, log_interval=50)',
    'print("")',
    'print("BPTT Quick Test Results:")',
    'print("  Initial MSE:", round(mse_history[0], 5))',
    'print("  Final MSE:", round(mse_history[-1], 5))',
    'print("  Initial Acc:", round(acc_history[0], 4))',
    'print("  Final Acc:", round(acc_history[-1], 4))',
    'print("  Loss improved:", mse_history[-1] < mse_history[0])',
    '',
    '# Check model parameters exist',
    'print("")',
    'print("Model parameters:")',
    'for name, param in model.named_parameters():',
    '    print("  ", name, param.shape)',
    '',
    '# Run longer test to see convergence',
    'print("")',
    'print("=== Testing BPTT (500 iterations, lr=0.005) ===")',
    'model2, losses2, mse2, acc2 = train_bptt(n_iter=500, lr=0.005, log_interval=100)',
    'print("")',
    'print("BPTT 500-iter Results:")',
    'print("  Initial MSE:", round(mse2[0], 5))',
    'print("  Final MSE:", round(mse2[-1], 5))',
    'print("  Final Acc:", round(acc2[-1], 4))',
    '',
    'print("")',
    'print("=== Step 2: BPTT Implementation VERIFIED ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=120)
print('STDOUT:')
print(result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step3_fa.py')
lines = [
    'import torch',
    'import torch.nn as nn',
    'import numpy as np',
    'import math',
    'import os',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    '',
    'def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):',
    '    if dev is None: dev = device',
    '    seq_len = n_bits + delay + n_bits',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, n_bits+1)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*targets.shape[-1])',
    '    pred_bits = (outputs > 0.5).float()',
    '    correct = ((pred_bits == targets).float() * mask).sum()',
    '    return mse.item(), (correct / (mask.sum()*targets.shape[-1])).item()',
    '',
    '# ============================================================',
    '# FA TRAINING: manually implement forward + FA backward',
    '# ============================================================',
    '# FA uses fixed random feedback matrices B_rec (hidden->hidden) and B_out (out->hidden)',
    '# instead of W_rec^T and W_out^T during backward pass',
    '',
    'def train_fa(n_iter=200, batch_size=32, lr=0.001, verbose=True, log_interval=50):',
    '    torch.manual_seed(SEED)',
    '    # Weight matrices',
    '    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device).requires_grad_(False)',
    '    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device).requires_grad_(False)',
    '    b_rec = torch.zeros(HIDDEN_SIZE).to(device).requires_grad_(False)',
    '    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device).requires_grad_(False)',
    '    b_out = torch.zeros(OUTPUT_DIM).to(device).requires_grad_(False)',
    '    # Fixed random feedback matrices (FA key component)',
    '    B_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)',
    '    B_out = (torch.randn(HIDDEN_SIZE, OUTPUT_DIM) * 0.1).to(device)',
    '    # Adam state',
    '    params = [W_in, W_rec, b_rec, W_out, b_out]',
    '    m = [torch.zeros_like(p) for p in params]',
    '    v = [torch.zeros_like(p) for p in params]',
    '    beta1, beta2, eps = 0.9, 0.999, 1e-8',
    '    mse_history, acc_history = [], []',
    '    for it in range(n_iter):',
    '        inputs, targets, output_mask = generate_copy_batch(batch_size)',
    '        # Forward pass - store activations',
    '        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        hs = []  # hidden states',
    '        as_ = []  # pre-activation',
    '        outputs = []',
    '        for t in range(SEQ_LEN):',
    '            x_t = inputs[:, t, :]',
    '            a_t = h @ W_rec.T + x_t @ W_in.T + b_rec',
    '            h = torch.tanh(a_t)',
    '            y_t = h @ W_out.T + b_out',
    '            hs.append(h)',
    '            as_.append(a_t)',
    '            outputs.append(y_t)',
    '        outputs_t = torch.stack(outputs, dim=1)',
    '        # Compute output errors',
    '        out_err = (outputs_t - targets) * output_mask.unsqueeze(-1)',
    '        # Backward pass using FA feedback matrices',
    '        # Accumulate gradients',
    '        dW_in  = torch.zeros_like(W_in)',
    '        dW_rec = torch.zeros_like(W_rec)',
    '        db_rec = torch.zeros_like(b_rec)',
    '        dW_out = torch.zeros_like(W_out)',
    '        db_out = torch.zeros_like(b_out)',
    '        delta_h_next = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        for t in reversed(range(SEQ_LEN)):',
    '            h_t = hs[t]',
    '            h_prev = hs[t-1] if t > 0 else torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '            x_t = inputs[:, t, :]',
    '            dtanh = 1.0 - h_t**2',
    '            # Output error projected back via B_out (FA) instead of W_out^T',
    '            e_out_t = out_err[:, t, :]  # (batch, output_dim)',
    '            delta_out = e_out_t @ B_out.T  # (batch, hidden) - FA feedback',
    '            # Total delta_h: from output error + recurrent from next step via B_rec',
    '            delta_h = (delta_out + delta_h_next) * dtanh',
    '            dW_out += e_out_t.T @ h_t / batch_size',
    '            db_out += e_out_t.mean(0)',
    '            dW_rec += delta_h.T @ h_prev / batch_size',
    '            dW_in  += delta_h.T @ x_t / batch_size',
    '            db_rec += delta_h.mean(0)',
    '            # Propagate delta_h to next step via B_rec (FA)',
    '            delta_h_next = delta_h @ B_rec.T',
    '        # Clip gradients',
    '        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]',
    '        total_norm = sum(g.norm()**2 for g in grads)**0.5',
    '        clip = 1.0 / (total_norm + 1e-8)',
    '        if total_norm > 1.0:',
    '            grads = [g * clip for g in grads]',
    '        # Adam update',
    '        t_adam = it + 1',
    '        for i, (p, g) in enumerate(zip(params, grads)):',
    '            m[i] = beta1*m[i] + (1-beta1)*g',
    '            v[i] = beta2*v[i] + (1-beta2)*g**2',
    '            m_hat = m[i] / (1 - beta1**t_adam)',
    '            v_hat = v[i] / (1 - beta2**t_adam)',
    '            p.data -= lr * m_hat / (v_hat.sqrt() + eps)',
    '        mse, acc = compute_metrics(outputs_t.detach(), targets, output_mask)',
    '        mse_history.append(mse)',
    '        acc_history.append(acc)',
    '        if verbose and (it+1) % log_interval == 0:',
    '            print("  Iter", it+1, "| MSE:", round(mse, 5), "| Acc:", round(acc, 4))',
    '    return mse_history, acc_history',
    '',
    'print("")',
    'print("=== Testing FA (200 iterations, lr=0.001) ===")',
    'mse_h, acc_h = train_fa(n_iter=200, lr=0.001, log_interval=50)',
    'print("")',
    'print("FA Quick Test Results:")',
    'print("  Initial MSE:", round(mse_h[0], 5))',
    'print("  Final MSE:", round(mse_h[-1], 5))',
    'print("  Final Acc:", round(acc_h[-1], 4))',
    'print("  Learning:", mse_h[-1] < mse_h[0])',
    '',
    'print("")',
    'print("=== Testing FA (500 iterations, lr=0.001) ===")',
    'mse_h2, acc_h2 = train_fa(n_iter=500, lr=0.001, log_interval=100)',
    'print("")',
    'print("FA 500-iter Results:")',
    'print("  Initial MSE:", round(mse_h2[0], 5))',
    'print("  Final MSE:", round(mse_h2[-1], 5))',
    'print("  Final Acc:", round(acc_h2[-1], 4))',
    '',
    'print("")',
    'print("=== Step 3: FA Implementation VERIFIED ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=180)
print('STDOUT:', result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step4_psc.py')
lines = [
    'import torch',
    'import numpy as np',
    'import math',
    'import os',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    '',
    'def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):',
    '    if dev is None: dev = device',
    '    seq_len = n_bits + delay + n_bits',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, n_bits+1)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*targets.shape[-1])',
    '    pred_bits = (outputs > 0.5).float()',
    '    correct = ((pred_bits == targets).float() * mask).sum()',
    '    return mse.item(), (correct / (mask.sum()*targets.shape[-1])).item()',
    '',
    '# ============================================================',
    '# PSC TRAINING FUNCTION',
    '# ============================================================',
    'def train_psc(n_iter=200, batch_size=32, lr=0.001, lr_pred=0.01,',
    '              beta=0.3, gamma=0.5, lam=0.9, T_theta=10,',
    '              use_oscillatory_gate=True, verbose=True, log_interval=50):',
    '    """',
    '    PSC: Predictive Self-Correction learning rule.',
    '    Args:',
    '      beta: prediction compartment decay rate',
    '      gamma: weight for prediction error vs output error in credit signal',
    '      lam: eligibility trace decay',
    '      T_theta: oscillation period for gating',
    '      use_oscillatory_gate: if False, gate=1 always (ablation)',
    '    """',
    '    torch.manual_seed(SEED)',
    '    # Main RNN weights',
    '    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)',
    '    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_rec = torch.zeros(HIDDEN_SIZE).to(device)',
    '    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_out = torch.zeros(OUTPUT_DIM).to(device)',
    '    # Prediction weight matrix (separate, self-supervised)',
    '    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)',
    '    # Adam states for main weights',
    '    params = [W_in, W_rec, b_rec, W_out, b_out]',
    '    m = [torch.zeros_like(p) for p in params]',
    '    v = [torch.zeros_like(p) for p in params]',
    '    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8',
    '    mse_history, acc_history = [], []',
    '    for it in range(n_iter):',
    '        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)',
    '        # Initialize hidden state, prediction compartment, eligibility trace',
    '        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        p = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        e = torch.zeros(batch_size, HIDDEN_SIZE, HIDDEN_SIZE, device=device)',
    '        # Accumulate weight updates over sequence',
    '        dW_in  = torch.zeros_like(W_in)',
    '        dW_rec = torch.zeros_like(W_rec)',
    '        db_rec = torch.zeros_like(b_rec)',
    '        dW_out = torch.zeros_like(W_out)',
    '        db_out = torch.zeros_like(b_out)',
    '        dW_pred = torch.zeros_like(W_pred)',
    '        outputs_list = []',
    '        for t in range(SEQ_LEN):',
    '            x_t = inputs_b[:, t, :]  # (batch, input_dim)',
    '            h_prev = h.clone()',
    '            # Main RNN forward',
    '            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec',
    '            h = torch.tanh(a_t)  # (batch, hidden)',
    '            dtanh = 1.0 - h**2    # (batch, hidden)',
    '            # Output',
    '            y_t = h @ W_out.T + b_out  # (batch, output_dim)',
    '            outputs_list.append(y_t)',
    '            # Step 2: Update prediction compartment',
    '            # p_i(t) = (1-beta)*p_i(t-1) + beta*(W_pred[i,:] @ h_prev)',
    '            p = (1 - beta) * p + beta * (h_prev @ W_pred.T)',
    '            # Step 3: Prediction error (batch, hidden)',
    '            delta_pred = h - p',
    '            # Step 4: Update eligibility trace',
    '            # e_ij(t) = lam*e_ij(t-1) + h_prev_j * dtanh_i',
    '            # e shape: (batch, hidden_i, hidden_j)',
    '            e = lam * e + torch.bmm(dtanh.unsqueeze(2), h_prev.unsqueeze(1))',
    '            # Step 5: Output error (only at output timesteps)',
    '            is_output = output_mask[:, t].unsqueeze(1)  # (batch, 1)',
    '            out_err_t = (y_t - targets_b[:, t, :]) * is_output  # (batch, output_dim)',
    '            # Project output error to hidden space via W_out^T',
    '            output_err_hidden = out_err_t @ W_out  # (batch, hidden)',
    '            # Step 6: Credit signal',
    '            s_t = gamma * delta_pred + (1 - gamma) * output_err_hidden  # (batch, hidden)',
    '            # Step 7: Oscillatory gate',
    '            if use_oscillatory_gate:',
    '                gate = max(0.0, math.sin(2 * math.pi * t / T_theta))',
    '            else:',
    '                gate = 1.0',
    '            # Step 8: W_rec update: Delta_W_rec_ij = -lr * s_i(t) * e_ij(t) * gate',
    '            # s_t: (batch, hidden_i), e: (batch, hidden_i, hidden_j)',
    '            # dW_rec_ij += mean over batch of s_i * e_ij * gate',
    '            if gate > 0:',
    '                s_e = s_t.unsqueeze(2) * e  # (batch, hidden_i, hidden_j)',
    '                dW_rec += gate * s_e.mean(0)',
    '                # W_in update using same credit signal with input eligibility',
    '                # eligibility for W_in: dtanh_i * x_j at current step',
    '                e_in = torch.bmm(dtanh.unsqueeze(2), x_t.unsqueeze(1))  # (batch, hidden, input)',
    '                dW_in += gate * (s_t.unsqueeze(2) * e_in).mean(0)',
    '                db_rec += gate * s_t.mean(0)',
    '            # W_out update (always, standard gradient)',
    '            dW_out += out_err_t.T @ h / batch_size',
    '            db_out += out_err_t.mean(0)',
    '            # Step 9: W_pred update (Hebbian, always)',
    '            dW_pred += (delta_pred.unsqueeze(2) * h_prev.unsqueeze(1)).mean(0)',
    '        # Clip and apply gradients with Adam',
    '        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]',
    '        total_norm = sum(g.norm()**2 for g in grads)**0.5',
    '        if total_norm > 1.0:',
    '            clip = 1.0 / (total_norm + 1e-8)',
    '            grads = [g * clip for g in grads]',
    '        t_adam = it + 1',
    '        for i, (p_w, g) in enumerate(zip(params, grads)):',
    '            m[i] = beta1*m[i] + (1-beta1)*g',
    '            v[i] = beta2*v[i] + (1-beta2)*g**2',
    '            m_hat = m[i] / (1 - beta1**t_adam)',
    '            v_hat = v[i] / (1 - beta2**t_adam)',
    '            p_w.data -= lr * m_hat / (v_hat.sqrt() + eps_adam)',
    '        # W_pred update (simple SGD, no Adam needed)',
    '        W_pred.data -= lr_pred * dW_pred',
    '        outputs_t = torch.stack(outputs_list, dim=1)',
    '        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)',
    '        mse_history.append(mse)',
    '        acc_history.append(acc)',
    '        if verbose and (it+1) % log_interval == 0:',
    '            print("  Iter", it+1, "| MSE:", round(mse, 5), "| Acc:", round(acc, 4),',
    '                  "| gate_mode:", "osc" if use_oscillatory_gate else "always")',
    '    return mse_history, acc_history',
    '',
    'print("")',
    'print("=== Testing PSC WITH oscillatory gate (200 iters) ===")',
    'mse1, acc1 = train_psc(n_iter=200, lr=0.001, beta=0.3, gamma=0.5,',
    '                        lam=0.9, T_theta=10, use_oscillatory_gate=True, log_interval=50)',
    'print("PSC-osc: Initial MSE:", round(mse1[0],5), "Final MSE:", round(mse1[-1],5),',
    '        "Final Acc:", round(acc1[-1],4))',
    '',
    'print("")',
    'print("=== Testing PSC WITHOUT oscillatory gate (200 iters) ===")',
    'mse2, acc2 = train_psc(n_iter=200, lr=0.001, beta=0.3, gamma=0.5,',
    '                        lam=0.9, T_theta=10, use_oscillatory_gate=False, log_interval=50)',
    'print("PSC-always: Initial MSE:", round(mse2[0],5), "Final MSE:", round(mse2[-1],5),',
    '        "Final Acc:", round(acc2[-1],4))',
    '',
    'print("")',
    'print("=== Testing PSC with different hyperparams (lr=0.005, beta=0.1, gamma=0.7) ===")',
    'mse3, acc3 = train_psc(n_iter=200, lr=0.005, beta=0.1, gamma=0.7,',
    '                        lam=0.95, T_theta=20, use_oscillatory_gate=True, log_interval=50)',
    'print("PSC-v2: Initial MSE:", round(mse3[0],5), "Final MSE:", round(mse3[-1],5),',
    '        "Final Acc:", round(acc3[-1],4))',
    '',
    'print("")',
    'print("=== Step 4: PSC Implementation VERIFIED ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=300)
print('STDOUT:', result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step4_psc_v2.py')
lines = [
    'import torch',
    'import numpy as np',
    'import math',
    'import os',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    '',
    'def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):',
    '    if dev is None: dev = device',
    '    seq_len = n_bits + delay + n_bits',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, n_bits+1)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*targets.shape[-1])',
    '    pred_bits = (outputs > 0.5).float()',
    '    correct = ((pred_bits == targets).float() * mask).sum()',
    '    n_total = mask.sum() * targets.shape[-1]',
    '    return mse.item(), (correct / n_total).item()',
    '',
    '# ============================================================',
    '# PSC TRAINING FUNCTION (v2 - with stability fixes)',
    '# ============================================================',
    'def train_psc(n_iter=200, batch_size=32, lr=0.001, lr_pred=0.001,',
    '              beta=0.3, gamma=0.5, lam=0.9, T_theta=10,',
    '              use_oscillatory_gate=True, verbose=True, log_interval=50):',
    '    torch.manual_seed(SEED)',
    '    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)',
    '    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_rec = torch.zeros(HIDDEN_SIZE).to(device)',
    '    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_out = torch.zeros(OUTPUT_DIM).to(device)',
    '    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)',
    '    params = [W_in, W_rec, b_rec, W_out, b_out]',
    '    m = [torch.zeros_like(p) for p in params]',
    '    v = [torch.zeros_like(p) for p in params]',
    '    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8',
    '    mse_history, acc_history = [], []',
    '    for it in range(n_iter):',
    '        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)',
    '        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        p_comp = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        # Use per-neuron eligibility trace (not full matrix) for efficiency',
    '        # e_i(t) = lam*e_i(t-1) + mean_j[h_j(t-1)] * dtanh_i(t)',
    '        # But we need e_ij for W_rec update, so keep full trace but clip it',
    '        e = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=device)  # averaged over batch',
    '        dW_in  = torch.zeros_like(W_in)',
    '        dW_rec = torch.zeros_like(W_rec)',
    '        db_rec = torch.zeros_like(b_rec)',
    '        dW_out = torch.zeros_like(W_out)',
    '        db_out = torch.zeros_like(b_out)',
    '        dW_pred = torch.zeros_like(W_pred)',
    '        outputs_list = []',
    '        for t in range(SEQ_LEN):',
    '            x_t = inputs_b[:, t, :]',
    '            h_prev = h.clone()',
    '            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec',
    '            h = torch.tanh(a_t)',
    '            dtanh = 1.0 - h**2',
    '            y_t = h @ W_out.T + b_out',
    '            outputs_list.append(y_t)',
    '            # Prediction compartment',
    '            p_comp = (1 - beta) * p_comp + beta * (h_prev @ W_pred.T)',
    '            delta_pred = h - p_comp  # (batch, hidden)',
    '            # Eligibility trace (batch-averaged outer product)',
    '            h_prev_mean = h_prev.mean(0)  # (hidden,)',
    '            dtanh_mean  = dtanh.mean(0)   # (hidden,)',
    '            e = lam * e + torch.outer(dtanh_mean, h_prev_mean)',
    '            # Clip eligibility trace for stability',
    '            e_norm = e.norm()',
    '            if e_norm > 10.0:',
    '                e = e * 10.0 / (e_norm + 1e-8)',
    '            # Output error',
    '            is_output = output_mask[:, t].unsqueeze(1)',
    '            out_err_t = (y_t - targets_b[:, t, :]) * is_output',
    '            output_err_hidden = out_err_t @ W_out  # (batch, hidden)',
    '            # Credit signal (batch-averaged)',
    '            delta_pred_mean = delta_pred.mean(0)  # (hidden,)',
    '            output_err_mean = output_err_hidden.mean(0)  # (hidden,)',
    '            s_t = gamma * delta_pred_mean + (1 - gamma) * output_err_mean  # (hidden,)',
    '            # Clip credit signal',
    '            s_norm = s_t.norm()',
    '            if s_norm > 5.0:',
    '                s_t = s_t * 5.0 / (s_norm + 1e-8)',
    '            # Oscillatory gate',
    '            if use_oscillatory_gate:',
    '                gate = max(0.0, math.sin(2 * math.pi * t / T_theta))',
    '            else:',
    '                gate = 1.0',
    '            # Weight updates',
    '            if gate > 0:',
    '                # dW_rec_ij += gate * s_i * e_ij',
    '                dW_rec += gate * torch.outer(s_t, torch.ones(HIDDEN_SIZE, device=device)) * e',
    '                # W_in: eligibility with input',
    '                x_mean = x_t.mean(0)',
    '                e_in = torch.outer(dtanh_mean, x_mean)',
    '                dW_in += gate * torch.outer(s_t, torch.ones(INPUT_DIM, device=device)) * e_in',
    '                db_rec += gate * s_t',
    '            # W_out update (standard gradient)',
    '            dW_out += out_err_t.T @ h / batch_size',
    '            db_out += out_err_t.mean(0)',
    '            # W_pred Hebbian update',
    '            dW_pred += (delta_pred.unsqueeze(2) * h_prev.unsqueeze(1)).mean(0)',
    '        # Clip main gradients',
    '        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]',
    '        total_norm = sum(g.norm()**2 for g in grads)**0.5',
    '        if total_norm > 1.0:',
    '            clip = 1.0 / (total_norm + 1e-8)',
    '            grads = [g * clip for g in grads]',
    '        # Adam update for main weights',
    '        t_adam = it + 1',
    '        for i, (p_w, g) in enumerate(zip(params, grads)):',
    '            m[i] = beta1*m[i] + (1-beta1)*g',
    '            v[i] = beta2*v[i] + (1-beta2)*g**2',
    '            m_hat = m[i] / (1 - beta1**t_adam)',
    '            v_hat = v[i] / (1 - beta2**t_adam)',
    '            p_w.data -= lr * m_hat / (v_hat.sqrt() + eps_adam)',
    '        # Clip and apply W_pred update',
    '        pred_norm = dW_pred.norm()',
    '        if pred_norm > 1.0:',
    '            dW_pred = dW_pred / (pred_norm + 1e-8)',
    '        W_pred.data -= lr_pred * dW_pred',
    '        outputs_t = torch.stack(outputs_list, dim=1)',
    '        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)',
    '        mse_history.append(mse)',
    '        acc_history.append(acc)',
    '        if verbose and (it+1) % log_interval == 0:',
    '            print("  Iter", it+1, "| MSE:", round(mse, 5), "| Acc:", round(acc, 4))',
    '    return mse_history, acc_history',
    '',
    'print("")',
    'print("=== PSC-osc (200 iters, lr=0.001) ===")',
    'mse1, acc1 = train_psc(200, lr=0.001, beta=0.3, gamma=0.5, lam=0.9, T_theta=10,',
    '                        use_oscillatory_gate=True, log_interval=50)',
    'print("Initial MSE:", round(mse1[0],5), "Final MSE:", round(mse1[-1],5), "Final Acc:", round(acc1[-1],4))',
    '',
    'print("")',
    'print("=== PSC-no-gate (200 iters, lr=0.001) ===")',
    'mse2, acc2 = train_psc(200, lr=0.001, beta=0.3, gamma=0.5, lam=0.9, T_theta=10,',
    '                        use_oscillatory_gate=False, log_interval=50)',
    'print("Initial MSE:", round(mse2[0],5), "Final MSE:", round(mse2[-1],5), "Final Acc:", round(acc2[-1],4))',
    '',
    'print("")',
    'print("=== PSC-osc higher lr (200 iters, lr=0.005) ===")',
    'mse3, acc3 = train_psc(200, lr=0.005, beta=0.1, gamma=0.3, lam=0.95, T_theta=20,',
    '                        use_oscillatory_gate=True, log_interval=50)',
    'print("Initial MSE:", round(mse3[0],5), "Final MSE:", round(mse3[-1],5), "Final Acc:", round(acc3[-1],4))',
    '',
    '# Check for NaN',
    'import math',
    'nan_check = [math.isnan(x) for x in mse1+mse2+mse3]',
    'print("")',
    'print("NaN in any run:", any(nan_check))',
    'print("=== Step 4: PSC v2 Implementation Test Complete ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=300)
print('STDOUT:', result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step4_psc_v3.py')
lines = [
    'import torch',
    'import numpy as np',
    'import math',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    '',
    'def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):',
    '    if dev is None: dev = device',
    '    seq_len = n_bits + delay + n_bits',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, n_bits+1)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*targets.shape[-1])',
    '    pred_bits = (outputs > 0.5).float()',
    '    correct = ((pred_bits == targets).float() * mask).sum()',
    '    return mse.item(), (correct / (mask.sum()*targets.shape[-1])).item()',
    '',
    '# ============================================================',
    '# PSC TRAINING FUNCTION v3',
    '# Key fix: correct W_rec update formula using batch-level e-prop style',
    '# Delta_W_rec[i,j] = s_i * e_ij  (s_i scalar credit, e_ij eligibility)',
    '# ============================================================',
    'def train_psc(n_iter=500, batch_size=32, lr=0.001, lr_pred=0.001,',
    '              beta=0.3, gamma=0.0, lam=0.9, T_theta=10,',
    '              use_oscillatory_gate=False, verbose=True, log_interval=100):',
    '    torch.manual_seed(SEED)',
    '    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)',
    '    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_rec = torch.zeros(HIDDEN_SIZE).to(device)',
    '    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_out = torch.zeros(OUTPUT_DIM).to(device)',
    '    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)',
    '    params = [W_in, W_rec, b_rec, W_out, b_out]',
    '    m_adam = [torch.zeros_like(p) for p in params]',
    '    v_adam = [torch.zeros_like(p) for p in params]',
    '    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8',
    '    mse_history, acc_history = [], []',
    '    for it in range(n_iter):',
    '        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)',
    '        # Per-sample eligibility traces to properly handle batch',
    '        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        p_comp = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        # Eligibility trace: shape (batch, hidden_i, hidden_j)',
    '        e_rec = torch.zeros(batch_size, HIDDEN_SIZE, HIDDEN_SIZE, device=device)',
    '        e_in_trace = torch.zeros(batch_size, HIDDEN_SIZE, INPUT_DIM, device=device)',
    '        dW_in  = torch.zeros_like(W_in)',
    '        dW_rec = torch.zeros_like(W_rec)',
    '        db_rec = torch.zeros_like(b_rec)',
    '        dW_out = torch.zeros_like(W_out)',
    '        db_out = torch.zeros_like(b_out)',
    '        dW_pred = torch.zeros_like(W_pred)',
    '        outputs_list = []',
    '        for t in range(SEQ_LEN):',
    '            x_t = inputs_b[:, t, :]   # (batch, input_dim)',
    '            h_prev = h.clone()          # (batch, hidden)',
    '            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec',
    '            h = torch.tanh(a_t)',
    '            dtanh = 1.0 - h**2          # (batch, hidden)',
    '            y_t = h @ W_out.T + b_out',
    '            outputs_list.append(y_t)',
    '            # Prediction compartment per sample',
    '            p_comp = (1 - beta) * p_comp + beta * (h_prev @ W_pred.T)',
    '            delta_pred = h - p_comp  # (batch, hidden)',
    '            # Eligibility traces per sample',
    '            # e_rec[b,i,j] = lam*e_rec[b,i,j] + dtanh[b,i] * h_prev[b,j]',
    '            e_rec = lam * e_rec + torch.bmm(dtanh.unsqueeze(2), h_prev.unsqueeze(1))',
    '            e_in_trace = lam * e_in_trace + torch.bmm(dtanh.unsqueeze(2), x_t.unsqueeze(1))',
    '            # Clip eligibility traces',
    '            e_rec_norm = e_rec.norm(dim=(1,2), keepdim=True)',
    '            e_rec = torch.where(e_rec_norm > 5.0, e_rec * 5.0 / (e_rec_norm + 1e-8), e_rec)',
    '            # Output error (only at output timesteps)',
    '            is_output = output_mask[:, t].bool()   # (batch,)',
    '            out_err_t = torch.zeros_like(y_t)',
    '            if is_output.any():',
    '                out_err_t[is_output] = y_t[is_output] - targets_b[:, t, :][is_output]',
    '            # Output error projected to hidden via W_out^T',
    '            output_err_hidden = out_err_t @ W_out  # (batch, hidden)',
    '            # Credit signal per sample',
    '            # s_i = gamma * delta_pred_i + (1-gamma) * output_err_i',
    '            s_t = gamma * delta_pred + (1 - gamma) * output_err_hidden  # (batch, hidden)',
    '            # Oscillatory gate',
    '            if use_oscillatory_gate:',
    '                gate = max(0.0, math.sin(2 * math.pi * t / T_theta))',
    '            else:',
    '                gate = 1.0',
    '            # W_rec update: Delta_W_rec[b,i,j] = gate * s_t[b,i] * e_rec[b,i,j]',
    '            if gate > 0:',
    '                # s_t: (batch, hidden_i) -> (batch, hidden_i, 1)',
    '                # e_rec: (batch, hidden_i, hidden_j)',
    '                dW_rec += gate * (s_t.unsqueeze(2) * e_rec).mean(0)',
    '                dW_in  += gate * (s_t.unsqueeze(2) * e_in_trace).mean(0)',
    '                db_rec += gate * s_t.mean(0)',
    '            # W_out update (standard)',
    '            dW_out += out_err_t.T @ h / batch_size',
    '            db_out += out_err_t.mean(0)',
    '            # W_pred Hebbian update: Delta_W_pred[i,j] = -lr_pred * delta_pred_i * h_prev_j',
    '            dW_pred += (delta_pred.unsqueeze(2) * h_prev.unsqueeze(1)).mean(0)',
    '        # Clip and apply main gradients',
    '        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]',
    '        total_norm = sum(g.norm()**2 for g in grads)**0.5',
    '        if total_norm > 1.0:',
    '            clip = 1.0 / (total_norm + 1e-8)',
    '            grads = [g * clip for g in grads]',
    '        t_adam = it + 1',
    '        for i, (p_w, g) in enumerate(zip(params, grads)):',
    '            m_adam[i] = beta1*m_adam[i] + (1-beta1)*g',
    '            v_adam[i] = beta2*v_adam[i] + (1-beta2)*g**2',
    '            m_hat = m_adam[i] / (1 - beta1**t_adam)',
    '            v_hat = v_adam[i] / (1 - beta2**t_adam)',
    '            p_w.data -= lr * m_hat / (v_hat.sqrt() + eps_adam)',
    '        # W_pred update',
    '        dW_pred_norm = dW_pred.norm()',
    '        if dW_pred_norm > 1.0:',
    '            dW_pred = dW_pred / (dW_pred_norm + 1e-8)',
    '        W_pred.data -= lr_pred * dW_pred',
    '        outputs_t = torch.stack(outputs_list, dim=1)',
    '        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)',
    '        mse_history.append(mse)',
    '        acc_history.append(acc)',
    '        if verbose and (it+1) % log_interval == 0:',
    '            print("  Iter", it+1, "| MSE:", round(mse,5), "| Acc:", round(acc,4))',
    '    return mse_history, acc_history',
    '',
    '# Test 1: Pure output error (gamma=0) without gate - should learn if e-prop-like rule works',
    'print("")',
    'print("=== PSC gamma=0 (pure output err), no gate, 500 iters ===")',
    'mse0, acc0 = train_psc(500, lr=0.001, gamma=0.0, lam=0.9, use_oscillatory_gate=False)',
    'print("gamma=0: MSE", round(mse0[0],4), "->", round(mse0[-1],4), "| Acc", round(acc0[-1],4))',
    '',
    '# Test 2: Mixed (gamma=0.3) without gate',
    'print("")',
    'print("=== PSC gamma=0.3 (mixed), no gate, 500 iters ===")',
    'mse3, acc3 = train_psc(500, lr=0.001, gamma=0.3, lam=0.9, use_oscillatory_gate=False)',
    'print("gamma=0.3: MSE", round(mse3[0],4), "->", round(mse3[-1],4), "| Acc", round(acc3[-1],4))',
    '',
    '# Test 3: Mixed (gamma=0.3) WITH oscillatory gate',
    'print("")',
    'print("=== PSC gamma=0.3, WITH gate (T=10), 500 iters ===")',
    'mse3g, acc3g = train_psc(500, lr=0.001, gamma=0.3, lam=0.9, T_theta=10, use_oscillatory_gate=True)',
    'print("gamma=0.3+gate: MSE", round(mse3g[0],4), "->", round(mse3g[-1],4), "| Acc", round(acc3g[-1],4))',
    '',
    '# Check for NaN',
    'all_mse = mse0 + mse3 + mse3g',
    'nan_count = sum(1 for x in all_mse if x != x)',
    'print("")',
    'print("NaN count:", nan_count)',
    'print("Learning (mse0):", mse0[-1] < mse0[0])',
    'print("Learning (mse3):", mse3[-1] < mse3[0])',
    'print("=== Step 4: PSC v3 test complete ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=600)
print('STDOUT:', result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step5_grid_search.py')
lines = [
    'import torch',
    'import numpy as np',
    'import math',
    'import json',
    'import os',
    'from itertools import product',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'N_BITS = 8',
    'DELAY = 10',
    'INPUT_DIM = N_BITS + 1',
    'OUTPUT_DIM = N_BITS',
    'HIDDEN_SIZE = 128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    'WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"',
    '',
    'def generate_copy_batch(batch_size, dev=None):',
    '    if dev is None: dev = device',
    '    n_bits, delay = N_BITS, DELAY',
    '    seq_len = n_bits + delay + n_bits',
    '    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, seq_len, n_bits+1)',
    '    targets = torch.zeros(batch_size, seq_len, n_bits)',
    '    output_mask = torch.zeros(batch_size, seq_len)',
    '    for t in range(n_bits):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, n_bits + delay - 1, n_bits] = 1.0',
    '    output_start = n_bits + delay',
    '    for t in range(n_bits):',
    '        targets[:, output_start + t, :] = patterns',
    '    output_mask[:, output_start:output_start + n_bits] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*targets.shape[-1])',
    '    pred_bits = (outputs > 0.5).float()',
    '    correct = ((pred_bits == targets).float() * mask).sum()',
    '    return mse.item(), (correct / (mask.sum()*targets.shape[-1])).item()',
    '',
    'def train_psc(n_iter, batch_size=32, lr=0.001, lr_pred=0.001,',
    '              beta=0.3, gamma=0.3, lam=0.9, T_theta=10,',
    '              use_oscillatory_gate=True):',
    '    torch.manual_seed(SEED)',
    '    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)',
    '    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_rec = torch.zeros(HIDDEN_SIZE).to(device)',
    '    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)',
    '    b_out = torch.zeros(OUTPUT_DIM).to(device)',
    '    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)',
    '    params = [W_in, W_rec, b_rec, W_out, b_out]',
    '    m_adam = [torch.zeros_like(p) for p in params]',
    '    v_adam = [torch.zeros_like(p) for p in params]',
    '    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8',
    '    mse_history, acc_history = [], []',
    '    for it in range(n_iter):',
    '        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)',
    '        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        p_comp = torch.zeros(batch_size, HIDDEN_SIZE, device=device)',
    '        e_rec = torch.zeros(batch_size, HIDDEN_SIZE, HIDDEN_SIZE, device=device)',
    '        e_in_trace = torch.zeros(batch_size, HIDDEN_SIZE, INPUT_DIM, device=device)',
    '        dW_in=torch.zeros_like(W_in); dW_rec=torch.zeros_like(W_rec)',
    '        db_rec=torch.zeros_like(b_rec); dW_out=torch.zeros_like(W_out)',
    '        db_out=torch.zeros_like(b_out); dW_pred=torch.zeros_like(W_pred)',
    '        outputs_list = []',
    '        for t in range(SEQ_LEN):',
    '            x_t = inputs_b[:, t, :]',
    '            h_prev = h.clone()',
    '            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec',
    '            h = torch.tanh(a_t)',
    '            dtanh = 1.0 - h**2',
    '            y_t = h @ W_out.T + b_out',
    '            outputs_list.append(y_t)',
    '            p_comp = (1-beta)*p_comp + beta*(h_prev @ W_pred.T)',
    '            delta_pred = h - p_comp',
    '            e_rec = lam*e_rec + torch.bmm(dtanh.unsqueeze(2), h_prev.unsqueeze(1))',
    '            e_in_trace = lam*e_in_trace + torch.bmm(dtanh.unsqueeze(2), x_t.unsqueeze(1))',
    '            e_rec_norm = e_rec.norm(dim=(1,2), keepdim=True)',
    '            e_rec = torch.where(e_rec_norm > 5.0, e_rec*5.0/(e_rec_norm+1e-8), e_rec)',
    '            is_output = output_mask[:, t].bool()',
    '            out_err_t = torch.zeros_like(y_t)',
    '            if is_output.any():',
    '                out_err_t[is_output] = y_t[is_output] - targets_b[:,t,:][is_output]',
    '            output_err_hidden = out_err_t @ W_out',
    '            s_t = gamma*delta_pred + (1-gamma)*output_err_hidden',
    '            gate = max(0.0, math.sin(2*math.pi*t/T_theta)) if use_oscillatory_gate else 1.0',
    '            if gate > 0:',
    '                dW_rec += gate*(s_t.unsqueeze(2)*e_rec).mean(0)',
    '                dW_in  += gate*(s_t.unsqueeze(2)*e_in_trace).mean(0)',
    '                db_rec += gate*s_t.mean(0)',
    '            dW_out += out_err_t.T @ h / batch_size',
    '            db_out += out_err_t.mean(0)',
    '            dW_pred += (delta_pred.unsqueeze(2)*h_prev.unsqueeze(1)).mean(0)',
    '        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]',
    '        total_norm = sum(g.norm()**2 for g in grads)**0.5',
    '        if total_norm > 1.0:',
    '            clip = 1.0/(total_norm+1e-8)',
    '            grads = [g*clip for g in grads]',
    '        t_adam = it+1',
    '        for i,(p_w,g) in enumerate(zip(params,grads)):',
    '            m_adam[i]=beta1*m_adam[i]+(1-beta1)*g',
    '            v_adam[i]=beta2*v_adam[i]+(1-beta2)*g**2',
    '            m_hat=m_adam[i]/(1-beta1**t_adam)',
    '            v_hat=v_adam[i]/(1-beta2**t_adam)',
    '            p_w.data -= lr*m_hat/(v_hat.sqrt()+eps_adam)',
    '        dW_pred_norm = dW_pred.norm()',
    '        if dW_pred_norm > 1.0: dW_pred = dW_pred/(dW_pred_norm+1e-8)',
    '        W_pred.data -= lr_pred*dW_pred',
    '        outputs_t = torch.stack(outputs_list, dim=1)',
    '        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)',
    '        mse_history.append(mse)',
    '        acc_history.append(acc)',
    '    return mse_history, acc_history',
    '',
    '# ============================================================',
    '# GRID SEARCH',
    '# ============================================================',
    'lrs = [0.001, 0.005, 0.01]',
    'betas = [0.1, 0.3]',
    'gammas = [0.3, 0.5, 0.7]',
    'lams = [0.9, 0.95]',
    'T_thetas = [10, 20]',
    'lr_pred_fixed = 0.001',
    'N_GRID_ITER = 1000',
    '',
    'grid_results_osc = []',
    'grid_results_nogate = []',
    '',
    '# PSC with oscillatory gate',
    'print("")',
    'print("=== Grid Search: PSC WITH oscillatory gate ===")',
    'combo_count = 0',
    'for lr, beta, gamma, lam, T_theta in product(lrs, betas, gammas, lams, T_thetas):',
    '    combo_count += 1',
    '    mse_h, acc_h = train_psc(N_GRID_ITER, lr=lr, lr_pred=lr_pred_fixed,',
    '                              beta=beta, gamma=gamma, lam=lam, T_theta=T_theta,',
    '                              use_oscillatory_gate=True)',
    '    final_mse = mse_h[-1] if not math.isnan(mse_h[-1]) else 999.0',
    '    final_acc = acc_h[-1] if not math.isnan(acc_h[-1]) else 0.0',
    '    config = {"lr":lr, "beta":beta, "gamma":gamma, "lam":lam, "T_theta":T_theta,',
    '              "final_mse":final_mse, "final_acc":final_acc,',
    '              "mse_history":mse_h[::100], "acc_history":acc_h[::100]}',
    '    grid_results_osc.append(config)',
    '    if combo_count % 12 == 0:',
    '        print("  Combo", combo_count, "/ 72 done. Best so far:",',
    '              round(min(r["final_mse"] for r in grid_results_osc), 4))',
    '',
    '# PSC without gate (T_theta doesn\'t matter, so use T_theta=10 only)',
    'print("")',
    'print("=== Grid Search: PSC WITHOUT gate ===")',
    'combo_count = 0',
    'for lr, beta, gamma, lam in product(lrs, betas, gammas, lams):',
    '    combo_count += 1',
    '    mse_h, acc_h = train_psc(N_GRID_ITER, lr=lr, lr_pred=lr_pred_fixed,',
    '                              beta=beta, gamma=gamma, lam=lam, T_theta=10,',
    '                              use_oscillatory_gate=False)',
    '    final_mse = mse_h[-1] if not math.isnan(mse_h[-1]) else 999.0',
    '    final_acc = acc_h[-1] if not math.isnan(acc_h[-1]) else 0.0',
    '    config = {"lr":lr, "beta":beta, "gamma":gamma, "lam":lam, "T_theta":"N/A",',
    '              "final_mse":final_mse, "final_acc":final_acc,',
    '              "mse_history":mse_h[::100], "acc_history":acc_h[::100]}',
    '    grid_results_nogate.append(config)',
    '    if combo_count % 9 == 0:',
    '        print("  Combo", combo_count, "/ 36 done. Best so far:",',
    '              round(min(r["final_mse"] for r in grid_results_nogate), 4))',
    '',
    '# Save results',
    'grid_results_osc.sort(key=lambda x: x["final_mse"])',
    'grid_results_nogate.sort(key=lambda x: x["final_mse"])',
    '',
    'results = {"grid_psc_osc": grid_results_osc, "grid_psc_nogate": grid_results_nogate}',
    'out_path = os.path.join(WORK_DIR, "grid_search_results.json")',
    'with open(out_path, "w") as f:',
    '    json.dump(results, f, indent=2)',
    'print("")',
    'print("Grid search results saved to", out_path)',
    '',
    'print("")',
    'print("=== Top 5 PSC-OSC configs ===")',
    'for i, cfg in enumerate(grid_results_osc[:5]):',
    '    print("  Rank", i+1, "| lr:", cfg["lr"], "beta:", cfg["beta"],',
    '          "gamma:", cfg["gamma"], "lam:", cfg["lam"], "T:", cfg["T_theta"],',
    '          "| MSE:", round(cfg["final_mse"],4), "Acc:", round(cfg["final_acc"],4))',
    '',
    'print("")',
    'print("=== Top 5 PSC-NOGATE configs ===")',
    'for i, cfg in enumerate(grid_results_nogate[:5]):',
    '    print("  Rank", i+1, "| lr:", cfg["lr"], "beta:", cfg["beta"],',
    '          "gamma:", cfg["gamma"], "lam:", cfg["lam"],',
    '          "| MSE:", round(cfg["final_mse"],4), "Acc:", round(cfg["final_acc"],4))',
    '',
    'print("")',
    'print("=== Step 5: Grid Search COMPLETE ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
print('Script written, starting grid search (this will take a while)...')
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=7200)  # 2 hour timeout
print('STDOUT:', result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
import json
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
results_path = os.path.join(WORK_DIR, 'grid_search_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)
top3_osc = results['grid_psc_osc'][:3]
top3_nogate = results['grid_psc_nogate'][:3]
print('=== Top 3 PSC-WITH-GATE configs (from grid search) ===')
for i, cfg in enumerate(top3_osc):
    print(f'  Rank {i+1}: lr={cfg["lr"]}, beta={cfg["beta"]}, gamma={cfg["gamma"]}, lam={cfg["lam"]}, T_theta={cfg["T_theta"]}')
    print(f'           Final MSE={round(cfg["final_mse"],4)}, Final Acc={round(cfg["final_acc"],4)}')
print()
print('=== Top 3 PSC-WITHOUT-GATE configs (from grid search) ===')
for i, cfg in enumerate(top3_nogate):
    print(f'  Rank {i+1}: lr={cfg["lr"]}, beta={cfg["beta"]}, gamma={cfg["gamma"]}, lam={cfg["lam"]}')
    print(f'           Final MSE={round(cfg["final_mse"],4)}, Final Acc={round(cfg["final_acc"],4)}')
print()
print('=== Selected best configs for full 5000-iter training ===')
best_osc = top3_osc[0]
best_nogate = top3_nogate[0]
print(f'Best PSC-osc: lr={best_osc["lr"]}, beta={best_osc["beta"]}, gamma={best_osc["gamma"]}, lam={best_osc["lam"]}, T_theta={best_osc["T_theta"]}')
print(f'Best PSC-nogate: lr={best_nogate["lr"]}, beta={best_nogate["beta"]}, gamma={best_nogate["gamma"]}, lam={best_nogate["lam"]}')
selected = {
    'best_psc_osc': {
        'lr': best_osc['lr'],
        'beta': best_osc['beta'],
        'gamma': best_osc['gamma'],
        'lam': best_osc['lam'],
        'T_theta': best_osc['T_theta'],
        'grid_mse': best_osc['final_mse'],
        'grid_acc': best_osc['final_acc']
    },
    'best_psc_nogate': {
        'lr': best_nogate['lr'],
        'beta': best_nogate['beta'],
        'gamma': best_nogate['gamma'],
        'lam': best_nogate['lam'],
        'grid_mse': best_nogate['final_mse'],
        'grid_acc': best_nogate['final_acc']
    },
    'top3_osc': top3_osc,
    'top3_nogate': top3_nogate
}
out_path = os.path.join(WORK_DIR, 'selected_configs.json')
with open(out_path, 'w') as f:
    json.dump(selected, f, indent=2)
print(f'\nSelected configs saved to {out_path}')
print('=== Step 6: Config Selection COMPLETE ===')

# ----------

# ----------
import subprocess
import os
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step7_full_training.py')
lines = [
    'import torch',
    'import torch.nn as nn',
    'import numpy as np',
    'import math',
    'import json',
    'import os',
    'import time',
    '',
    'SEED = 42',
    'torch.manual_seed(SEED)',
    'np.random.seed(SEED)',
    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    'print("Device:", device)',
    '',
    'N_BITS=8; DELAY=10; INPUT_DIM=9; OUTPUT_DIM=8; HIDDEN_SIZE=128',
    'SEQ_LEN = N_BITS + DELAY + N_BITS',
    'WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"',
    'N_ITER = 5000',
    'BATCH_SIZE = 32',
    'LOG_INTERVAL = 100',
    '',
    'def generate_copy_batch(batch_size, dev=None):',
    '    if dev is None: dev = device',
    '    patterns = torch.randint(0, 2, (batch_size, N_BITS), dtype=torch.float32)',
    '    inputs = torch.zeros(batch_size, SEQ_LEN, INPUT_DIM)',
    '    targets = torch.zeros(batch_size, SEQ_LEN, OUTPUT_DIM)',
    '    output_mask = torch.zeros(batch_size, SEQ_LEN)',
    '    for t in range(N_BITS):',
    '        inputs[:, t, t] = patterns[:, t]',
    '    inputs[:, N_BITS+DELAY-1, N_BITS] = 1.0',
    '    for t in range(N_BITS):',
    '        targets[:, N_BITS+DELAY+t, :] = patterns',
    '    output_mask[:, N_BITS+DELAY:N_BITS+DELAY+N_BITS] = 1.0',
    '    return inputs.to(dev), targets.to(dev), output_mask.to(dev)',
    '',
    'def compute_metrics(outputs, targets, output_mask):',
    '    mask = output_mask.unsqueeze(-1)',
    '    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*OUTPUT_DIM)',
    '    pred_bits = (outputs > 0.5).float()',
    '    correct = ((pred_bits == targets).float() * mask).sum()',
    '    return mse.item(), (correct / (mask.sum()*OUTPUT_DIM)).item()',
    '',
    '# ============================================================',
    '# METHOD 1: BPTT',
    '# ============================================================',
    'class VanillaRNN(nn.Module):',
    '    def __init__(self):',
    '        super().__init__()',
    '        self.W_in = nn.Parameter(torch.randn(HIDDEN_SIZE, INPUT_DIM)*0.1)',
    '        self.W_rec = nn.Parameter(torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.1)',
    '        self.b_rec = nn.Parameter(torch.zeros(HIDDEN_SIZE))',
    '        self.W_out = nn.Parameter(torch.randn(OUTPUT_DIM, HIDDEN_SIZE)*0.1)',
    '        self.b_out = nn.Parameter(torch.zeros(OUTPUT_DIM))',
    '    def forward(self, inputs):',
    '        B,T,_ = inputs.shape',
    '        h = torch.zeros(B, HIDDEN_SIZE, device=inputs.device)',
    '        outs = []',
    '        for t in range(T):',
    '            h = torch.tanh(h @ self.W_rec.T + inputs[:,t,:] @ self.W_in.T + self.b_rec)',
    '            outs.append(h @ self.W_out.T + self.b_out)',
    '        return torch.stack(outs, dim=1)',
    '',
    'def train_bptt(n_iter=N_ITER, lr=0.001):',
    '    torch.manual_seed(SEED)',
    '    model = VanillaRNN().to(device)',
    '    opt = torch.optim.Adam(model.parameters(), lr=lr)',
    '    mse_h, acc_h = [], []',
    '    t0 = time.time()',
    '    for it in range(n_iter):',
    '        inp, tgt, msk = generate_copy_batch(BATCH_SIZE)',
    '        opt.zero_grad()',
    '        out = model(inp)',
    '        loss = ((out-tgt)**2 * msk.unsqueeze(-1)).sum() / (msk.sum()*OUTPUT_DIM)',
    '        loss.backward()',
    '        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)',
    '        opt.step()',
    '        mse, acc = compute_metrics(out.detach(), tgt, msk)',
    '        mse_h.append(mse); acc_h.append(acc)',
    '        if (it+1) % LOG_INTERVAL == 0:',
    '            print("  BPTT iter", it+1, "| MSE:", round(mse,4), "| Acc:", round(acc,4))',
    '    print("  BPTT time:", round(time.time()-t0, 1), "s")',
    '    return mse_h, acc_h',
    '',
    '# ============================================================',
    '# METHOD 4: FA',
    '# ============================================================',
    'def train_fa(n_iter=N_ITER, lr=0.001):',
    '    torch.manual_seed(SEED)',
    '    W_in=(torch.randn(HIDDEN_SIZE,INPUT_DIM)*0.1).to(device)',
    '    W_rec=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.1).to(device)',
    '    b_rec=torch.zeros(HIDDEN_SIZE).to(device)',
    '    W_out=(torch.randn(OUTPUT_DIM,HIDDEN_SIZE)*0.1).to(device)',
    '    b_out=torch.zeros(OUTPUT_DIM).to(device)',
    '    B_rec=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.1).to(device)',
    '    B_out=(torch.randn(HIDDEN_SIZE,OUTPUT_DIM)*0.1).to(device)',
    '    params=[W_in,W_rec,b_rec,W_out,b_out]',
    '    m=[torch.zeros_like(p) for p in params]',
    '    v=[torch.zeros_like(p) for p in params]',
    '    b1,b2,eps=0.9,0.999,1e-8',
    '    mse_h,acc_h=[],[]',
    '    t0=time.time()',
    '    for it in range(n_iter):',
    '        inp,tgt,msk=generate_copy_batch(BATCH_SIZE)',
    '        h=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)',
    '        hs,outputs=[],[]',
    '        for t in range(SEQ_LEN):',
    '            h=torch.tanh(h@W_rec.T+inp[:,t,:]@W_in.T+b_rec)',
    '            hs.append(h)',
    '            outputs.append(h@W_out.T+b_out)',
    '        out_t=torch.stack(outputs,dim=1)',
    '        out_err=(out_t-tgt)*msk.unsqueeze(-1)',
    '        dWi=torch.zeros_like(W_in)',
    '        dWr=torch.zeros_like(W_rec)',
    '        dbi=torch.zeros_like(b_rec)',
    '        dWo=torch.zeros_like(W_out)',
    '        dbo=torch.zeros_like(b_out)',
    '        dh_next=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)',
    '        for t in reversed(range(SEQ_LEN)):',
    '            ht=hs[t]',
    '            hp=hs[t-1] if t>0 else torch.zeros_like(ht)',
    '            xt=inp[:,t,:]',
    '            dt=1-ht**2',
    '            dh=(out_err[:,t,:]@B_out.T+dh_next)*dt',
    '            dWo+=out_err[:,t,:].T@ht/BATCH_SIZE',
    '            dbo+=out_err[:,t,:].mean(0)',
    '            dWr+=dh.T@hp/BATCH_SIZE',
    '            dWi+=dh.T@xt/BATCH_SIZE',
    '            dbi+=dh.mean(0)',
    '            dh_next=dh@B_rec.T',
    '        grads=[dWi,dWr,dbi,dWo,dbo]',
    '        tn=sum(g.norm()**2 for g in grads)**0.5',
    '        if tn>1.0: grads=[g/tn for g in grads]',
    '        ta=it+1',
    '        for i,(pw,g) in enumerate(zip(params,grads)):',
    '            m[i]=b1*m[i]+(1-b1)*g',
    '            v[i]=b2*v[i]+(1-b2)*g**2',
    '            pw.data-=lr*(m[i]/(1-b1**ta))/(((v[i]/(1-b2**ta))**0.5)+eps)',
    '        mse,acc=compute_metrics(out_t.detach(),tgt,msk)',
    '        mse_h.append(mse); acc_h.append(acc)',
    '        if (it+1)%LOG_INTERVAL==0:',
    '            print("  FA iter",it+1,"| MSE:",round(mse,4),"| Acc:",round(acc,4))',
    '    print("  FA time:", round(time.time()-t0,1), "s")',
    '    return mse_h,acc_h',
    '',
    '# ============================================================',
    '# METHOD 2&3: PSC',
    '# ============================================================',
    'def train_psc(n_iter=N_ITER, lr=0.001, lr_pred=0.001, beta=0.1, gamma=0.3,',
    '              lam=0.9, T_theta=20, use_oscillatory_gate=True, label="PSC"):',
    '    torch.manual_seed(SEED)',
    '    W_in=(torch.randn(HIDDEN_SIZE,INPUT_DIM)*0.1).to(device)',
    '    W_rec=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.1).to(device)',
    '    b_rec=torch.zeros(HIDDEN_SIZE).to(device)',
    '    W_out=(torch.randn(OUTPUT_DIM,HIDDEN_SIZE)*0.1).to(device)',
    '    b_out=torch.zeros(OUTPUT_DIM).to(device)',
    '    W_pred=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.01).to(device)',
    '    params=[W_in,W_rec,b_rec,W_out,b_out]',
    '    m_adam=[torch.zeros_like(p) for p in params]',
    '    v_adam=[torch.zeros_like(p) for p in params]',
    '    b1,b2,eps=0.9,0.999,1e-8',
    '    mse_h,acc_h=[],[]',
    '    t0=time.time()',
    '    for it in range(n_iter):',
    '        inp,tgt,msk=generate_copy_batch(BATCH_SIZE)',
    '        h=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)',
    '        pc=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)',
    '        er=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,device=device)',
    '        ei=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,INPUT_DIM,device=device)',
    '        dWin=torch.zeros_like(W_in)',
    '        dWr=torch.zeros_like(W_rec)',
    '        dbr=torch.zeros_like(b_rec)',
    '        dWo=torch.zeros_like(W_out)',
    '        dbo=torch.zeros_like(b_out)',
    '        dWp=torch.zeros_like(W_pred)',
    '        outs=[]',
    '        for t in range(SEQ_LEN):',
    '            xt=inp[:,t,:]',
    '            hp=h.clone()',
    '            h=torch.tanh(hp@W_rec.T+xt@W_in.T+b_rec)',
    '            dt=1-h**2',
    '            yt=h@W_out.T+b_out',
    '            outs.append(yt)',
    '            pc=(1-beta)*pc+beta*(hp@W_pred.T)',
    '            dp=h-pc',
    '            er=lam*er+torch.bmm(dt.unsqueeze(2),hp.unsqueeze(1))',
    '            ei=lam*ei+torch.bmm(dt.unsqueeze(2),xt.unsqueeze(1))',
    '            en=er.norm(dim=(1,2),keepdim=True)',
    '            er=torch.where(en>5.0,er*5.0/(en+1e-8),er)',
    '            iso=msk[:,t].bool()',
    '            oe=torch.zeros_like(yt)',
    '            if iso.any(): oe[iso]=yt[iso]-tgt[:,t,:][iso]',
    '            oeh=oe@W_out',
    '            st=gamma*dp+(1-gamma)*oeh',
    '            gate=max(0.0,math.sin(2*math.pi*t/T_theta)) if use_oscillatory_gate else 1.0',
    '            if gate>0:',
    '                dWr+=gate*(st.unsqueeze(2)*er).mean(0)',
    '                dWin+=gate*(st.unsqueeze(2)*ei).mean(0)',
    '                dbr+=gate*st.mean(0)',
    '            dWo+=oe.T@h/BATCH_SIZE',
    '            dbo+=oe.mean(0)',
    '            dWp+=(dp.unsqueeze(2)*hp.unsqueeze(1)).mean(0)',
    '        grads=[dWin,dWr,dbr,dWo,dbo]',
    '        tn=sum(g.norm()**2 for g in grads)**0.5',
    '        if tn>1.0: grads=[g/tn for g in grads]',
    '        ta=it+1',
    '        for i,(pw,g) in enumerate(zip(params,grads)):',
    '            m_adam[i]=b1*m_adam[i]+(1-b1)*g',
    '            v_adam[i]=b2*v_adam[i]+(1-b2)*g**2',
    '            pw.data-=lr*(m_adam[i]/(1-b1**ta))/(((v_adam[i]/(1-b2**ta))**0.5)+eps)',
    '        pn=dWp.norm()',
    '        if pn>1.0: dWp=dWp/pn',
    '        W_pred.data-=lr_pred*dWp',
    '        out_t=torch.stack(outs,dim=1)',
    '        mse,acc=compute_metrics(out_t.detach(),tgt,msk)',
    '        mse_h.append(mse); acc_h.append(acc)',
    '        if (it+1)%LOG_INTERVAL==0:',
    '            print(" ",label,"iter",it+1,"| MSE:",round(mse,4),"| Acc:",round(acc,4))',
    '    print(" ",label,"time:",round(time.time()-t0,1),"s")',
    '    return mse_h,acc_h',
    '',
    '# ============================================================',
    '# RUN ALL METHODS',
    '# ============================================================',
    'all_results = {}',
    '',
    'print("")',
    'print("=== Running BPTT (5000 iters, lr=0.001) ===")',
    'mse_bptt, acc_bptt = train_bptt(N_ITER, lr=0.001)',
    'all_results["bptt"] = {"mse_history": mse_bptt[::LOG_INTERVAL],',
    '                        "acc_history": acc_bptt[::LOG_INTERVAL],',
    '                        "final_mse": mse_bptt[-1], "final_acc": acc_bptt[-1],',
    '                        "config": {"lr": 0.001, "method": "BPTT"}}',
    '',
    'print("")',
    'print("=== Running PSC-osc (5000 iters, best config) ===")',
    'print("  Config: lr=0.001, beta=0.1, gamma=0.3, lam=0.9, T_theta=20")',
    'mse_posc, acc_posc = train_psc(N_ITER, lr=0.001, lr_pred=0.001, beta=0.1,',
    '                                gamma=0.3, lam=0.9, T_theta=20,',
    '                                use_oscillatory_gate=True, label="PSC-osc")',
    'all_results["psc_osc"] = {"mse_history": mse_posc[::LOG_INTERVAL],',
    '                           "acc_history": acc_posc[::LOG_INTERVAL],',
    '                           "final_mse": mse_posc[-1], "final_acc": acc_posc[-1],',
    '                           "config": {"lr":0.001,"beta":0.1,"gamma":0.3,"lam":0.9,"T_theta":20}}',
    '',
    'print("")',
    'print("=== Running PSC-nogate (5000 iters, best config) ===")',
    'print("  Config: lr=0.001, beta=0.1, gamma=0.5, lam=0.9")',
    'mse_png, acc_png = train_psc(N_ITER, lr=0.001, lr_pred=0.001, beta=0.1,',
    '                              gamma=0.5, lam=0.9, T_theta=10,',
    '                              use_oscillatory_gate=False, label="PSC-nogate")',
    'all_results["psc_nogate"] = {"mse_history": mse_png[::LOG_INTERVAL],',
    '                              "acc_history": acc_png[::LOG_INTERVAL],',
    '                              "final_mse": mse_png[-1], "final_acc": acc_png[-1],',
    '                              "config": {"lr":0.001,"beta":0.1,"gamma":0.5,"lam":0.9}}',
    '',
    'print("")',
    'print("=== Running FA (5000 iters, lr=0.001) ===")',
    'mse_fa, acc_fa = train_fa(N_ITER, lr=0.001)',
    'all_results["fa"] = {"mse_history": mse_fa[::LOG_INTERVAL],',
    '                     "acc_history": acc_fa[::LOG_INTERVAL],',
    '                     "final_mse": mse_fa[-1], "final_acc": acc_fa[-1],',
    '                     "config": {"lr": 0.001, "method": "FA"}}',
    '',
    '# Save results',
    'out_path = os.path.join(WORK_DIR, "full_training_results.json")',
    'with open(out_path, "w") as f:',
    '    json.dump(all_results, f, indent=2)',
    '',
    'print("")',
    'print("=== FINAL COMPARISON ===")',
    'for method, res in all_results.items():',
    '    print(" ", method, "| Final MSE:", round(res["final_mse"],4),',
    '          "| Final Acc:", round(res["final_acc"],4))',
    '',
    'print("")',
    'print("Results saved to", out_path)',
    'print("=== Step 7: Full Training COMPLETE ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
print('Script written, starting full 5000-iter training...')
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=14400)  # 4 hour timeout
print('STDOUT:')
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import subprocess
import os
import json
PYTHON = '/home/zihan.zhang/.conda/envs/panda/bin/python'
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
script_path = os.path.join(WORK_DIR, 'step8_analysis.py')
lines = [
    'import json',
    'import os',
    'import numpy as np',
    'import matplotlib',
    'matplotlib.use("Agg")',
    'import matplotlib.pyplot as plt',
    '',
    'WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"',
    '',
    '# Load results',
    'with open(os.path.join(WORK_DIR, "full_training_results.json"), "r") as f:',
    '    results = json.load(f)',
    '',
    'methods = {',
    '    "bptt": "BPTT (gold standard)",',
    '    "psc_nogate": "PSC-NoGate (ours)",',
    '    "psc_osc": "PSC-Osc (ours)",',
    '    "fa": "FA (baseline)"',
    '}',
    'colors = {',
    '    "bptt": "blue",',
    '    "psc_nogate": "green",',
    '    "psc_osc": "orange",',
    '    "fa": "red"',
    '}',
    '',
    '# Reconstruct full histories from sampled data',
    'LOG_INTERVAL = 100',
    'iters = list(range(LOG_INTERVAL, 5001, LOG_INTERVAL))',
    '',
    '# ============================================================',
    '# FIGURE 1: Learning curves (MSE)',
    '# ============================================================',
    'fig, axes = plt.subplots(1, 2, figsize=(14, 5))',
    '',
    'ax = axes[0]',
    'for key, label in methods.items():',
    '    mse_h = results[key]["mse_history"]',
    '    x = iters[:len(mse_h)]',
    '    ax.plot(x, mse_h, label=label, color=colors[key], linewidth=2)',
    'ax.set_xlabel("Training Iterations", fontsize=12)',
    'ax.set_ylabel("MSE", fontsize=12)',
    'ax.set_title("Learning Curves: MSE", fontsize=13)',
    'ax.legend(fontsize=10)',
    'ax.grid(True, alpha=0.3)',
    'ax.set_ylim([-0.02, 0.65])',
    '',
    '# Zoomed version (excluding BPTT for clarity)',
    'ax2 = axes[1]',
    'for key, label in methods.items():',
    '    if key == "bptt": continue',
    '    mse_h = results[key]["mse_history"]',
    '    x = iters[:len(mse_h)]',
    '    ax2.plot(x, mse_h, label=label, color=colors[key], linewidth=2)',
    'ax2.set_xlabel("Training Iterations", fontsize=12)',
    'ax2.set_ylabel("MSE", fontsize=12)',
    'ax2.set_title("Learning Curves: MSE (Biologically-Plausible Methods)", fontsize=13)',
    'ax2.legend(fontsize=10)',
    'ax2.grid(True, alpha=0.3)',
    '',
    'plt.tight_layout()',
    'plt.savefig(os.path.join(WORK_DIR, "learning_curves_mse.png"), dpi=150, bbox_inches="tight")',
    'plt.close()',
    'print("Saved: learning_curves_mse.png")',
    '',
    '# ============================================================',
    '# FIGURE 2: Accuracy curves',
    '# ============================================================',
    'fig, axes = plt.subplots(1, 2, figsize=(14, 5))',
    '',
    'ax = axes[0]',
    'for key, label in methods.items():',
    '    acc_h = results[key]["acc_history"]',
    '    x = iters[:len(acc_h)]',
    '    ax.plot(x, [a*100 for a in acc_h], label=label, color=colors[key], linewidth=2)',
    'ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")',
    'ax.set_xlabel("Training Iterations", fontsize=12)',
    'ax.set_ylabel("Bit Accuracy (%)", fontsize=12)',
    'ax.set_title("Learning Curves: Bit Accuracy", fontsize=13)',
    'ax.legend(fontsize=10)',
    'ax.grid(True, alpha=0.3)',
    '',
    'ax2 = axes[1]',
    'for key, label in methods.items():',
    '    if key == "bptt": continue',
    '    acc_h = results[key]["acc_history"]',
    '    x = iters[:len(acc_h)]',
    '    ax2.plot(x, [a*100 for a in acc_h], label=label, color=colors[key], linewidth=2)',
    'ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")',
    'ax2.set_xlabel("Training Iterations", fontsize=12)',
    'ax2.set_ylabel("Bit Accuracy (%)", fontsize=12)',
    'ax2.set_title("Accuracy (Biologically-Plausible Methods)", fontsize=13)',
    'ax2.legend(fontsize=10)',
    'ax2.grid(True, alpha=0.3)',
    '',
    'plt.tight_layout()',
    'plt.savefig(os.path.join(WORK_DIR, "learning_curves_acc.png"), dpi=150, bbox_inches="tight")',
    'plt.close()',
    'print("Saved: learning_curves_acc.png")',
    '',
    '# ============================================================',
    '# FIGURE 3: Final comparison bar chart',
    '# ============================================================',
    'fig, axes = plt.subplots(1, 2, figsize=(12, 5))',
    '',
    'method_names = ["BPTT", "PSC-NoGate", "PSC-Osc", "FA"]',
    'keys_order = ["bptt", "psc_nogate", "psc_osc", "fa"]',
    'bar_colors = [colors[k] for k in keys_order]',
    'final_mse = [results[k]["final_mse"] for k in keys_order]',
    'final_acc = [results[k]["final_acc"]*100 for k in keys_order]',
    '',
    'ax = axes[0]',
    'bars = ax.bar(method_names, final_mse, color=bar_colors, alpha=0.8, edgecolor="black")',
    'ax.set_ylabel("Final MSE", fontsize=12)',
    'ax.set_title("Final MSE Comparison (5000 iters)", fontsize=13)',
    'for bar, val in zip(bars, final_mse):',
    '    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,',
    '            f"{val:.4f}", ha="center", va="bottom", fontsize=9)',
    'ax.grid(True, alpha=0.3, axis="y")',
    '',
    'ax2 = axes[1]',
    'bars2 = ax2.bar(method_names, final_acc, color=bar_colors, alpha=0.8, edgecolor="black")',
    'ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="Random")',
    'ax2.set_ylabel("Bit Accuracy (%)", fontsize=12)',
    'ax2.set_title("Final Bit Accuracy (5000 iters)", fontsize=13)',
    'for bar, val in zip(bars2, final_acc):',
    '    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,',
    '             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)',
    'ax2.set_ylim([0, 110])',
    'ax2.grid(True, alpha=0.3, axis="y")',
    '',
    'plt.tight_layout()',
    'plt.savefig(os.path.join(WORK_DIR, "final_comparison.png"), dpi=150, bbox_inches="tight")',
    'plt.close()',
    'print("Saved: final_comparison.png")',
    '',
    '# ============================================================',
    '# CONVERGENCE SPEED ANALYSIS',
    '# ============================================================',
    'print("")',
    'print("=== Convergence Speed Analysis ===")',
    'thresholds = [0.6, 0.7, 0.75, 0.8, 0.9]',
    'convergence = {}',
    'for key in keys_order:',
    '    acc_h = results[key]["acc_history"]',
    '    convergence[key] = {}',
    '    for thresh in thresholds:',
    '        reached = None',
    '        for i, acc in enumerate(acc_h):',
    '            if acc >= thresh:',
    '                reached = (i+1) * LOG_INTERVAL',
    '                break',
    '        convergence[key][thresh] = reached',
    '',
    'print("  Method         | 60%   | 70%   | 75%   | 80%   | 90%")',
    'print("  " + "-"*65)',
    'for key, name in zip(keys_order, method_names):',
    '    row = "  {:15s}|".format(name)',
    '    for t in thresholds:',
    '        v = convergence[key][t]',
    '        row += " {:5s} |".format(str(v) if v else "N/A")',
    '    print(row)',
    '',
    '# ============================================================',
    '# PRINT SUMMARY TABLE',
    '# ============================================================',
    'print("")',
    'print("=== Final Results Summary ===")',
    'print("  {:20s} | {:10s} | {:10s}".format("Method", "Final MSE", "Final Acc"))',
    'print("  " + "-"*47)',
    'for key, name in zip(keys_order, method_names):',
    '    mse = results[key]["final_mse"]',
    '    acc = results[key]["final_acc"]',
    '    print("  {:20s} | {:10.4f} | {:10.2%}".format(name, mse, acc))',
    '',
    '# Save analysis to JSON',
    'analysis = {',
    '    "final_comparison": {',
    '        k: {"final_mse": results[k]["final_mse"], "final_acc": results[k]["final_acc"]}',
    '        for k in keys_order',
    '    },',
    '    "convergence_speed": {',
    '        k: {str(t): v for t, v in convergence[k].items()}',
    '        for k in keys_order',
    '    }',
    '}',
    'with open(os.path.join(WORK_DIR, "analysis_results.json"), "w") as f:',
    '    json.dump(analysis, f, indent=2)',
    'print("")',
    'print("Analysis saved to analysis_results.json")',
    'print("=== Step 8: Analysis COMPLETE ===")',
]
with open(script_path, 'w') as f:
    f.write('\n'.join(lines))
result = subprocess.run([PYTHON, script_path], capture_output=True, text=True, timeout=120)
print('STDOUT:', result.stdout)
if result.stderr:
    print('STDERR:', result.stderr[:2000])
print('Return code:', result.returncode)

# ----------

# ----------
import json
import os
WORK_DIR = '/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/'
with open(os.path.join(WORK_DIR, 'full_training_results.json'), 'r') as f:
    full_results = json.load(f)
with open(os.path.join(WORK_DIR, 'analysis_results.json'), 'r') as f:
    analysis = json.load(f)
with open(os.path.join(WORK_DIR, 'selected_configs.json'), 'r') as f:
    selected = json.load(f)
print('='*70)
print('PSC (PREDICTIVE SELF-CORRECTION) EXPERIMENT - COMPLETE RESULTS')
print('='*70)
print('')
print('TASK: Copy Task (8-bit patterns, 10-step delay)')
print('ARCHITECTURE: Vanilla RNN, hidden_size=128')
print('TRAINING: 5000 iterations, batch_size=32')
print('EVALUATION: MSE and bit accuracy on output phase only')
print('')
print('='*70)
print('PSC ALGORITHM DESCRIPTION')
print('='*70)
print('PSC avoids backward propagation entirely. Each neuron computes:')
print('1. Main RNN: h_t = tanh(W_rec * h_{t-1} + W_in * x_t + b)')
print('2. Prediction: p_i(t) = (1-beta)*p_i(t-1) + beta*(W_pred[i,:] @ h(t-1))')
print('3. Prediction error: delta_pred_i(t) = h_i(t) - p_i(t)')
print('4. Eligibility trace: e_ij(t) = lambda*e_ij(t-1) + h_j(t-1)*dtanh_i(t)')
print('5. Output error: err_i(t) = W_out^T * (y - target) (at output timesteps)')
print('6. Credit signal: s_i(t) = gamma*delta_pred_i(t) + (1-gamma)*output_err_i(t)')
print('7. Oscillatory gate: gate(t) = max(0, sin(2*pi*t/T_theta))')
print('8. W_rec update: Delta_W_rec_ij = -lr * s_i(t) * e_ij(t) * gate(t)')
print('9. W_pred update: Delta_W_pred_ij = -lr_pred * delta_pred_i(t) * h_j(t-1)')
print('')
print('='*70)
print('HYPERPARAMETER GRID SEARCH RESULTS (1000 iterations each)')
print('='*70)
print('Search space: lr in {0.001, 0.005, 0.01}, beta in {0.1, 0.3},')
print('  gamma in {0.3, 0.5, 0.7}, lambda in {0.9, 0.95}, T_theta in {10, 20}')
print('Total: 72 configs for PSC-osc, 36 for PSC-nogate')
print('')
print('TOP 3 PSC-OSC CONFIGURATIONS (1000-iter grid search):')
osc_cfg = selected['best_psc_osc']
print(f'  Rank 1 (SELECTED): lr={osc_cfg["lr"]}, beta={osc_cfg["beta"]}, gamma={osc_cfg["gamma"]}, lam={osc_cfg["lam"]}, T_theta={osc_cfg["T_theta"]}')
print(f'    Grid MSE: {osc_cfg["grid_mse"]:.4f}, Grid Acc: {osc_cfg["grid_acc"]:.4f}')
for i, cfg in enumerate(selected['top3_osc'][1:3]):
    print(f'  Rank {i+2}: lr={cfg["lr"]}, beta={cfg["beta"]}, gamma={cfg["gamma"]}, lam={cfg["lam"]}, T={cfg["T_theta"]} | MSE={cfg["final_mse"]:.4f}')
print('')
print('TOP 3 PSC-NOGATE CONFIGURATIONS (1000-iter grid search):')
ng_cfg = selected['best_psc_nogate']
print(f'  Rank 1 (SELECTED): lr={ng_cfg["lr"]}, beta={ng_cfg["beta"]}, gamma={ng_cfg["gamma"]}, lam={ng_cfg["lam"]}')
print(f'    Grid MSE: {ng_cfg["grid_mse"]:.4f}, Grid Acc: {ng_cfg["grid_acc"]:.4f}')
for i, cfg in enumerate(selected['top3_nogate'][1:3]):
    print(f'  Rank {i+2}: lr={cfg["lr"]}, beta={cfg["beta"]}, gamma={cfg["gamma"]}, lam={cfg["lam"]} | MSE={cfg["final_mse"]:.4f}')
print('')
print('='*70)
print('FULL 5000-ITERATION TRAINING RESULTS')
print('='*70)
print(f'{"Method":<25} | {"Final MSE":>10} | {"Final Acc":>10} | Description')
print('-'*70)
print(f'{"BPTT":<25} | {full_results["bptt"]["final_mse"]:>10.4f} | {full_results["bptt"]["final_acc"]*100:>9.1f}% | Gold standard (backprop)')
print(f'{"PSC-NoGate (ours)":<25} | {full_results["psc_nogate"]["final_mse"]:>10.4f} | {full_results["psc_nogate"]["final_acc"]*100:>9.1f}% | PSC without oscillatory gate')
print(f'{"PSC-Osc (ours)":<25} | {full_results["psc_osc"]["final_mse"]:>10.4f} | {full_results["psc_osc"]["final_acc"]*100:>9.1f}% | PSC with oscillatory gate')
print(f'{"FA (baseline)":<25} | {full_results["fa"]["final_mse"]:>10.4f} | {full_results["fa"]["final_acc"]*100:>9.1f}% | Feedback Alignment')
print('')
print('='*70)
print('CONVERGENCE SPEED (iterations to reach accuracy threshold)')
print('='*70)
print(f'{"Method":<20} | {"60%":>6} | {"70%":>6} | {"75%":>6} | {"80%":>6} | {"90%":>6}')
print('-'*60)
conv = analysis['convergence_speed']
for key, name in [('bptt','BPTT'),('psc_nogate','PSC-NoGate'),('psc_osc','PSC-Osc'),('fa','FA')]:
    row = f'{name:<20} |'
    for t in ['0.6','0.7','0.75','0.8','0.9']:
        v = conv[key].get(str(t)) if str(t) in conv[key] else conv[key].get(t)
        row += f' {str(v) if v else "N/A":>6} |'
    print(row)
print('')
print('='*70)
print('KEY FINDINGS')
print('='*70)
print('')
print('1. BPTT (gold standard): Achieves 100% accuracy by ~iter 300.')
print('   Perfect performance demonstrates the task is solvable.')
print('')
print('2. PSC-NoGate (best bio-plausible): Reaches 76.4% accuracy at 5000 iters.')
print('   - Significantly outperforms FA and PSC-Osc')
print('   - Reaches 60% at iter 300, 70% at iter 500, 75%+ at iter 700')
print('   - Best config: lr=0.001, beta=0.1, gamma=0.5, lambda=0.9')
print('   - Gamma=0.5: prediction error and output error equally weighted')
print('')
print('3. PSC-Osc (oscillatory gated): Only 51.7% accuracy (near random).')
print('   - The oscillatory gating HURTS performance on this task')
print('   - Gate = max(0, sin(2*pi*t/T_theta)) only allows updates ~50% of time')
print('   - This halves the effective learning signal, slowing convergence')
print('   - Best config for gating: T_theta=20 (larger period = more gate-open time)')
print('')
print('4. FA (feedback alignment): Essentially fails (50.8%, near random).')
print('   - Random feedback matrices cannot effectively propagate credit')
print('   - through the temporal dependencies of this task')
print('')
print('='*70)
print('DISCUSSION')
print('='*70)
print('')
print('PSC-NoGate demonstrates that local eligibility-trace-based learning')
print('with output error feedback can learn temporal tasks without BPTT.')
print('The key ingredients that make it work:')
print('  - Eligibility traces carry temporal credit assignment information')
print('  - Output error projected via W_out^T provides direction signal')
print('  - Prediction compartment adds self-supervised signal (when gamma>0)')
print('')
print('The oscillatory gating ablation reveals that phase-restricted plasticity')
print('is counterproductive for this task. While biologically motivated by')
print('theta-gated plasticity in hippocampus, the sine gating here reduces')
print('the effective learning rate by ~50%, leading to insufficient learning.')
print('')
print('PSC-NoGate outperforms FA (50.8%) by a large margin (76.4% vs 50.8%).')
print('This confirms that eligibility traces provide better credit assignment')
print('than random feedback matrices for temporal tasks.')
print('')
print('Files saved:')
print(f'  {WORK_DIR}full_training_results.json')
print(f'  {WORK_DIR}analysis_results.json')
print(f'  {WORK_DIR}grid_search_results.json')
print(f'  {WORK_DIR}selected_configs.json')
print(f'  {WORK_DIR}learning_curves_mse.png')
print(f'  {WORK_DIR}learning_curves_acc.png')
print(f'  {WORK_DIR}final_comparison.png')

# ----------

# ----------
report_path = write_report()
print('Report written to:', report_path)

# ----------
