import torch
import torch.nn as nn
import numpy as np
import json
import math
import os
import time
from itertools import product

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"
os.makedirs(WORK_DIR, exist_ok=True)
print("Working directory:", WORK_DIR)

# Task configuration
N_BITS = 8
DELAY = 10
INPUT_DIM = N_BITS + 1
OUTPUT_DIM = N_BITS
HIDDEN_SIZE = 128
SEQ_LEN = N_BITS + DELAY + N_BITS

print("")
print("Task Configuration:")
print("  N_BITS=", N_BITS, " DELAY=", DELAY, " SEQ_LEN=", SEQ_LEN)
print("  INPUT_DIM=", INPUT_DIM, " OUTPUT_DIM=", OUTPUT_DIM, " HIDDEN_SIZE=", HIDDEN_SIZE)

def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):
    if dev is None:
        dev = device
    seq_len = n_bits + delay + n_bits
    input_dim = n_bits + 1
    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)
    inputs = torch.zeros(batch_size, seq_len, input_dim)
    targets = torch.zeros(batch_size, seq_len, n_bits)
    output_mask = torch.zeros(batch_size, seq_len)
    for t in range(n_bits):
        inputs[:, t, t] = patterns[:, t]
    inputs[:, n_bits + delay - 1, n_bits] = 1.0
    output_start = n_bits + delay
    for t in range(n_bits):
        targets[:, output_start + t, :] = patterns
    output_mask[:, output_start:output_start + n_bits] = 1.0
    return inputs.to(dev), targets.to(dev), output_mask.to(dev)

# Test the data generator
batch_size = 4
torch.manual_seed(SEED)
inputs, targets, output_mask = generate_copy_batch(batch_size)

print("")
print("Batch shapes:")
print("  inputs:", inputs.shape)
print("  targets:", targets.shape)
print("  output_mask:", output_mask.shape)

print("")
print("Sample sequence (batch item 0):")
print("  Pattern to copy:", targets[0, N_BITS+DELAY, :].cpu().numpy())
print("  Input at t=0:", inputs[0, 0, :N_BITS].cpu().numpy())
print("  Input at t=3:", inputs[0, 3, :N_BITS].cpu().numpy())
print("  Input at t=7:", inputs[0, 7, :N_BITS].cpu().numpy())
print("  Recall cue at t=17:", inputs[0, N_BITS+DELAY-1, N_BITS].item())
print("  Target at t=18:", targets[0, N_BITS+DELAY, :].cpu().numpy())
print("  Output mask:", output_mask[0].cpu().numpy())

# Verify correctness
for i in range(batch_size):
    pat_from_input = torch.zeros(N_BITS)
    for t in range(N_BITS):
        pat_from_input[t] = inputs[i, t, t].cpu()
    for t in range(N_BITS):
        tgt_t = targets[i, N_BITS+DELAY+t, :].cpu()
        assert torch.allclose(tgt_t, pat_from_input), "Mismatch at batch " + str(i)
print("[PASS] Input/target consistency check passed")

# Test batch dimensions
for trial in range(3):
    inp, tgt, msk = generate_copy_batch(32)
    assert inp.shape == (32, SEQ_LEN, INPUT_DIM)
    assert tgt.shape == (32, SEQ_LEN, OUTPUT_DIM)
    assert msk.shape == (32, SEQ_LEN)
    assert msk[:, :N_BITS+DELAY].sum().item() == 0
    assert msk[:, N_BITS+DELAY:].sum().item() == 32 * N_BITS
print("[PASS] Batch shape/mask tests passed")

print("")
print("=== Step 1: Environment Setup COMPLETE ===")
print("  PyTorch:", torch.__version__)
print("  CUDA:", torch.cuda.is_available(), "Device:", device)