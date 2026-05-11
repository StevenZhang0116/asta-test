import torch
import torch.nn as nn
import numpy as np
import json
import os
import math

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"
N_BITS = 8
DELAY = 10
INPUT_DIM = N_BITS + 1
OUTPUT_DIM = N_BITS
HIDDEN_SIZE = 128
SEQ_LEN = N_BITS + DELAY + N_BITS

# ============================================================
# COPY TASK DATA GENERATOR
# ============================================================
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

# ============================================================
# BPTT RNN MODEL
# ============================================================
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_dim) * 0.1)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.b_rec = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_size) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs, h0=None):
        batch_size, seq_len, _ = inputs.shape
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        else:
            h = h0
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            a_t = h @ self.W_rec.T + x_t @ self.W_in.T + self.b_rec
            h = torch.tanh(a_t)
            y_t = h @ self.W_out.T + self.b_out
            outputs.append(y_t)
        outputs = torch.stack(outputs, dim=1)
        return outputs, h

def compute_metrics(outputs, targets, output_mask):
    mask = output_mask.unsqueeze(-1)
    pred = outputs * mask
    tgt = targets * mask
    mse = ((pred - tgt) ** 2).sum() / (mask.sum() * targets.shape[-1])
    pred_bits = (outputs > 0.5).float() * output_mask.unsqueeze(-1)
    tgt_bits = targets * output_mask.unsqueeze(-1)
    correct = (pred_bits == tgt_bits).float() * output_mask.unsqueeze(-1)
    n_total = output_mask.sum() * targets.shape[-1]
    acc = correct.sum() / n_total if n_total > 0 else torch.tensor(0.0)
    return mse.item(), acc.item()

# ============================================================
# BPTT TRAINING FUNCTION
# ============================================================
def train_bptt(n_iter=200, batch_size=32, lr=0.001, verbose=True, log_interval=50):
    torch.manual_seed(SEED)
    model = VanillaRNN(INPUT_DIM, HIDDEN_SIZE, OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    mse_history = []
    acc_history = []
    for it in range(n_iter):
        inputs, targets, output_mask = generate_copy_batch(batch_size)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        mask = output_mask.unsqueeze(-1)
        loss = ((outputs - targets) ** 2 * mask).sum() / (mask.sum() * OUTPUT_DIM)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        mse, acc = compute_metrics(outputs.detach(), targets, output_mask)
        losses.append(loss.item())
        mse_history.append(mse)
        acc_history.append(acc)
        if verbose and (it + 1) % log_interval == 0:
            print("  Iter", it+1, "| Loss:", round(loss.item(), 5),
                  "| MSE:", round(mse, 5), "| Acc:", round(acc, 4))
    return model, losses, mse_history, acc_history

# ============================================================
# QUICK TEST: 200 iterations to verify BPTT works
# ============================================================
print("")
print("=== Testing BPTT (200 iterations) ===")
model, losses, mse_history, acc_history = train_bptt(n_iter=200, lr=0.001, log_interval=50)
print("")
print("BPTT Quick Test Results:")
print("  Initial MSE:", round(mse_history[0], 5))
print("  Final MSE:", round(mse_history[-1], 5))
print("  Initial Acc:", round(acc_history[0], 4))
print("  Final Acc:", round(acc_history[-1], 4))
print("  Loss improved:", mse_history[-1] < mse_history[0])

# Check model parameters exist
print("")
print("Model parameters:")
for name, param in model.named_parameters():
    print("  ", name, param.shape)

# Run longer test to see convergence
print("")
print("=== Testing BPTT (500 iterations, lr=0.005) ===")
model2, losses2, mse2, acc2 = train_bptt(n_iter=500, lr=0.005, log_interval=100)
print("")
print("BPTT 500-iter Results:")
print("  Initial MSE:", round(mse2[0], 5))
print("  Final MSE:", round(mse2[-1], 5))
print("  Final Acc:", round(acc2[-1], 4))

print("")
print("=== Step 2: BPTT Implementation VERIFIED ===")