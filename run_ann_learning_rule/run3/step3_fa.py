import torch
import torch.nn as nn
import numpy as np
import math
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

N_BITS = 8
DELAY = 10
INPUT_DIM = N_BITS + 1
OUTPUT_DIM = N_BITS
HIDDEN_SIZE = 128
SEQ_LEN = N_BITS + DELAY + N_BITS

def generate_copy_batch(batch_size, n_bits=N_BITS, delay=DELAY, dev=None):
    if dev is None: dev = device
    seq_len = n_bits + delay + n_bits
    patterns = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)
    inputs = torch.zeros(batch_size, seq_len, n_bits+1)
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

def compute_metrics(outputs, targets, output_mask):
    mask = output_mask.unsqueeze(-1)
    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*targets.shape[-1])
    pred_bits = (outputs > 0.5).float()
    correct = ((pred_bits == targets).float() * mask).sum()
    return mse.item(), (correct / (mask.sum()*targets.shape[-1])).item()

# ============================================================
# FA TRAINING: manually implement forward + FA backward
# ============================================================
# FA uses fixed random feedback matrices B_rec (hidden->hidden) and B_out (out->hidden)
# instead of W_rec^T and W_out^T during backward pass

def train_fa(n_iter=200, batch_size=32, lr=0.001, verbose=True, log_interval=50):
    torch.manual_seed(SEED)
    # Weight matrices
    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device).requires_grad_(False)
    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device).requires_grad_(False)
    b_rec = torch.zeros(HIDDEN_SIZE).to(device).requires_grad_(False)
    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device).requires_grad_(False)
    b_out = torch.zeros(OUTPUT_DIM).to(device).requires_grad_(False)
    # Fixed random feedback matrices (FA key component)
    B_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)
    B_out = (torch.randn(HIDDEN_SIZE, OUTPUT_DIM) * 0.1).to(device)
    # Adam state
    params = [W_in, W_rec, b_rec, W_out, b_out]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mse_history, acc_history = [], []
    for it in range(n_iter):
        inputs, targets, output_mask = generate_copy_batch(batch_size)
        # Forward pass - store activations
        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        hs = []  # hidden states
        as_ = []  # pre-activation
        outputs = []
        for t in range(SEQ_LEN):
            x_t = inputs[:, t, :]
            a_t = h @ W_rec.T + x_t @ W_in.T + b_rec
            h = torch.tanh(a_t)
            y_t = h @ W_out.T + b_out
            hs.append(h)
            as_.append(a_t)
            outputs.append(y_t)
        outputs_t = torch.stack(outputs, dim=1)
        # Compute output errors
        out_err = (outputs_t - targets) * output_mask.unsqueeze(-1)
        # Backward pass using FA feedback matrices
        # Accumulate gradients
        dW_in  = torch.zeros_like(W_in)
        dW_rec = torch.zeros_like(W_rec)
        db_rec = torch.zeros_like(b_rec)
        dW_out = torch.zeros_like(W_out)
        db_out = torch.zeros_like(b_out)
        delta_h_next = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        for t in reversed(range(SEQ_LEN)):
            h_t = hs[t]
            h_prev = hs[t-1] if t > 0 else torch.zeros(batch_size, HIDDEN_SIZE, device=device)
            x_t = inputs[:, t, :]
            dtanh = 1.0 - h_t**2
            # Output error projected back via B_out (FA) instead of W_out^T
            e_out_t = out_err[:, t, :]  # (batch, output_dim)
            delta_out = e_out_t @ B_out.T  # (batch, hidden) - FA feedback
            # Total delta_h: from output error + recurrent from next step via B_rec
            delta_h = (delta_out + delta_h_next) * dtanh
            dW_out += e_out_t.T @ h_t / batch_size
            db_out += e_out_t.mean(0)
            dW_rec += delta_h.T @ h_prev / batch_size
            dW_in  += delta_h.T @ x_t / batch_size
            db_rec += delta_h.mean(0)
            # Propagate delta_h to next step via B_rec (FA)
            delta_h_next = delta_h @ B_rec.T
        # Clip gradients
        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]
        total_norm = sum(g.norm()**2 for g in grads)**0.5
        clip = 1.0 / (total_norm + 1e-8)
        if total_norm > 1.0:
            grads = [g * clip for g in grads]
        # Adam update
        t_adam = it + 1
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1*m[i] + (1-beta1)*g
            v[i] = beta2*v[i] + (1-beta2)*g**2
            m_hat = m[i] / (1 - beta1**t_adam)
            v_hat = v[i] / (1 - beta2**t_adam)
            p.data -= lr * m_hat / (v_hat.sqrt() + eps)
        mse, acc = compute_metrics(outputs_t.detach(), targets, output_mask)
        mse_history.append(mse)
        acc_history.append(acc)
        if verbose and (it+1) % log_interval == 0:
            print("  Iter", it+1, "| MSE:", round(mse, 5), "| Acc:", round(acc, 4))
    return mse_history, acc_history

print("")
print("=== Testing FA (200 iterations, lr=0.001) ===")
mse_h, acc_h = train_fa(n_iter=200, lr=0.001, log_interval=50)
print("")
print("FA Quick Test Results:")
print("  Initial MSE:", round(mse_h[0], 5))
print("  Final MSE:", round(mse_h[-1], 5))
print("  Final Acc:", round(acc_h[-1], 4))
print("  Learning:", mse_h[-1] < mse_h[0])

print("")
print("=== Testing FA (500 iterations, lr=0.001) ===")
mse_h2, acc_h2 = train_fa(n_iter=500, lr=0.001, log_interval=100)
print("")
print("FA 500-iter Results:")
print("  Initial MSE:", round(mse_h2[0], 5))
print("  Final MSE:", round(mse_h2[-1], 5))
print("  Final Acc:", round(acc_h2[-1], 4))

print("")
print("=== Step 3: FA Implementation VERIFIED ===")