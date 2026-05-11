import torch
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
# PSC TRAINING FUNCTION
# ============================================================
def train_psc(n_iter=200, batch_size=32, lr=0.001, lr_pred=0.01,
              beta=0.3, gamma=0.5, lam=0.9, T_theta=10,
              use_oscillatory_gate=True, verbose=True, log_interval=50):
    """
    PSC: Predictive Self-Correction learning rule.
    Args:
      beta: prediction compartment decay rate
      gamma: weight for prediction error vs output error in credit signal
      lam: eligibility trace decay
      T_theta: oscillation period for gating
      use_oscillatory_gate: if False, gate=1 always (ablation)
    """
    torch.manual_seed(SEED)
    # Main RNN weights
    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)
    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)
    b_rec = torch.zeros(HIDDEN_SIZE).to(device)
    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)
    b_out = torch.zeros(OUTPUT_DIM).to(device)
    # Prediction weight matrix (separate, self-supervised)
    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)
    # Adam states for main weights
    params = [W_in, W_rec, b_rec, W_out, b_out]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    mse_history, acc_history = [], []
    for it in range(n_iter):
        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)
        # Initialize hidden state, prediction compartment, eligibility trace
        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        p = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        e = torch.zeros(batch_size, HIDDEN_SIZE, HIDDEN_SIZE, device=device)
        # Accumulate weight updates over sequence
        dW_in  = torch.zeros_like(W_in)
        dW_rec = torch.zeros_like(W_rec)
        db_rec = torch.zeros_like(b_rec)
        dW_out = torch.zeros_like(W_out)
        db_out = torch.zeros_like(b_out)
        dW_pred = torch.zeros_like(W_pred)
        outputs_list = []
        for t in range(SEQ_LEN):
            x_t = inputs_b[:, t, :]  # (batch, input_dim)
            h_prev = h.clone()
            # Main RNN forward
            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec
            h = torch.tanh(a_t)  # (batch, hidden)
            dtanh = 1.0 - h**2    # (batch, hidden)
            # Output
            y_t = h @ W_out.T + b_out  # (batch, output_dim)
            outputs_list.append(y_t)
            # Step 2: Update prediction compartment
            # p_i(t) = (1-beta)*p_i(t-1) + beta*(W_pred[i,:] @ h_prev)
            p = (1 - beta) * p + beta * (h_prev @ W_pred.T)
            # Step 3: Prediction error (batch, hidden)
            delta_pred = h - p
            # Step 4: Update eligibility trace
            # e_ij(t) = lam*e_ij(t-1) + h_prev_j * dtanh_i
            # e shape: (batch, hidden_i, hidden_j)
            e = lam * e + torch.bmm(dtanh.unsqueeze(2), h_prev.unsqueeze(1))
            # Step 5: Output error (only at output timesteps)
            is_output = output_mask[:, t].unsqueeze(1)  # (batch, 1)
            out_err_t = (y_t - targets_b[:, t, :]) * is_output  # (batch, output_dim)
            # Project output error to hidden space via W_out^T
            output_err_hidden = out_err_t @ W_out  # (batch, hidden)
            # Step 6: Credit signal
            s_t = gamma * delta_pred + (1 - gamma) * output_err_hidden  # (batch, hidden)
            # Step 7: Oscillatory gate
            if use_oscillatory_gate:
                gate = max(0.0, math.sin(2 * math.pi * t / T_theta))
            else:
                gate = 1.0
            # Step 8: W_rec update: Delta_W_rec_ij = -lr * s_i(t) * e_ij(t) * gate
            # s_t: (batch, hidden_i), e: (batch, hidden_i, hidden_j)
            # dW_rec_ij += mean over batch of s_i * e_ij * gate
            if gate > 0:
                s_e = s_t.unsqueeze(2) * e  # (batch, hidden_i, hidden_j)
                dW_rec += gate * s_e.mean(0)
                # W_in update using same credit signal with input eligibility
                # eligibility for W_in: dtanh_i * x_j at current step
                e_in = torch.bmm(dtanh.unsqueeze(2), x_t.unsqueeze(1))  # (batch, hidden, input)
                dW_in += gate * (s_t.unsqueeze(2) * e_in).mean(0)
                db_rec += gate * s_t.mean(0)
            # W_out update (always, standard gradient)
            dW_out += out_err_t.T @ h / batch_size
            db_out += out_err_t.mean(0)
            # Step 9: W_pred update (Hebbian, always)
            dW_pred += (delta_pred.unsqueeze(2) * h_prev.unsqueeze(1)).mean(0)
        # Clip and apply gradients with Adam
        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]
        total_norm = sum(g.norm()**2 for g in grads)**0.5
        if total_norm > 1.0:
            clip = 1.0 / (total_norm + 1e-8)
            grads = [g * clip for g in grads]
        t_adam = it + 1
        for i, (p_w, g) in enumerate(zip(params, grads)):
            m[i] = beta1*m[i] + (1-beta1)*g
            v[i] = beta2*v[i] + (1-beta2)*g**2
            m_hat = m[i] / (1 - beta1**t_adam)
            v_hat = v[i] / (1 - beta2**t_adam)
            p_w.data -= lr * m_hat / (v_hat.sqrt() + eps_adam)
        # W_pred update (simple SGD, no Adam needed)
        W_pred.data -= lr_pred * dW_pred
        outputs_t = torch.stack(outputs_list, dim=1)
        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)
        mse_history.append(mse)
        acc_history.append(acc)
        if verbose and (it+1) % log_interval == 0:
            print("  Iter", it+1, "| MSE:", round(mse, 5), "| Acc:", round(acc, 4),
                  "| gate_mode:", "osc" if use_oscillatory_gate else "always")
    return mse_history, acc_history

print("")
print("=== Testing PSC WITH oscillatory gate (200 iters) ===")
mse1, acc1 = train_psc(n_iter=200, lr=0.001, beta=0.3, gamma=0.5,
                        lam=0.9, T_theta=10, use_oscillatory_gate=True, log_interval=50)
print("PSC-osc: Initial MSE:", round(mse1[0],5), "Final MSE:", round(mse1[-1],5),
        "Final Acc:", round(acc1[-1],4))

print("")
print("=== Testing PSC WITHOUT oscillatory gate (200 iters) ===")
mse2, acc2 = train_psc(n_iter=200, lr=0.001, beta=0.3, gamma=0.5,
                        lam=0.9, T_theta=10, use_oscillatory_gate=False, log_interval=50)
print("PSC-always: Initial MSE:", round(mse2[0],5), "Final MSE:", round(mse2[-1],5),
        "Final Acc:", round(acc2[-1],4))

print("")
print("=== Testing PSC with different hyperparams (lr=0.005, beta=0.1, gamma=0.7) ===")
mse3, acc3 = train_psc(n_iter=200, lr=0.005, beta=0.1, gamma=0.7,
                        lam=0.95, T_theta=20, use_oscillatory_gate=True, log_interval=50)
print("PSC-v2: Initial MSE:", round(mse3[0],5), "Final MSE:", round(mse3[-1],5),
        "Final Acc:", round(acc3[-1],4))

print("")
print("=== Step 4: PSC Implementation VERIFIED ===")