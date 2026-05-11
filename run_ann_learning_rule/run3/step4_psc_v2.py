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
    n_total = mask.sum() * targets.shape[-1]
    return mse.item(), (correct / n_total).item()

# ============================================================
# PSC TRAINING FUNCTION (v2 - with stability fixes)
# ============================================================
def train_psc(n_iter=200, batch_size=32, lr=0.001, lr_pred=0.001,
              beta=0.3, gamma=0.5, lam=0.9, T_theta=10,
              use_oscillatory_gate=True, verbose=True, log_interval=50):
    torch.manual_seed(SEED)
    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)
    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)
    b_rec = torch.zeros(HIDDEN_SIZE).to(device)
    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)
    b_out = torch.zeros(OUTPUT_DIM).to(device)
    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)
    params = [W_in, W_rec, b_rec, W_out, b_out]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    mse_history, acc_history = [], []
    for it in range(n_iter):
        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)
        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        p_comp = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        # Use per-neuron eligibility trace (not full matrix) for efficiency
        # e_i(t) = lam*e_i(t-1) + mean_j[h_j(t-1)] * dtanh_i(t)
        # But we need e_ij for W_rec update, so keep full trace but clip it
        e = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=device)  # averaged over batch
        dW_in  = torch.zeros_like(W_in)
        dW_rec = torch.zeros_like(W_rec)
        db_rec = torch.zeros_like(b_rec)
        dW_out = torch.zeros_like(W_out)
        db_out = torch.zeros_like(b_out)
        dW_pred = torch.zeros_like(W_pred)
        outputs_list = []
        for t in range(SEQ_LEN):
            x_t = inputs_b[:, t, :]
            h_prev = h.clone()
            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec
            h = torch.tanh(a_t)
            dtanh = 1.0 - h**2
            y_t = h @ W_out.T + b_out
            outputs_list.append(y_t)
            # Prediction compartment
            p_comp = (1 - beta) * p_comp + beta * (h_prev @ W_pred.T)
            delta_pred = h - p_comp  # (batch, hidden)
            # Eligibility trace (batch-averaged outer product)
            h_prev_mean = h_prev.mean(0)  # (hidden,)
            dtanh_mean  = dtanh.mean(0)   # (hidden,)
            e = lam * e + torch.outer(dtanh_mean, h_prev_mean)
            # Clip eligibility trace for stability
            e_norm = e.norm()
            if e_norm > 10.0:
                e = e * 10.0 / (e_norm + 1e-8)
            # Output error
            is_output = output_mask[:, t].unsqueeze(1)
            out_err_t = (y_t - targets_b[:, t, :]) * is_output
            output_err_hidden = out_err_t @ W_out  # (batch, hidden)
            # Credit signal (batch-averaged)
            delta_pred_mean = delta_pred.mean(0)  # (hidden,)
            output_err_mean = output_err_hidden.mean(0)  # (hidden,)
            s_t = gamma * delta_pred_mean + (1 - gamma) * output_err_mean  # (hidden,)
            # Clip credit signal
            s_norm = s_t.norm()
            if s_norm > 5.0:
                s_t = s_t * 5.0 / (s_norm + 1e-8)
            # Oscillatory gate
            if use_oscillatory_gate:
                gate = max(0.0, math.sin(2 * math.pi * t / T_theta))
            else:
                gate = 1.0
            # Weight updates
            if gate > 0:
                # dW_rec_ij += gate * s_i * e_ij
                dW_rec += gate * torch.outer(s_t, torch.ones(HIDDEN_SIZE, device=device)) * e
                # W_in: eligibility with input
                x_mean = x_t.mean(0)
                e_in = torch.outer(dtanh_mean, x_mean)
                dW_in += gate * torch.outer(s_t, torch.ones(INPUT_DIM, device=device)) * e_in
                db_rec += gate * s_t
            # W_out update (standard gradient)
            dW_out += out_err_t.T @ h / batch_size
            db_out += out_err_t.mean(0)
            # W_pred Hebbian update
            dW_pred += (delta_pred.unsqueeze(2) * h_prev.unsqueeze(1)).mean(0)
        # Clip main gradients
        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]
        total_norm = sum(g.norm()**2 for g in grads)**0.5
        if total_norm > 1.0:
            clip = 1.0 / (total_norm + 1e-8)
            grads = [g * clip for g in grads]
        # Adam update for main weights
        t_adam = it + 1
        for i, (p_w, g) in enumerate(zip(params, grads)):
            m[i] = beta1*m[i] + (1-beta1)*g
            v[i] = beta2*v[i] + (1-beta2)*g**2
            m_hat = m[i] / (1 - beta1**t_adam)
            v_hat = v[i] / (1 - beta2**t_adam)
            p_w.data -= lr * m_hat / (v_hat.sqrt() + eps_adam)
        # Clip and apply W_pred update
        pred_norm = dW_pred.norm()
        if pred_norm > 1.0:
            dW_pred = dW_pred / (pred_norm + 1e-8)
        W_pred.data -= lr_pred * dW_pred
        outputs_t = torch.stack(outputs_list, dim=1)
        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)
        mse_history.append(mse)
        acc_history.append(acc)
        if verbose and (it+1) % log_interval == 0:
            print("  Iter", it+1, "| MSE:", round(mse, 5), "| Acc:", round(acc, 4))
    return mse_history, acc_history

print("")
print("=== PSC-osc (200 iters, lr=0.001) ===")
mse1, acc1 = train_psc(200, lr=0.001, beta=0.3, gamma=0.5, lam=0.9, T_theta=10,
                        use_oscillatory_gate=True, log_interval=50)
print("Initial MSE:", round(mse1[0],5), "Final MSE:", round(mse1[-1],5), "Final Acc:", round(acc1[-1],4))

print("")
print("=== PSC-no-gate (200 iters, lr=0.001) ===")
mse2, acc2 = train_psc(200, lr=0.001, beta=0.3, gamma=0.5, lam=0.9, T_theta=10,
                        use_oscillatory_gate=False, log_interval=50)
print("Initial MSE:", round(mse2[0],5), "Final MSE:", round(mse2[-1],5), "Final Acc:", round(acc2[-1],4))

print("")
print("=== PSC-osc higher lr (200 iters, lr=0.005) ===")
mse3, acc3 = train_psc(200, lr=0.005, beta=0.1, gamma=0.3, lam=0.95, T_theta=20,
                        use_oscillatory_gate=True, log_interval=50)
print("Initial MSE:", round(mse3[0],5), "Final MSE:", round(mse3[-1],5), "Final Acc:", round(acc3[-1],4))

# Check for NaN
import math
nan_check = [math.isnan(x) for x in mse1+mse2+mse3]
print("")
print("NaN in any run:", any(nan_check))
print("=== Step 4: PSC v2 Implementation Test Complete ===")