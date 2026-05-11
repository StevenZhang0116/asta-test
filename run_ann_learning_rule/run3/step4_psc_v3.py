import torch
import numpy as np
import math

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
# PSC TRAINING FUNCTION v3
# Key fix: correct W_rec update formula using batch-level e-prop style
# Delta_W_rec[i,j] = s_i * e_ij  (s_i scalar credit, e_ij eligibility)
# ============================================================
def train_psc(n_iter=500, batch_size=32, lr=0.001, lr_pred=0.001,
              beta=0.3, gamma=0.0, lam=0.9, T_theta=10,
              use_oscillatory_gate=False, verbose=True, log_interval=100):
    torch.manual_seed(SEED)
    W_in  = (torch.randn(HIDDEN_SIZE, INPUT_DIM) * 0.1).to(device)
    W_rec = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1).to(device)
    b_rec = torch.zeros(HIDDEN_SIZE).to(device)
    W_out = (torch.randn(OUTPUT_DIM, HIDDEN_SIZE) * 0.1).to(device)
    b_out = torch.zeros(OUTPUT_DIM).to(device)
    W_pred = (torch.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.01).to(device)
    params = [W_in, W_rec, b_rec, W_out, b_out]
    m_adam = [torch.zeros_like(p) for p in params]
    v_adam = [torch.zeros_like(p) for p in params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    mse_history, acc_history = [], []
    for it in range(n_iter):
        inputs_b, targets_b, output_mask = generate_copy_batch(batch_size)
        # Per-sample eligibility traces to properly handle batch
        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        p_comp = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        # Eligibility trace: shape (batch, hidden_i, hidden_j)
        e_rec = torch.zeros(batch_size, HIDDEN_SIZE, HIDDEN_SIZE, device=device)
        e_in_trace = torch.zeros(batch_size, HIDDEN_SIZE, INPUT_DIM, device=device)
        dW_in  = torch.zeros_like(W_in)
        dW_rec = torch.zeros_like(W_rec)
        db_rec = torch.zeros_like(b_rec)
        dW_out = torch.zeros_like(W_out)
        db_out = torch.zeros_like(b_out)
        dW_pred = torch.zeros_like(W_pred)
        outputs_list = []
        for t in range(SEQ_LEN):
            x_t = inputs_b[:, t, :]   # (batch, input_dim)
            h_prev = h.clone()          # (batch, hidden)
            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec
            h = torch.tanh(a_t)
            dtanh = 1.0 - h**2          # (batch, hidden)
            y_t = h @ W_out.T + b_out
            outputs_list.append(y_t)
            # Prediction compartment per sample
            p_comp = (1 - beta) * p_comp + beta * (h_prev @ W_pred.T)
            delta_pred = h - p_comp  # (batch, hidden)
            # Eligibility traces per sample
            # e_rec[b,i,j] = lam*e_rec[b,i,j] + dtanh[b,i] * h_prev[b,j]
            e_rec = lam * e_rec + torch.bmm(dtanh.unsqueeze(2), h_prev.unsqueeze(1))
            e_in_trace = lam * e_in_trace + torch.bmm(dtanh.unsqueeze(2), x_t.unsqueeze(1))
            # Clip eligibility traces
            e_rec_norm = e_rec.norm(dim=(1,2), keepdim=True)
            e_rec = torch.where(e_rec_norm > 5.0, e_rec * 5.0 / (e_rec_norm + 1e-8), e_rec)
            # Output error (only at output timesteps)
            is_output = output_mask[:, t].bool()   # (batch,)
            out_err_t = torch.zeros_like(y_t)
            if is_output.any():
                out_err_t[is_output] = y_t[is_output] - targets_b[:, t, :][is_output]
            # Output error projected to hidden via W_out^T
            output_err_hidden = out_err_t @ W_out  # (batch, hidden)
            # Credit signal per sample
            # s_i = gamma * delta_pred_i + (1-gamma) * output_err_i
            s_t = gamma * delta_pred + (1 - gamma) * output_err_hidden  # (batch, hidden)
            # Oscillatory gate
            if use_oscillatory_gate:
                gate = max(0.0, math.sin(2 * math.pi * t / T_theta))
            else:
                gate = 1.0
            # W_rec update: Delta_W_rec[b,i,j] = gate * s_t[b,i] * e_rec[b,i,j]
            if gate > 0:
                # s_t: (batch, hidden_i) -> (batch, hidden_i, 1)
                # e_rec: (batch, hidden_i, hidden_j)
                dW_rec += gate * (s_t.unsqueeze(2) * e_rec).mean(0)
                dW_in  += gate * (s_t.unsqueeze(2) * e_in_trace).mean(0)
                db_rec += gate * s_t.mean(0)
            # W_out update (standard)
            dW_out += out_err_t.T @ h / batch_size
            db_out += out_err_t.mean(0)
            # W_pred Hebbian update: Delta_W_pred[i,j] = -lr_pred * delta_pred_i * h_prev_j
            dW_pred += (delta_pred.unsqueeze(2) * h_prev.unsqueeze(1)).mean(0)
        # Clip and apply main gradients
        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]
        total_norm = sum(g.norm()**2 for g in grads)**0.5
        if total_norm > 1.0:
            clip = 1.0 / (total_norm + 1e-8)
            grads = [g * clip for g in grads]
        t_adam = it + 1
        for i, (p_w, g) in enumerate(zip(params, grads)):
            m_adam[i] = beta1*m_adam[i] + (1-beta1)*g
            v_adam[i] = beta2*v_adam[i] + (1-beta2)*g**2
            m_hat = m_adam[i] / (1 - beta1**t_adam)
            v_hat = v_adam[i] / (1 - beta2**t_adam)
            p_w.data -= lr * m_hat / (v_hat.sqrt() + eps_adam)
        # W_pred update
        dW_pred_norm = dW_pred.norm()
        if dW_pred_norm > 1.0:
            dW_pred = dW_pred / (dW_pred_norm + 1e-8)
        W_pred.data -= lr_pred * dW_pred
        outputs_t = torch.stack(outputs_list, dim=1)
        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)
        mse_history.append(mse)
        acc_history.append(acc)
        if verbose and (it+1) % log_interval == 0:
            print("  Iter", it+1, "| MSE:", round(mse,5), "| Acc:", round(acc,4))
    return mse_history, acc_history

# Test 1: Pure output error (gamma=0) without gate - should learn if e-prop-like rule works
print("")
print("=== PSC gamma=0 (pure output err), no gate, 500 iters ===")
mse0, acc0 = train_psc(500, lr=0.001, gamma=0.0, lam=0.9, use_oscillatory_gate=False)
print("gamma=0: MSE", round(mse0[0],4), "->", round(mse0[-1],4), "| Acc", round(acc0[-1],4))

# Test 2: Mixed (gamma=0.3) without gate
print("")
print("=== PSC gamma=0.3 (mixed), no gate, 500 iters ===")
mse3, acc3 = train_psc(500, lr=0.001, gamma=0.3, lam=0.9, use_oscillatory_gate=False)
print("gamma=0.3: MSE", round(mse3[0],4), "->", round(mse3[-1],4), "| Acc", round(acc3[-1],4))

# Test 3: Mixed (gamma=0.3) WITH oscillatory gate
print("")
print("=== PSC gamma=0.3, WITH gate (T=10), 500 iters ===")
mse3g, acc3g = train_psc(500, lr=0.001, gamma=0.3, lam=0.9, T_theta=10, use_oscillatory_gate=True)
print("gamma=0.3+gate: MSE", round(mse3g[0],4), "->", round(mse3g[-1],4), "| Acc", round(acc3g[-1],4))

# Check for NaN
all_mse = mse0 + mse3 + mse3g
nan_count = sum(1 for x in all_mse if x != x)
print("")
print("NaN count:", nan_count)
print("Learning (mse0):", mse0[-1] < mse0[0])
print("Learning (mse3):", mse3[-1] < mse3[0])
print("=== Step 4: PSC v3 test complete ===")