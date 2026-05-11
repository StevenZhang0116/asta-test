import torch
import numpy as np
import math
import json
import os
from itertools import product

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
WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"

def generate_copy_batch(batch_size, dev=None):
    if dev is None: dev = device
    n_bits, delay = N_BITS, DELAY
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

def train_psc(n_iter, batch_size=32, lr=0.001, lr_pred=0.001,
              beta=0.3, gamma=0.3, lam=0.9, T_theta=10,
              use_oscillatory_gate=True):
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
        h = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        p_comp = torch.zeros(batch_size, HIDDEN_SIZE, device=device)
        e_rec = torch.zeros(batch_size, HIDDEN_SIZE, HIDDEN_SIZE, device=device)
        e_in_trace = torch.zeros(batch_size, HIDDEN_SIZE, INPUT_DIM, device=device)
        dW_in=torch.zeros_like(W_in); dW_rec=torch.zeros_like(W_rec)
        db_rec=torch.zeros_like(b_rec); dW_out=torch.zeros_like(W_out)
        db_out=torch.zeros_like(b_out); dW_pred=torch.zeros_like(W_pred)
        outputs_list = []
        for t in range(SEQ_LEN):
            x_t = inputs_b[:, t, :]
            h_prev = h.clone()
            a_t = h_prev @ W_rec.T + x_t @ W_in.T + b_rec
            h = torch.tanh(a_t)
            dtanh = 1.0 - h**2
            y_t = h @ W_out.T + b_out
            outputs_list.append(y_t)
            p_comp = (1-beta)*p_comp + beta*(h_prev @ W_pred.T)
            delta_pred = h - p_comp
            e_rec = lam*e_rec + torch.bmm(dtanh.unsqueeze(2), h_prev.unsqueeze(1))
            e_in_trace = lam*e_in_trace + torch.bmm(dtanh.unsqueeze(2), x_t.unsqueeze(1))
            e_rec_norm = e_rec.norm(dim=(1,2), keepdim=True)
            e_rec = torch.where(e_rec_norm > 5.0, e_rec*5.0/(e_rec_norm+1e-8), e_rec)
            is_output = output_mask[:, t].bool()
            out_err_t = torch.zeros_like(y_t)
            if is_output.any():
                out_err_t[is_output] = y_t[is_output] - targets_b[:,t,:][is_output]
            output_err_hidden = out_err_t @ W_out
            s_t = gamma*delta_pred + (1-gamma)*output_err_hidden
            gate = max(0.0, math.sin(2*math.pi*t/T_theta)) if use_oscillatory_gate else 1.0
            if gate > 0:
                dW_rec += gate*(s_t.unsqueeze(2)*e_rec).mean(0)
                dW_in  += gate*(s_t.unsqueeze(2)*e_in_trace).mean(0)
                db_rec += gate*s_t.mean(0)
            dW_out += out_err_t.T @ h / batch_size
            db_out += out_err_t.mean(0)
            dW_pred += (delta_pred.unsqueeze(2)*h_prev.unsqueeze(1)).mean(0)
        grads = [dW_in, dW_rec, db_rec, dW_out, db_out]
        total_norm = sum(g.norm()**2 for g in grads)**0.5
        if total_norm > 1.0:
            clip = 1.0/(total_norm+1e-8)
            grads = [g*clip for g in grads]
        t_adam = it+1
        for i,(p_w,g) in enumerate(zip(params,grads)):
            m_adam[i]=beta1*m_adam[i]+(1-beta1)*g
            v_adam[i]=beta2*v_adam[i]+(1-beta2)*g**2
            m_hat=m_adam[i]/(1-beta1**t_adam)
            v_hat=v_adam[i]/(1-beta2**t_adam)
            p_w.data -= lr*m_hat/(v_hat.sqrt()+eps_adam)
        dW_pred_norm = dW_pred.norm()
        if dW_pred_norm > 1.0: dW_pred = dW_pred/(dW_pred_norm+1e-8)
        W_pred.data -= lr_pred*dW_pred
        outputs_t = torch.stack(outputs_list, dim=1)
        mse, acc = compute_metrics(outputs_t.detach(), targets_b, output_mask)
        mse_history.append(mse)
        acc_history.append(acc)
    return mse_history, acc_history

# ============================================================
# GRID SEARCH
# ============================================================
lrs = [0.001, 0.005, 0.01]
betas = [0.1, 0.3]
gammas = [0.3, 0.5, 0.7]
lams = [0.9, 0.95]
T_thetas = [10, 20]
lr_pred_fixed = 0.001
N_GRID_ITER = 1000

grid_results_osc = []
grid_results_nogate = []

# PSC with oscillatory gate
print("")
print("=== Grid Search: PSC WITH oscillatory gate ===")
combo_count = 0
for lr, beta, gamma, lam, T_theta in product(lrs, betas, gammas, lams, T_thetas):
    combo_count += 1
    mse_h, acc_h = train_psc(N_GRID_ITER, lr=lr, lr_pred=lr_pred_fixed,
                              beta=beta, gamma=gamma, lam=lam, T_theta=T_theta,
                              use_oscillatory_gate=True)
    final_mse = mse_h[-1] if not math.isnan(mse_h[-1]) else 999.0
    final_acc = acc_h[-1] if not math.isnan(acc_h[-1]) else 0.0
    config = {"lr":lr, "beta":beta, "gamma":gamma, "lam":lam, "T_theta":T_theta,
              "final_mse":final_mse, "final_acc":final_acc,
              "mse_history":mse_h[::100], "acc_history":acc_h[::100]}
    grid_results_osc.append(config)
    if combo_count % 12 == 0:
        print("  Combo", combo_count, "/ 72 done. Best so far:",
              round(min(r["final_mse"] for r in grid_results_osc), 4))

# PSC without gate (T_theta doesn't matter, so use T_theta=10 only)
print("")
print("=== Grid Search: PSC WITHOUT gate ===")
combo_count = 0
for lr, beta, gamma, lam in product(lrs, betas, gammas, lams):
    combo_count += 1
    mse_h, acc_h = train_psc(N_GRID_ITER, lr=lr, lr_pred=lr_pred_fixed,
                              beta=beta, gamma=gamma, lam=lam, T_theta=10,
                              use_oscillatory_gate=False)
    final_mse = mse_h[-1] if not math.isnan(mse_h[-1]) else 999.0
    final_acc = acc_h[-1] if not math.isnan(acc_h[-1]) else 0.0
    config = {"lr":lr, "beta":beta, "gamma":gamma, "lam":lam, "T_theta":"N/A",
              "final_mse":final_mse, "final_acc":final_acc,
              "mse_history":mse_h[::100], "acc_history":acc_h[::100]}
    grid_results_nogate.append(config)
    if combo_count % 9 == 0:
        print("  Combo", combo_count, "/ 36 done. Best so far:",
              round(min(r["final_mse"] for r in grid_results_nogate), 4))

# Save results
grid_results_osc.sort(key=lambda x: x["final_mse"])
grid_results_nogate.sort(key=lambda x: x["final_mse"])

results = {"grid_psc_osc": grid_results_osc, "grid_psc_nogate": grid_results_nogate}
out_path = os.path.join(WORK_DIR, "grid_search_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("")
print("Grid search results saved to", out_path)

print("")
print("=== Top 5 PSC-OSC configs ===")
for i, cfg in enumerate(grid_results_osc[:5]):
    print("  Rank", i+1, "| lr:", cfg["lr"], "beta:", cfg["beta"],
          "gamma:", cfg["gamma"], "lam:", cfg["lam"], "T:", cfg["T_theta"],
          "| MSE:", round(cfg["final_mse"],4), "Acc:", round(cfg["final_acc"],4))

print("")
print("=== Top 5 PSC-NOGATE configs ===")
for i, cfg in enumerate(grid_results_nogate[:5]):
    print("  Rank", i+1, "| lr:", cfg["lr"], "beta:", cfg["beta"],
          "gamma:", cfg["gamma"], "lam:", cfg["lam"],
          "| MSE:", round(cfg["final_mse"],4), "Acc:", round(cfg["final_acc"],4))

print("")
print("=== Step 5: Grid Search COMPLETE ===")