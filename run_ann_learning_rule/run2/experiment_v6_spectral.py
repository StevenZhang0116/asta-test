"""
Experiment v6: Spectral Gating Multi-Timescale Eligibility (SGMTE)

Novel addition: Slow-band neurons GATE the eligibility accumulation of fast-band neurons.
This creates a hierarchical credit structure:
- Slow bands maintain long-range context
- When slow bands are active, they "open gates" for fast bands to accumulate eligibility
- This means fast bands only accumulate credit in temporally relevant windows
  defined by the slow band's activity pattern

Biological analogy: Theta rhythm (slow) gates gamma activity (fast) in hippocampus.
The slow oscillation defines "what epoch we're in" and the fast neurons encode
fine-grained details within that epoch.

Also tests on adding problem (seq_len=50, 100).
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import json
import time

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)


def generate_copy_task(batch_size, seq_len=10, delay=10, n_symbols=8):
    total_len = seq_len + delay + seq_len + 2
    n_input = n_symbols + 2
    symbols = torch.randint(0, n_symbols, (batch_size, seq_len))
    x = torch.zeros(batch_size, total_len, n_input, device=DEVICE)
    for t in range(seq_len):
        x[:, t, :n_symbols] = torch.nn.functional.one_hot(symbols[:, t], n_symbols).float().to(DEVICE)
    x[:, seq_len, n_symbols] = 1.0
    x[:, seq_len + delay + 1, n_symbols + 1] = 1.0
    target = torch.zeros(batch_size, total_len, n_symbols, device=DEVICE)
    for t in range(seq_len):
        target[:, seq_len + delay + 2 + t, :] = torch.nn.functional.one_hot(symbols[:, t], n_symbols).float().to(DEVICE)
    mask = torch.zeros(batch_size, total_len, device=DEVICE)
    mask[:, seq_len + delay + 2:] = 1.0
    return x, target, mask


def generate_adding_problem(batch_size, seq_len=50):
    values = torch.rand(batch_size, seq_len, 1, device=DEVICE)
    mask_seq = torch.zeros(batch_size, seq_len, 1, device=DEVICE)
    for b in range(batch_size):
        positions = torch.randperm(seq_len)[:2]
        mask_seq[b, positions, 0] = 1.0
    x = torch.cat([values, mask_seq], dim=-1)
    target = (values * mask_seq).sum(dim=1)
    return x, target


class SGMTE:
    """
    Spectral Gating Multi-Timescale Eligibility.

    Neurons in K bands with matched λ. Slow bands gate fast band eligibility:
    - gate_fast(t) = σ(W_gate @ h_slow(t))
    - e_ij(t) = λ_i * e_ij(t-1) + gate_i(t) * (1-h_i²) * h_j(t-1)

    For slow band neurons, gate = 1 always (they accumulate freely).
    For fast band neurons, gate depends on slow band activity.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 n_bands=4, lam_min=0.8, lam_max=0.998, lr=0.01,
                 use_gating=True):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_bands = n_bands
        self.lr = lr
        self.use_gating = use_gating
        self.band_size = hidden_size // n_bands

        # Per-neuron lambda
        self.lam = torch.zeros(hidden_size, device=DEVICE)
        lam_values = torch.linspace(lam_max, lam_min, n_bands)
        for k in range(n_bands):
            start = k * self.band_size
            end = start + self.band_size if k < n_bands - 1 else hidden_size
            self.lam[start:end] = lam_values[k]

        # Weights
        scale_h = 1.0 / np.sqrt(hidden_size)
        scale_i = 1.0 / np.sqrt(input_size)
        self.W = (torch.randn(hidden_size, hidden_size) * scale_h).to(DEVICE)
        self.U = (torch.randn(hidden_size, input_size) * scale_i).to(DEVICE)
        self.V = (torch.randn(output_size, hidden_size) * scale_h).to(DEVICE)
        self.b_h = torch.zeros(hidden_size, device=DEVICE)
        self.b_o = torch.zeros(output_size, device=DEVICE)
        self.B = (torch.randn(hidden_size, output_size) / np.sqrt(output_size)).to(DEVICE)

        # Gating weights: slow band → fast bands
        # W_gate projects from slow band neurons to all other neurons
        if use_gating:
            slow_size = self.band_size  # first band is slowest
            self.W_gate = (torch.randn(hidden_size, slow_size) * 0.1).to(DEVICE)
            self.b_gate = torch.zeros(hidden_size, device=DEVICE)

    def reset(self, batch_size):
        self.h = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.elig_W = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=DEVICE)
        self.elig_U = torch.zeros(batch_size, self.hidden_size, self.input_size, device=DEVICE)
        self.dW_accum = torch.zeros_like(self.W)
        self.dU_accum = torch.zeros_like(self.U)
        self.dV_accum = torch.zeros_like(self.V)
        self.n_updates = 0

    def forward_step(self, x_t):
        z = torch.matmul(self.h, self.W.T) + torch.matmul(x_t, self.U.T) + self.b_h
        h_new = torch.tanh(z)
        post_deriv = 1 - h_new**2  # [B, N]

        # Compute gating signal from slow band
        if self.use_gating:
            h_slow = h_new[:, :self.band_size]  # [B, slow_size]
            gate = torch.sigmoid(torch.matmul(h_slow, self.W_gate.T) + self.b_gate)  # [B, N]
            # Slow band always gates at 1 (they don't need gating)
            gate[:, :self.band_size] = 1.0
        else:
            gate = torch.ones(x_t.shape[0], self.hidden_size, device=DEVICE)

        # Gated eligibility update
        gated_post_deriv = gate * post_deriv  # [B, N]

        lam_expanded = self.lam.unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        self.elig_W = lam_expanded * self.elig_W + torch.einsum('bi,bj->bij', gated_post_deriv, self.h)
        self.elig_U = lam_expanded * self.elig_U + torch.einsum('bi,bd->bid', gated_post_deriv, x_t)

        self.h = h_new
        return torch.matmul(self.h, self.V.T) + self.b_o

    def accumulate_update(self, error, mask_t):
        if mask_t.sum() == 0:
            return
        batch_size = error.shape[0]
        masked_error = error * mask_t.unsqueeze(-1)
        L = torch.matmul(masked_error, self.B.T)
        self.dW_accum += torch.einsum('bi,bij->ij', L, self.elig_W) / batch_size
        self.dU_accum += torch.einsum('bi,bid->id', L, self.elig_U) / batch_size
        self.dV_accum += torch.einsum('bo,bh->oh', masked_error, self.h) / batch_size
        self.n_updates += 1

    def apply_update(self):
        if self.n_updates > 0:
            scale = 1.0 / self.n_updates
            self.W += self.lr * torch.clamp(self.dW_accum * scale, -0.5, 0.5)
            self.U += self.lr * torch.clamp(self.dU_accum * scale, -0.5, 0.5)
            self.V += self.lr * torch.clamp(self.dV_accum * scale, -0.5, 0.5)
            if self.use_gating:
                # Gate weights get a small update too (via eligibility of slow neurons)
                pass  # Gate weights are fixed for now — simplifies analysis


class BPTT_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_all=True):
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        if return_all:
            return self.fc(out)
        else:
            return self.fc(out[:, -1, :])


def run_copy(model, is_bptt, n_steps, batch_size, seq_len, delay, label):
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    total_len = seq_len + delay + seq_len + 2
    if is_bptt:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    accs = []
    for step in range(n_steps):
        x, target, mask = generate_copy_task(batch_size, seq_len, delay, n_symbols)
        if is_bptt:
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * output_size + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            output_all = output.detach()
        else:
            model.reset(batch_size)
            outputs = []
            for t in range(total_len):
                y_hat = model.forward_step(x[:, t, :])
                outputs.append(y_hat.detach().clone())
                error = target[:, t, :] - y_hat
                model.accumulate_update(error, mask[:, t])
            model.apply_update()
            output_all = torch.stack(outputs, dim=1)

        output_phase = output_all[:, seq_len+delay+2:, :]
        target_phase = target[:, seq_len+delay+2:, :]
        pred = output_phase.argmax(dim=-1)
        true = target_phase.argmax(dim=-1)
        valid = target_phase.sum(dim=-1) > 0
        acc = (pred[valid] == true[valid]).float().mean().item() if valid.sum() > 0 else 0.0
        accs.append(acc)

        if (step+1) % 2500 == 0:
            ra = np.mean(accs[-2500:])
            print(f"    {label} step {step+1}: acc={ra:.3f}", flush=True)
    return accs


def run_adding(model, is_bptt, n_steps, batch_size, seq_len, label):
    input_size = 2
    output_size = 1
    if is_bptt:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for step in range(n_steps):
        x, target = generate_adding_problem(batch_size, seq_len)
        if is_bptt:
            optimizer.zero_grad()
            output = model(x, return_all=False)
            loss = ((output - target)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        else:
            model.reset(batch_size)
            for t in range(seq_len):
                y_hat = model.forward_step(x[:, t, :])
            error = target - y_hat
            mask_t = torch.ones(batch_size, device=DEVICE)
            model.accumulate_update(error, mask_t)
            model.apply_update()
            losses.append((error**2).mean().item())

        if (step+1) % 2500 == 0:
            rl = np.mean(losses[-2500:])
            print(f"    {label} step {step+1}: mse={rl:.4f}", flush=True)
    return losses


def main():
    print("="*70, flush=True)
    print("EXPERIMENT v6: Spectral Gating MTE (SGMTE)", flush=True)
    print("="*70, flush=True)
    t0 = time.time()

    N_STEPS = 10000
    BATCH = 32
    HIDDEN = 64
    results = {}

    # ─── COPY TASK at d=30, 50 ───
    for delay in [30, 50]:
        print(f"\n{'─'*50} COPY d={delay} {'─'*50}", flush=True)
        n_symbols = 8
        inp, out = n_symbols + 2, n_symbols

        # BPTT
        m = BPTT_RNN(inp, HIDDEN, out).to(DEVICE)
        a = run_copy(m, True, N_STEPS, BATCH, 10, delay, f"BPTT_d{delay}")
        results[f"copy_BPTT_d{delay}"] = float(np.mean(a[-2000:]))

        # MTE (no gating) — same as v5 best
        m = SGMTE(inp, HIDDEN, out, n_bands=4, lam_min=0.8, lam_max=0.998, lr=0.01, use_gating=False)
        a = run_copy(m, False, N_STEPS, BATCH, 10, delay, f"MTE_d{delay}")
        results[f"copy_MTE_d{delay}"] = float(np.mean(a[-2000:]))

        # SGMTE (with gating) — our novel addition
        m = SGMTE(inp, HIDDEN, out, n_bands=4, lam_min=0.8, lam_max=0.998, lr=0.01, use_gating=True)
        a = run_copy(m, False, N_STEPS, BATCH, 10, delay, f"SGMTE_d{delay}")
        results[f"copy_SGMTE_d{delay}"] = float(np.mean(a[-2000:]))

    # ─── ADDING PROBLEM ───
    for seq_len in [50, 100]:
        print(f"\n{'─'*50} ADDING len={seq_len} {'─'*50}", flush=True)

        # BPTT
        m = BPTT_RNN(2, HIDDEN, 1).to(DEVICE)
        l = run_adding(m, True, N_STEPS, BATCH, seq_len, f"BPTT_add{seq_len}")
        results[f"add_BPTT_len{seq_len}"] = float(np.mean(l[-2000:]))

        # MTE (no gating)
        m = SGMTE(2, HIDDEN, 1, n_bands=4, lam_min=0.8, lam_max=0.998, lr=0.005, use_gating=False)
        l = run_adding(m, False, N_STEPS, BATCH, seq_len, f"MTE_add{seq_len}")
        results[f"add_MTE_len{seq_len}"] = float(np.mean(l[-2000:]))

        # SGMTE (with gating)
        m = SGMTE(2, HIDDEN, 1, n_bands=4, lam_min=0.8, lam_max=0.998, lr=0.005, use_gating=True)
        l = run_adding(m, False, N_STEPS, BATCH, seq_len, f"SGMTE_add{seq_len}")
        results[f"add_SGMTE_len{seq_len}"] = float(np.mean(l[-2000:]))

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL RESULTS (elapsed {elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\nCopy Task:", flush=True)
    print(f"  {'Method':<12} {'d=30':<10} {'d=50':<10}", flush=True)
    for prefix in ["BPTT", "MTE", "SGMTE"]:
        row = f"  {prefix:<12} "
        for d in [30, 50]:
            val = results.get(f"copy_{prefix}_d{d}", 0)
            row += f"{val:<10.3f} "
        print(row, flush=True)

    print(f"\nAdding Problem (random baseline MSE ≈ 0.167):", flush=True)
    print(f"  {'Method':<12} {'len=50':<10} {'len=100':<10}", flush=True)
    for prefix in ["BPTT", "MTE", "SGMTE"]:
        row = f"  {prefix:<12} "
        for sl in [50, 100]:
            val = results.get(f"add_{prefix}_len{sl}", 0)
            row += f"{val:<10.4f} "
        print(row, flush=True)

    print(f"\n--- Gating Effect ---", flush=True)
    for d in [30, 50]:
        mte = results.get(f"copy_MTE_d{d}", 0)
        sgmte = results.get(f"copy_SGMTE_d{d}", 0)
        print(f"  Copy d={d}: MTE={mte:.3f} → SGMTE={sgmte:.3f} (gating effect={sgmte-mte:+.3f})", flush=True)
    for sl in [50, 100]:
        mte = results.get(f"add_MTE_len{sl}", 0)
        sgmte = results.get(f"add_SGMTE_len{sl}", 0)
        print(f"  Adding len={sl}: MTE={mte:.4f} → SGMTE={sgmte:.4f} (gating effect={mte-sgmte:+.4f} lower is better)", flush=True)

    with open("experiment_v6_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiment_v6_results.json", flush=True)


if __name__ == "__main__":
    main()
