"""
Experiment v5: PFSC-v2 — Multi-Timescale Pairwise Eligibility

Combines:
- Full pairwise eligibility e_ij(t) (from e-prop) — rich temporal structure
- Frequency-stratified decay λ_i per neuron (from PFSC) — multi-timescale credit

Each neuron i has a decay rate λ_i matched to its frequency band:
- Slow neurons (low ω): λ close to 1 → eligibility persists for many timesteps
- Fast neurons (high ω): λ smaller → eligibility captures recent events precisely

The readout combines all timescales, learning task-appropriate weighting.
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


class MultiTimescaleEprop:
    """
    PFSC-v2: Multi-Timescale e-prop.

    Full pairwise eligibility e_ij(t) with post-synaptic neuron-specific decay:
      e_ij(t) = λ_i * e_ij(t-1) + (1 - h_i²) * h_j(t-1)

    where λ_i depends on neuron i's assigned frequency band.
    Slow neurons have λ close to 1, fast neurons have smaller λ.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 n_bands=4, lam_min=0.85, lam_max=0.995, lr=0.01):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_bands = n_bands
        self.lr = lr

        # Assign neurons to bands and set per-neuron lambda
        band_size = hidden_size // n_bands
        self.lam = torch.zeros(hidden_size, device=DEVICE)
        lam_values = torch.linspace(lam_max, lam_min, n_bands)
        for k in range(n_bands):
            start = k * band_size
            end = start + band_size if k < n_bands - 1 else hidden_size
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

    def reset(self, batch_size):
        self.h = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        # Full pairwise eligibility: [B, N_post, N_pre]
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

        # Multi-timescale eligibility: λ_i applied per row (post-synaptic neuron)
        lam_expanded = self.lam.unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        self.elig_W = lam_expanded * self.elig_W + torch.einsum('bi,bj->bij', post_deriv, self.h)

        lam_expanded_u = self.lam.unsqueeze(0).unsqueeze(2)  # [1, N, 1]
        self.elig_U = lam_expanded_u * self.elig_U + torch.einsum('bi,bd->bid', post_deriv, x_t)

        self.h = h_new
        return torch.matmul(self.h, self.V.T) + self.b_o

    def accumulate_update(self, error, mask_t):
        if mask_t.sum() == 0:
            return
        batch_size = error.shape[0]
        masked_error = error * mask_t.unsqueeze(-1)
        L = torch.matmul(masked_error, self.B.T)  # [B, N]

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


class UniformEprop:
    """e-prop with single uniform λ (baseline)."""

    def __init__(self, input_size, hidden_size, output_size, lam=0.95, lr=0.01):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lam = lam
        self.lr = lr

        scale_h = 1.0 / np.sqrt(hidden_size)
        scale_i = 1.0 / np.sqrt(input_size)
        self.W = (torch.randn(hidden_size, hidden_size) * scale_h).to(DEVICE)
        self.U = (torch.randn(hidden_size, input_size) * scale_i).to(DEVICE)
        self.V = (torch.randn(output_size, hidden_size) * scale_h).to(DEVICE)
        self.b_h = torch.zeros(hidden_size, device=DEVICE)
        self.b_o = torch.zeros(output_size, device=DEVICE)
        self.B = (torch.randn(hidden_size, output_size) / np.sqrt(output_size)).to(DEVICE)

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
        post_deriv = 1 - h_new**2
        self.elig_W = self.lam * self.elig_W + torch.einsum('bi,bj->bij', post_deriv, self.h)
        self.elig_U = self.lam * self.elig_U + torch.einsum('bi,bd->bid', post_deriv, x_t)
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


class BPTT_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out)


def run_experiment(model_obj, is_bptt, n_steps, batch_size, seq_len, delay, label):
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    total_len = seq_len + delay + seq_len + 2

    if is_bptt:
        optimizer = torch.optim.Adam(model_obj.parameters(), lr=0.001)

    losses, accs = [], []
    for step in range(n_steps):
        x, target, mask = generate_copy_task(batch_size, seq_len, delay, n_symbols)

        if is_bptt:
            optimizer.zero_grad()
            output = model_obj(x)
            loss = ((output - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * output_size + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_obj.parameters(), 1.0)
            optimizer.step()
            output_all = output.detach()
            losses.append(loss.item())
        else:
            model_obj.reset(batch_size)
            outputs = []
            for t in range(total_len):
                y_hat = model_obj.forward_step(x[:, t, :])
                outputs.append(y_hat.detach().clone())
                error = target[:, t, :] - y_hat
                model_obj.accumulate_update(error, mask[:, t])
            model_obj.apply_update()
            output_all = torch.stack(outputs, dim=1)
            lv = ((output_all - target)**2 * mask.unsqueeze(-1)).sum().item() / (mask.sum().item() * output_size + 1e-8)
            losses.append(lv)

        output_phase = output_all[:, seq_len+delay+2:, :]
        target_phase = target[:, seq_len+delay+2:, :]
        pred = output_phase.argmax(dim=-1)
        true = target_phase.argmax(dim=-1)
        valid = target_phase.sum(dim=-1) > 0
        acc = (pred[valid] == true[valid]).float().mean().item() if valid.sum() > 0 else 0.0
        accs.append(acc)

        if (step+1) % 2500 == 0:
            rl = np.mean(losses[-2500:])
            ra = np.mean(accs[-2500:])
            print(f"    {label} step {step+1}: loss={rl:.4f} acc={ra:.3f}", flush=True)

    return losses, accs


def main():
    print("="*70, flush=True)
    print("EXPERIMENT v5: Multi-Timescale Pairwise Eligibility (PFSC-v2)", flush=True)
    print("="*70, flush=True)
    t0 = time.time()

    N_STEPS = 10000
    BATCH = 32
    HIDDEN = 64
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    results = {}

    delays = [10, 20, 30, 50]

    for delay in delays:
        print(f"\n{'='*50} DELAY={delay} {'='*50}", flush=True)

        # BPTT
        print(f"  [BPTT]", flush=True)
        m = BPTT_RNN(input_size, HIDDEN, output_size).to(DEVICE)
        l, a = run_experiment(m, True, N_STEPS, BATCH, 10, delay, f"BPTT_d{delay}")
        results[f"BPTT_d{delay}"] = float(np.mean(a[-2000:]))

        # Uniform e-prop (λ=0.95)
        print(f"  [e-prop λ=0.95]", flush=True)
        m = UniformEprop(input_size, HIDDEN, output_size, lam=0.95, lr=0.01)
        l, a = run_experiment(m, False, N_STEPS, BATCH, 10, delay, f"EP95_d{delay}")
        results[f"EP95_d{delay}"] = float(np.mean(a[-2000:]))

        # Uniform e-prop (λ=0.99) — high lambda baseline
        print(f"  [e-prop λ=0.99]", flush=True)
        m = UniformEprop(input_size, HIDDEN, output_size, lam=0.99, lr=0.01)
        l, a = run_experiment(m, False, N_STEPS, BATCH, 10, delay, f"EP99_d{delay}")
        results[f"EP99_d{delay}"] = float(np.mean(a[-2000:]))

        # Multi-timescale (PFSC-v2) — our method
        print(f"  [PFSC-v2 multi-λ]", flush=True)
        m = MultiTimescaleEprop(input_size, HIDDEN, output_size,
                                n_bands=4, lam_min=0.85, lam_max=0.995, lr=0.01)
        l, a = run_experiment(m, False, N_STEPS, BATCH, 10, delay, f"MTEp_d{delay}")
        results[f"MTEp_d{delay}"] = float(np.mean(a[-2000:]))

        # Multi-timescale with more extreme range
        print(f"  [PFSC-v2 extreme λ]", flush=True)
        m = MultiTimescaleEprop(input_size, HIDDEN, output_size,
                                n_bands=4, lam_min=0.8, lam_max=0.998, lr=0.01)
        l, a = run_experiment(m, False, N_STEPS, BATCH, 10, delay, f"MTEpX_d{delay}")
        results[f"MTEpX_d{delay}"] = float(np.mean(a[-2000:]))

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL RESULTS (elapsed {elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Method':<16} {'d=10':<8} {'d=20':<8} {'d=30':<8} {'d=50':<8}", flush=True)
    print("-"*50, flush=True)
    for prefix, label in [("BPTT", "BPTT"), ("EP95", "e-prop λ=.95"),
                          ("EP99", "e-prop λ=.99"), ("MTEp", "PFSC-v2"),
                          ("MTEpX", "PFSC-v2 extreme")]:
        row = f"{label:<16} "
        for d in delays:
            val = results.get(f"{prefix}_d{d}", 0)
            row += f"{val:<8.3f} "
        print(row, flush=True)

    print(f"\n--- Multi-timescale advantage ---", flush=True)
    for d in delays:
        ep95 = results.get(f"EP95_d{d}", 0)
        ep99 = results.get(f"EP99_d{d}", 0)
        mt = results.get(f"MTEp_d{d}", 0)
        best_uniform = max(ep95, ep99)
        print(f"  d={d}: best_uniform={best_uniform:.3f} | multi-timescale={mt:.3f} | advantage={mt-best_uniform:+.3f}", flush=True)

    with open("experiment_v5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiment_v5_results.json", flush=True)


if __name__ == "__main__":
    main()
