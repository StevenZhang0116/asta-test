"""
Experiment v4: Phase-Frequency Spectrum Credit (PFSC)

Core idea: Neurons are organized into frequency bands. Each band has a matched
eligibility decay rate (slow frequency = slow decay = long temporal memory).
This creates a natural multi-timescale decomposition for credit assignment.

Focus on LONG DELAY regime (d=20, 30, 50) where BPTT fails due to vanishing gradients.
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


class PFSC_RNN:
    """
    Phase-Frequency Spectrum Credit (PFSC).

    Neurons are organized into K frequency bands, each with matched eligibility decay:
    - Band k has frequency omega_k and eligibility decay lambda_k = 1 - c/omega_k
    - Slow bands (low omega): long memory, captures long-range credit
    - Fast bands (high omega): short memory, captures precise short-range credit

    The readout combines information from all bands, learning which timescales
    are relevant for the task.

    Additionally, within-band lateral connections allow phase synchronization,
    creating coherent "ensembles" at each timescale.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 n_bands=4, lam_min=0.85, lam_max=0.995,
                 omega_min=0.05, omega_max=0.8, alpha=0.003,
                 lr=0.01, use_phase_sync=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_bands = n_bands
        self.band_size = hidden_size // n_bands
        self.alpha = alpha
        self.lr = lr
        self.use_phase_sync = use_phase_sync

        # Assign neurons to bands
        self.band_assignment = torch.zeros(hidden_size, dtype=torch.long, device=DEVICE)
        for k in range(n_bands):
            start = k * self.band_size
            end = start + self.band_size
            self.band_assignment[start:end] = k

        # Frequency and decay per band (logarithmically spaced)
        log_omegas = torch.linspace(np.log(omega_min), np.log(omega_max), n_bands)
        self.omega_per_band = torch.exp(log_omegas).to(DEVICE)  # [K]

        # Lambda matched to frequency: slow oscillation = slow decay
        self.lam_per_band = torch.linspace(lam_max, lam_min, n_bands).to(DEVICE)  # [K] slow→fast

        # Expand to per-neuron
        self.omega = torch.zeros(hidden_size, device=DEVICE)
        self.lam = torch.zeros(hidden_size, device=DEVICE)
        for k in range(n_bands):
            mask_k = self.band_assignment == k
            self.omega[mask_k] = self.omega_per_band[k]
            self.lam[mask_k] = self.lam_per_band[k]

        # Weights
        scale_h = 1.0 / np.sqrt(hidden_size)
        scale_i = 1.0 / np.sqrt(input_size)
        self.W = (torch.randn(hidden_size, hidden_size) * scale_h).to(DEVICE)
        self.U = (torch.randn(hidden_size, input_size) * scale_i).to(DEVICE)
        self.V = (torch.randn(output_size, hidden_size) * scale_h).to(DEVICE)
        self.b_h = torch.zeros(hidden_size, device=DEVICE)
        self.b_o = torch.zeros(output_size, device=DEVICE)

        # Random feedback (feedback alignment)
        self.B = (torch.randn(hidden_size, output_size) / np.sqrt(output_size)).to(DEVICE)

        # Within-band lateral coupling (for phase sync)
        if use_phase_sync:
            self.W_lateral = torch.zeros(hidden_size, hidden_size, device=DEVICE)
            for k in range(n_bands):
                start = k * self.band_size
                end = start + self.band_size
                # Small random coupling within band
                self.W_lateral[start:end, start:end] = torch.randn(self.band_size, self.band_size, device=DEVICE) * 0.02

    def reset(self, batch_size):
        self.h = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.phi = torch.rand(batch_size, self.hidden_size, device=DEVICE) * 2 * np.pi
        # Per-neuron eligibility (NOT per-synapse-pair to save memory)
        # Use factored eligibility: e_ij ≈ trace_i * pre_j (rank-1 per neuron)
        self.trace = torch.zeros(batch_size, self.hidden_size, device=DEVICE)  # post-synaptic trace
        self.pre_trace = torch.zeros(batch_size, self.hidden_size, device=DEVICE)  # pre-synaptic trace
        self.pre_trace_u = torch.zeros(batch_size, self.input_size, device=DEVICE)  # input trace
        self.dW_accum = torch.zeros_like(self.W)
        self.dU_accum = torch.zeros_like(self.U)
        self.dV_accum = torch.zeros_like(self.V)
        self.n_updates = 0

    def forward_step(self, x_t):
        # Standard RNN forward
        z = torch.matmul(self.h, self.W.T) + torch.matmul(x_t, self.U.T) + self.b_h
        h_new = torch.tanh(z)

        # Phase update with activity coupling
        phi_new = self.phi + self.omega.unsqueeze(0) + self.alpha * h_new
        if self.use_phase_sync:
            # Phase synchronization within bands via lateral coupling
            sync_signal = torch.matmul(torch.sin(self.phi), self.W_lateral.T)
            phi_new = phi_new + 0.01 * sync_signal
        phi_new = torch.remainder(phi_new, 2 * np.pi)

        # Per-neuron eligibility trace with frequency-matched decay
        post_deriv = 1 - h_new**2  # [B, N]

        # Update traces with band-specific decay
        lam_expanded = self.lam.unsqueeze(0)  # [1, N]
        self.trace = lam_expanded * self.trace + post_deriv  # [B, N]
        self.pre_trace = lam_expanded * self.pre_trace + self.h  # [B, N] (pre = previous h)
        self.pre_trace_u = self.lam.mean() * self.pre_trace_u + x_t  # [B, D]

        self.h = h_new
        self.phi = phi_new
        return torch.matmul(self.h, self.V.T) + self.b_o

    def accumulate_update(self, error, mask_t):
        if mask_t.sum() == 0:
            return
        batch_size = error.shape[0]
        masked_error = error * mask_t.unsqueeze(-1)

        # Learning signal via feedback alignment
        L = torch.matmul(masked_error, self.B.T)  # [B, N]

        # Weight update using factored eligibility:
        # dW_ij = L_i * trace_i * pre_trace_j (outer product of modulated trace with pre trace)
        modulated_trace = L * self.trace  # [B, N]

        # dW = mean_batch(outer(modulated_trace, pre_trace))
        self.dW_accum += torch.einsum('bi,bj->ij', modulated_trace, self.pre_trace) / batch_size
        self.dU_accum += torch.einsum('bi,bd->id', modulated_trace, self.pre_trace_u) / batch_size
        self.dV_accum += torch.einsum('bo,bh->oh', masked_error, self.h) / batch_size
        self.n_updates += 1

    def apply_update(self):
        if self.n_updates > 0:
            scale = 1.0 / self.n_updates
            self.W += self.lr * torch.clamp(self.dW_accum * scale, -0.5, 0.5)
            self.U += self.lr * torch.clamp(self.dU_accum * scale, -0.5, 0.5)
            self.V += self.lr * torch.clamp(self.dV_accum * scale, -0.5, 0.5)


class EpropBaseline:
    """Standard e-prop (factored version for fair comparison)."""

    def __init__(self, input_size, hidden_size, output_size, lam=0.95, lr=0.01):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lr = lr
        self.lam = lam

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
        self.trace = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.pre_trace = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.pre_trace_u = torch.zeros(batch_size, self.input_size, device=DEVICE)
        self.dW_accum = torch.zeros_like(self.W)
        self.dU_accum = torch.zeros_like(self.U)
        self.dV_accum = torch.zeros_like(self.V)
        self.n_updates = 0

    def forward_step(self, x_t):
        z = torch.matmul(self.h, self.W.T) + torch.matmul(x_t, self.U.T) + self.b_h
        h_new = torch.tanh(z)
        post_deriv = 1 - h_new**2
        self.trace = self.lam * self.trace + post_deriv
        self.pre_trace = self.lam * self.pre_trace + self.h
        self.pre_trace_u = self.lam * self.pre_trace_u + x_t
        self.h = h_new
        return torch.matmul(self.h, self.V.T) + self.b_o

    def accumulate_update(self, error, mask_t):
        if mask_t.sum() == 0:
            return
        batch_size = error.shape[0]
        masked_error = error * mask_t.unsqueeze(-1)
        L = torch.matmul(masked_error, self.B.T)
        modulated_trace = L * self.trace
        self.dW_accum += torch.einsum('bi,bj->ij', modulated_trace, self.pre_trace) / batch_size
        self.dU_accum += torch.einsum('bi,bd->id', modulated_trace, self.pre_trace_u) / batch_size
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


def train_eval(model_obj, is_bptt, n_steps, batch_size, seq_len, delay, label):
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
            loss_val = ((output_all - target)**2 * mask.unsqueeze(-1)).sum().item() / (mask.sum().item() * output_size + 1e-8)
            losses.append(loss_val)

        output_phase = output_all[:, seq_len+delay+2:, :]
        target_phase = target[:, seq_len+delay+2:, :]
        pred = output_phase.argmax(dim=-1)
        true = target_phase.argmax(dim=-1)
        valid = target_phase.sum(dim=-1) > 0
        acc = (pred[valid] == true[valid]).float().mean().item() if valid.sum() > 0 else 0.0
        accs.append(acc)

        if (step+1) % 2000 == 0:
            rl = np.mean(losses[-2000:])
            ra = np.mean(accs[-2000:])
            print(f"    {label} step {step+1}: loss={rl:.4f} acc={ra:.3f}", flush=True)

    return losses, accs


def main():
    print("="*70, flush=True)
    print("EXPERIMENT v4: PFSC — Frequency-Stratified Temporal Credit", flush=True)
    print("Focus: Long-delay regime where BPTT fails", flush=True)
    print("="*70, flush=True)
    t0 = time.time()

    N_STEPS = 10000
    BATCH = 32
    HIDDEN = 64
    SEQ_LEN = 10
    results = {}

    delays = [10, 20, 30, 50]

    for delay in delays:
        print(f"\n{'='*70}", flush=True)
        print(f"DELAY = {delay}", flush=True)
        print(f"{'='*70}", flush=True)

        # BPTT
        print(f"\n  --- BPTT ---", flush=True)
        n_symbols = 8
        input_size = n_symbols + 2
        output_size = n_symbols
        model_bptt = BPTT_RNN(input_size, HIDDEN, output_size).to(DEVICE)
        l, a = train_eval(model_bptt, True, N_STEPS, BATCH, SEQ_LEN, delay, f"BPTT_d{delay}")
        results[f"BPTT_d{delay}"] = {"final_acc": float(np.mean(a[-2000:])), "peak": float(max(np.mean(a[i:i+500]) for i in range(0, len(a)-499, 200)))}

        # e-prop (uniform lambda)
        print(f"\n  --- e-prop ---", flush=True)
        model_ep = EpropBaseline(input_size, HIDDEN, output_size, lam=0.95, lr=0.01)
        l, a = train_eval(model_ep, False, N_STEPS, BATCH, SEQ_LEN, delay, f"EPROP_d{delay}")
        results[f"EPROP_d{delay}"] = {"final_acc": float(np.mean(a[-2000:])), "peak": float(max(np.mean(a[i:i+500]) for i in range(0, len(a)-499, 200)))}

        # PFSC (frequency-stratified)
        print(f"\n  --- PFSC ---", flush=True)
        model_pfsc = PFSC_RNN(input_size, HIDDEN, output_size,
                              n_bands=4, lam_min=0.85, lam_max=0.995,
                              omega_min=0.05, omega_max=0.8, alpha=0.003,
                              lr=0.01, use_phase_sync=True)
        l, a = train_eval(model_pfsc, False, N_STEPS, BATCH, SEQ_LEN, delay, f"PFSC_d{delay}")
        results[f"PFSC_d{delay}"] = {"final_acc": float(np.mean(a[-2000:])), "peak": float(max(np.mean(a[i:i+500]) for i in range(0, len(a)-499, 200)))}

        # PFSC without phase sync (ablation)
        print(f"\n  --- PFSC-noSync ---", flush=True)
        model_pfsc_ns = PFSC_RNN(input_size, HIDDEN, output_size,
                                 n_bands=4, lam_min=0.85, lam_max=0.995,
                                 omega_min=0.05, omega_max=0.8, alpha=0.003,
                                 lr=0.01, use_phase_sync=False)
        l, a = train_eval(model_pfsc_ns, False, N_STEPS, BATCH, SEQ_LEN, delay, f"PFSC-NS_d{delay}")
        results[f"PFSC-NS_d{delay}"] = {"final_acc": float(np.mean(a[-2000:])), "peak": float(max(np.mean(a[i:i+500]) for i in range(0, len(a)-499, 200)))}

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"FULL RESULTS (elapsed {elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Method':<12} {'d=10':<8} {'d=20':<8} {'d=30':<8} {'d=50':<8}", flush=True)
    print("-"*45, flush=True)
    for method in ["BPTT", "EPROP", "PFSC", "PFSC-NS"]:
        row = f"{method:<12} "
        for d in delays:
            key = f"{method}_d{d}"
            acc = results.get(key, {}).get("final_acc", 0)
            row += f"{acc:<8.3f} "
        print(row, flush=True)

    print(f"\n--- PFSC Advantage over e-prop (per delay) ---", flush=True)
    for d in delays:
        ep = results.get(f"EPROP_d{d}", {}).get("final_acc", 0)
        pf = results.get(f"PFSC_d{d}", {}).get("final_acc", 0)
        bp = results.get(f"BPTT_d{d}", {}).get("final_acc", 0)
        print(f"  d={d}: BPTT={bp:.3f} | e-prop={ep:.3f} | PFSC={pf:.3f} | PFSC-eprop={pf-ep:+.3f}", flush=True)

    with open("experiment_v4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiment_v4_results.json", flush=True)


if __name__ == "__main__":
    main()
