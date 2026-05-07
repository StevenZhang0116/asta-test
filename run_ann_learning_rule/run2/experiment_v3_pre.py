"""
Experiment v3: Phase-Resonant Eligibility (PRE)

Key change from OPCR: Instead of splitting eligibility into phase bins,
use phase to MODULATE eligibility decay rate. This creates frequency-selective
temporal credit without diluting the signal.

Also includes proper e-prop baseline and end-of-sequence updates.
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


class EpropBaseline:
    """Standard e-prop with random feedback (no phase). Proper implementation."""

    def __init__(self, input_size, hidden_size, output_size, lam=0.95, lr=0.01):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
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
        post_deriv = 1 - h_new**2  # [B, N]

        # e-prop eligibility: e_ij(t) = lam * e_ij(t-1) + (dh_i/dz_i) * h_j(t-1)
        self.elig_W = self.lam * self.elig_W + torch.einsum('bi,bj->bij', post_deriv, self.h)
        self.elig_U = self.lam * self.elig_U + torch.einsum('bi,bd->bid', post_deriv, x_t)

        self.h = h_new
        return torch.matmul(self.h, self.V.T) + self.b_o

    def accumulate_update(self, error, mask_t):
        """Accumulate updates for timesteps where mask is active."""
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
            self.W += self.lr * self.dW_accum / self.n_updates
            self.U += self.lr * self.dU_accum / self.n_updates
            self.V += self.lr * self.dV_accum / self.n_updates


class PhaseResonantEligibility:
    """
    PRE: Phase-Resonant Eligibility.

    Key idea: Eligibility decay rate is modulated by phase alignment.
    - In-phase pairs: slower decay (longer temporal credit)
    - Anti-phase pairs: faster decay (shorter temporal credit)

    This creates frequency-selective temporal windows without signal dilution.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 lam_base=0.9, lam_delta=0.08,
                 omega_min=0.1, omega_max=0.6, alpha=0.005,
                 lr=0.01):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.lam_base = lam_base
        self.lam_delta = lam_delta
        self.alpha = alpha
        self.lr = lr

        scale_h = 1.0 / np.sqrt(hidden_size)
        scale_i = 1.0 / np.sqrt(input_size)
        self.W = (torch.randn(hidden_size, hidden_size) * scale_h).to(DEVICE)
        self.U = (torch.randn(hidden_size, input_size) * scale_i).to(DEVICE)
        self.V = (torch.randn(output_size, hidden_size) * scale_h).to(DEVICE)
        self.b_h = torch.zeros(hidden_size, device=DEVICE)
        self.b_o = torch.zeros(output_size, device=DEVICE)
        self.B = (torch.randn(hidden_size, output_size) / np.sqrt(output_size)).to(DEVICE)

        self.omega = torch.linspace(omega_min, omega_max, hidden_size, device=DEVICE)

    def reset(self, batch_size):
        self.h = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.phi = torch.rand(batch_size, self.hidden_size, device=DEVICE) * 2 * np.pi
        self.elig_W = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=DEVICE)
        self.elig_U = torch.zeros(batch_size, self.hidden_size, self.input_size, device=DEVICE)
        self.dW_accum = torch.zeros_like(self.W)
        self.dU_accum = torch.zeros_like(self.U)
        self.dV_accum = torch.zeros_like(self.V)
        self.n_updates = 0

    def forward_step(self, x_t):
        z = torch.matmul(self.h, self.W.T) + torch.matmul(x_t, self.U.T) + self.b_h
        h_new = torch.tanh(z)

        # Phase update
        phi_new = self.phi + self.omega.unsqueeze(0) + self.alpha * h_new
        phi_new = torch.remainder(phi_new, 2 * np.pi)

        # Phase-modulated decay: lambda_ij = lam_base + lam_delta * cos(phi_i - phi_j)
        # cos(phi_i - phi_j) is high when in-phase, low when anti-phase
        delta_phi = phi_new.unsqueeze(2) - self.phi.unsqueeze(1)  # [B, N_post, N_pre]
        lam_dynamic = self.lam_base + self.lam_delta * torch.cos(delta_phi)  # [B, N, N]
        lam_dynamic = torch.clamp(lam_dynamic, 0.0, 0.99)

        # For input eligibility, use just post phase (inputs have no phase)
        lam_input = self.lam_base + self.lam_delta * torch.cos(phi_new)  # [B, N]
        lam_input = torch.clamp(lam_input, 0.0, 0.99).unsqueeze(2)  # [B, N, 1]

        post_deriv = 1 - h_new**2  # [B, N]

        # Phase-resonant eligibility update
        self.elig_W = lam_dynamic * self.elig_W + torch.einsum('bi,bj->bij', post_deriv, self.h)
        self.elig_U = lam_input * self.elig_U + torch.einsum('bi,bd->bid', post_deriv, x_t)

        self.h = h_new
        self.phi = phi_new
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
            self.W += self.lr * torch.clamp(self.dW_accum * scale, -1, 1)
            self.U += self.lr * torch.clamp(self.dU_accum * scale, -1, 1)
            self.V += self.lr * torch.clamp(self.dV_accum * scale, -1, 1)


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


def train_and_eval(model_class, model_kwargs, n_steps, batch_size, seq_len, delay, label):
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    total_len = seq_len + delay + seq_len + 2

    if model_class == "bptt":
        model = BPTT_RNN(input_size, model_kwargs["hidden_size"], output_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        model = model_class(input_size, model_kwargs["hidden_size"], output_size, **{k:v for k,v in model_kwargs.items() if k != "hidden_size"})

    losses, accs = [], []
    for step in range(n_steps):
        x, target, mask = generate_copy_task(batch_size, seq_len, delay, n_symbols)

        if model_class == "bptt":
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * output_size + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            output_phase = output[:, seq_len+delay+2:, :].detach()
            losses.append(loss.item())
        else:
            model.reset(batch_size)
            outputs = []
            for t in range(total_len):
                y_hat = model.forward_step(x[:, t, :])
                outputs.append(y_hat.detach().clone())
                error = target[:, t, :] - y_hat
                model.accumulate_update(error, mask[:, t])
            model.apply_update()
            outputs = torch.stack(outputs, dim=1)
            output_phase = outputs[:, seq_len+delay+2:, :]
            # compute loss for logging
            loss_val = ((outputs - target)**2 * mask.unsqueeze(-1)).sum().item() / (mask.sum().item() * output_size + 1e-8)
            losses.append(loss_val)

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
    print("EXPERIMENT v3: Phase-Resonant Eligibility (PRE) + Proper e-prop baseline", flush=True)
    print("="*70, flush=True)
    t0 = time.time()

    N_STEPS = 10000
    BATCH = 32
    HIDDEN = 64
    results = {}

    configs = [
        ("BPTT_d5", "bptt", {"hidden_size": HIDDEN}, 5),
        ("BPTT_d10", "bptt", {"hidden_size": HIDDEN}, 10),
        ("BPTT_d20", "bptt", {"hidden_size": HIDDEN}, 20),
        ("EPROP_d5", EpropBaseline, {"hidden_size": HIDDEN, "lam": 0.95, "lr": 0.01}, 5),
        ("EPROP_d10", EpropBaseline, {"hidden_size": HIDDEN, "lam": 0.95, "lr": 0.01}, 10),
        ("EPROP_d20", EpropBaseline, {"hidden_size": HIDDEN, "lam": 0.95, "lr": 0.01}, 20),
        ("PRE_d5", PhaseResonantEligibility, {"hidden_size": HIDDEN, "lam_base": 0.9, "lam_delta": 0.08, "lr": 0.01, "omega_min": 0.1, "omega_max": 0.6, "alpha": 0.005}, 5),
        ("PRE_d10", PhaseResonantEligibility, {"hidden_size": HIDDEN, "lam_base": 0.9, "lam_delta": 0.08, "lr": 0.01, "omega_min": 0.1, "omega_max": 0.6, "alpha": 0.005}, 10),
        ("PRE_d20", PhaseResonantEligibility, {"hidden_size": HIDDEN, "lam_base": 0.9, "lam_delta": 0.08, "lr": 0.01, "omega_min": 0.1, "omega_max": 0.6, "alpha": 0.005}, 20),
    ]

    for label, model_class, kwargs, delay in configs:
        print(f"\n--- {label} ---", flush=True)
        losses, accs = train_and_eval(model_class, kwargs, N_STEPS, BATCH, 10, delay, label)
        fl = np.mean(losses[-2000:])
        fa = np.mean(accs[-2000:])
        pa = max(np.mean(accs[i:i+500]) for i in range(0, len(accs)-499, 200))
        results[label] = {"final_loss": float(fl), "final_acc": float(fa), "peak_acc": float(pa)}
        print(f"  → final_acc={fa:.3f}, peak={pa:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS (elapsed {elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Method':<12} {'Delay':<6} {'Final Acc':<10} {'Peak Acc':<10}", flush=True)
    print("-"*40, flush=True)
    for label, r in results.items():
        print(f"{label:<12} {r['final_acc']:<10.3f} {r['peak_acc']:<10.3f}", flush=True)

    # Key comparison: does phase-resonant decay help over fixed decay?
    print(f"\n--- Phase-Resonant Eligibility vs Standard e-prop ---", flush=True)
    for d in [5, 10, 20]:
        ep = results.get(f"EPROP_d{d}", {}).get("final_acc", 0)
        pr = results.get(f"PRE_d{d}", {}).get("final_acc", 0)
        bp = results.get(f"BPTT_d{d}", {}).get("final_acc", 0)
        diff = pr - ep
        print(f"  Delay={d}: BPTT={bp:.3f} | e-prop={ep:.3f} | PRE={pr:.3f} | PRE-eprop={diff:+.3f}", flush=True)

    with open("experiment_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiment_v3_results.json", flush=True)


if __name__ == "__main__":
    main()
