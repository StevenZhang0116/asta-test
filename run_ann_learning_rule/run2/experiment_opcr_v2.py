"""
Experiment v2: OPCR with fixed scaling, extended training, and ablation study.
Key fixes: normalized phase kernels, increased lambda, smaller hidden for speed.
Uses hidden_size=64 and batch=32 to fit in reasonable runtime.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import json
import time

sys.stdout = sys.stderr  # Force unbuffered output
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)


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


class OPCR_RNN_v2:
    """OPCR v2: Fixed phase kernel normalization, improved scaling."""

    def __init__(self, input_size, hidden_size, output_size,
                 M=8, omega_min=0.1, omega_max=0.5, alpha=0.01,
                 lam=0.98, lr=0.003, use_phase=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.M = M
        self.alpha = alpha
        self.lam = lam
        self.lr = lr
        self.use_phase = use_phase

        self.W = (torch.randn(hidden_size, hidden_size) * 0.1).to(DEVICE)
        self.U = (torch.randn(hidden_size, input_size) / np.sqrt(input_size)).to(DEVICE)
        self.V = (torch.randn(output_size, hidden_size) / np.sqrt(hidden_size)).to(DEVICE)
        self.b_h = torch.zeros(hidden_size, device=DEVICE)
        self.b_o = torch.zeros(output_size, device=DEVICE)

        self.B = (torch.randn(hidden_size, output_size) / np.sqrt(output_size)).to(DEVICE)
        self.omega = torch.linspace(omega_min, omega_max, hidden_size, device=DEVICE)
        self.theta_m = torch.tensor([2 * np.pi * m / M for m in range(M)], device=DEVICE)
        self.kappa = 2 * np.pi / (2 * M)

    def reset_state(self, batch_size):
        self.h = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.phi = torch.rand(batch_size, self.hidden_size, device=DEVICE) * 2 * np.pi
        self.elig = torch.zeros(batch_size, self.hidden_size, self.hidden_size, self.M, device=DEVICE)
        self.elig_u = torch.zeros(batch_size, self.hidden_size, self.input_size, self.M, device=DEVICE)

    def forward_step(self, x_t):
        z = torch.matmul(self.h, self.W.T) + torch.matmul(x_t, self.U.T) + self.b_h
        h_new = torch.tanh(z)

        phi_new = self.phi + self.omega.unsqueeze(0) + self.alpha * h_new
        phi_new = torch.remainder(phi_new, 2 * np.pi)

        # Phase-gated eligibility update (vectorized)
        delta_phi_rec = phi_new.unsqueeze(2) - self.phi.unsqueeze(1)  # [B, N, N]
        delta_phi_inp = phi_new.unsqueeze(2).expand(-1, -1, self.input_size)  # [B, N, D]

        post_deriv = 1 - h_new**2  # [B, N]
        pre = self.h  # [B, N]

        # Compute phase kernels (softmax normalized)
        def get_kernels(delta_phi):
            dp = delta_phi.unsqueeze(-1)  # [..., 1]
            tm = self.theta_m  # [M]
            diff = dp - tm
            diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
            logits = -diff**2 / (2 * self.kappa**2)
            if self.use_phase:
                return torch.softmax(logits, dim=-1)
            else:
                return torch.ones_like(logits) / self.M

        G_rec = get_kernels(delta_phi_rec)  # [B, N, N, M]
        G_inp = get_kernels(delta_phi_inp)  # [B, N, D, M]

        # Eligibility update
        elig_update_rec = G_rec * (post_deriv.unsqueeze(2).unsqueeze(3) * pre.unsqueeze(1).unsqueeze(3))
        self.elig = self.lam * self.elig + elig_update_rec

        elig_update_inp = G_inp * (post_deriv.unsqueeze(2).unsqueeze(3) * x_t.unsqueeze(1).unsqueeze(3))
        self.elig_u = self.lam * self.elig_u + elig_update_inp

        self.h = h_new
        self.phi = phi_new
        y_hat = torch.matmul(self.h, self.V.T) + self.b_o
        return y_hat

    def compute_update(self, error):
        batch_size = error.shape[0]
        L = torch.matmul(error, self.B.T)  # [B, N]

        if self.use_phase:
            phi_exp = self.phi.unsqueeze(-1)  # [B, N, 1]
            diff = phi_exp - self.theta_m.unsqueeze(0).unsqueeze(0)
            diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
            phase_credit = torch.softmax(-diff**2 / (2 * self.kappa**2), dim=-1)  # [B, N, M]
        else:
            phase_credit = torch.ones(batch_size, self.hidden_size, self.M, device=DEVICE) / self.M

        C_all = L.unsqueeze(-1) * phase_credit  # [B, N, M]

        dW = torch.einsum('bim,bijm->ij', C_all, self.elig) / batch_size
        dU = torch.einsum('bim,bidm->id', C_all, self.elig_u) / batch_size
        dV = torch.einsum('bo,bh->oh', error, self.h) / batch_size
        return dW, dU, dV

    def update_weights(self, dW, dU, dV):
        dW = torch.clamp(dW, -1.0, 1.0)
        dU = torch.clamp(dU, -1.0, 1.0)
        dV = torch.clamp(dV, -1.0, 1.0)
        self.W += self.lr * dW
        self.U += self.lr * dU
        self.V += self.lr * dV


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


def train_copy(model_type, n_steps, batch_size, seq_len, delay, hidden_size,
               M=8, lr=0.003, lam=0.98, use_phase=True):
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    total_len = seq_len + delay + seq_len + 2

    if model_type == "bptt":
        model = BPTT_RNN(input_size, hidden_size, output_size).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        model = OPCR_RNN_v2(input_size, hidden_size, output_size, M=M,
                            omega_min=0.1, omega_max=0.5, alpha=0.01,
                            lam=lam, lr=lr, use_phase=use_phase)

    losses, accuracies = [], []

    for step in range(n_steps):
        x, target, mask = generate_copy_task(batch_size, seq_len, delay, n_symbols)

        if model_type == "bptt":
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * output_size + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            output_phase = output[:, seq_len + delay + 2:, :].detach()
            losses.append(loss.item())
        else:
            model.reset_state(batch_size)
            total_loss = 0.0
            outputs = []
            for t in range(total_len):
                y_hat = model.forward_step(x[:, t, :])
                outputs.append(y_hat.detach().clone())
                if mask[:, t].sum() > 0:
                    error = target[:, t, :] - y_hat
                    masked_error = error * mask[:, t:t+1]
                    dW, dU, dV = model.compute_update(masked_error)
                    model.update_weights(dW, dU, dV)
                    total_loss += (masked_error**2).mean().item()
            outputs = torch.stack(outputs, dim=1)
            output_phase = outputs[:, seq_len + delay + 2:, :]
            losses.append(total_loss / max(seq_len, 1))

        target_phase = target[:, seq_len + delay + 2:, :]
        pred = output_phase.argmax(dim=-1)
        true = target_phase.argmax(dim=-1)
        valid = target_phase.sum(dim=-1) > 0
        acc = (pred[valid] == true[valid]).float().mean().item() if valid.sum() > 0 else 0.0
        accuracies.append(acc)

        if (step + 1) % 2000 == 0:
            rl = np.mean(losses[-2000:])
            ra = np.mean(accuracies[-2000:])
            tag = model_type.upper() + ("" if use_phase else "-NOPHASE")
            print(f"    {tag} d={delay} step {step+1}: loss={rl:.4f} acc={ra:.3f}", flush=True)

    return losses, accuracies


def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT v2: OPCR (fixed) + ablation + extended training", flush=True)
    print(f"Hidden=64, Batch=32, Steps=10000, M=8", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    results = {}
    N_STEPS = 10000
    BATCH = 32
    HIDDEN = 64

    configs = [
        ("bptt_d5", "bptt", 5, True),
        ("bptt_d10", "bptt", 10, True),
        ("opcr_d5", "opcr", 5, True),
        ("opcr_d10", "opcr", 10, True),
        ("nophase_d5", "opcr", 5, False),
        ("nophase_d10", "opcr", 10, False),
    ]

    for name, mtype, delay, use_phase in configs:
        print(f"\n--- {name.upper()} ---", flush=True)
        losses, accs = train_copy(mtype, N_STEPS, BATCH, 10, delay, HIDDEN,
                                  M=8, lr=0.005, lam=0.98, use_phase=use_phase)
        fl = np.mean(losses[-2000:])
        fa = np.mean(accs[-2000:])
        pa = max(np.mean(accs[i:i+500]) for i in range(0, len(accs)-499, 200))
        results[name] = {"final_loss": float(fl), "final_acc": float(fa), "peak_acc": float(pa)}
        print(f"  DONE: final_acc={fa:.3f}, peak_acc={pa:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY (elapsed {elapsed:.0f}s)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Method':<15} {'Delay':<6} {'Final Acc':<10} {'Peak Acc':<10}", flush=True)
    print("-" * 45, flush=True)
    for name, r in results.items():
        parts = name.split("_")
        print(f"{parts[0].upper():<15} {parts[1]:<6} {r['final_acc']:<10.3f} {r['peak_acc']:<10.3f}", flush=True)

    # Phase contribution
    print(f"\n--- Phase Contribution Analysis ---", flush=True)
    for d in [5, 10]:
        opcr = results.get(f"opcr_d{d}", {}).get("final_acc", 0)
        noph = results.get(f"nophase_d{d}", {}).get("final_acc", 0)
        bptt = results.get(f"bptt_d{d}", {}).get("final_acc", 0)
        print(f"  Delay={d}: BPTT={bptt:.3f} | OPCR={opcr:.3f} | NoPhase={noph:.3f} | Phase_effect={opcr-noph:+.3f}", flush=True)

    with open("experiment_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiment_v2_results.json", flush=True)


if __name__ == "__main__":
    main()
