"""
Experiment: OPCR (Oscillatory Phase Credit Routing) Implementation and Testing
Tests on Copy Task and Adding Problem against BPTT baseline.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# TASK GENERATORS
# ============================================================

def generate_copy_task(batch_size, seq_len=10, delay=10, n_symbols=8):
    """
    Copy task: memorize a sequence of symbols, then reproduce them after a delay.
    Input: [symbols..., delimiter, zeros..., go_signal]
    Output: [zeros..., zeros..., symbols...]
    """
    total_len = seq_len + delay + seq_len + 2  # symbols + delim + delay + go + output
    n_input = n_symbols + 2  # symbols + delimiter + go signal

    # Generate random sequences
    symbols = torch.randint(0, n_symbols, (batch_size, seq_len))

    # Build input sequence
    x = torch.zeros(batch_size, total_len, n_input, device=DEVICE)
    for t in range(seq_len):
        x[:, t, :n_symbols] = torch.nn.functional.one_hot(symbols[:, t], n_symbols).float().to(DEVICE)
    x[:, seq_len, n_symbols] = 1.0  # delimiter
    x[:, seq_len + delay + 1, n_symbols + 1] = 1.0  # go signal

    # Build target (only care about the output phase)
    target = torch.zeros(batch_size, total_len, n_symbols, device=DEVICE)
    for t in range(seq_len):
        target[:, seq_len + delay + 2 + t, :] = torch.nn.functional.one_hot(symbols[:, t], n_symbols).float().to(DEVICE)

    # Mask: only compute loss on output phase
    mask = torch.zeros(batch_size, total_len, device=DEVICE)
    mask[:, seq_len + delay + 2:] = 1.0

    return x, target, mask


def generate_adding_problem(batch_size, seq_len=50):
    """
    Adding problem: two input channels. First is random [0,1], second is binary mask
    with exactly 2 ones. Output = sum of masked values.
    """
    # Random values
    values = torch.rand(batch_size, seq_len, 1, device=DEVICE)

    # Binary mask with exactly 2 ones
    mask_seq = torch.zeros(batch_size, seq_len, 1, device=DEVICE)
    for b in range(batch_size):
        positions = torch.randperm(seq_len)[:2]
        mask_seq[b, positions, 0] = 1.0

    x = torch.cat([values, mask_seq], dim=-1)  # [batch, seq_len, 2]

    # Target is sum of masked values
    target = (values * mask_seq).sum(dim=1)  # [batch, 1]

    return x, target


# ============================================================
# OPCR IMPLEMENTATION
# ============================================================

class OPCR_RNN:
    """
    Oscillatory Phase Credit Routing RNN.
    Implements the OPCR learning rule manually (no autograd).
    """

    def __init__(self, input_size, hidden_size, output_size,
                 M=8, omega_min=0.1, omega_max=0.5, alpha=0.01,
                 lam=0.95, lr=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.M = M
        self.alpha = alpha
        self.lam = lam
        self.lr = lr

        # Initialize weights
        self.W = (torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)).to(DEVICE)
        self.U = (torch.randn(hidden_size, input_size) / np.sqrt(input_size)).to(DEVICE)
        self.V = (torch.randn(output_size, hidden_size) / np.sqrt(hidden_size)).to(DEVICE)
        self.b_h = torch.zeros(hidden_size, device=DEVICE)
        self.b_o = torch.zeros(output_size, device=DEVICE)

        # Random feedback weights (fixed, for feedback alignment)
        self.B = (torch.randn(hidden_size, output_size) / np.sqrt(output_size)).to(DEVICE)

        # Oscillator frequencies (uniformly distributed)
        self.omega = torch.linspace(omega_min, omega_max, hidden_size, device=DEVICE)

        # Phase bin centers
        self.theta_m = torch.tensor([2 * np.pi * m / M for m in range(M)], device=DEVICE)
        self.kappa = np.pi / M  # bin width

    def reset_state(self, batch_size):
        """Reset hidden state and eligibility traces."""
        self.h = torch.zeros(batch_size, self.hidden_size, device=DEVICE)
        self.phi = torch.rand(batch_size, self.hidden_size, device=DEVICE) * 2 * np.pi
        # Eligibility bank: [batch, hidden, hidden, M]
        self.elig = torch.zeros(batch_size, self.hidden_size, self.hidden_size, self.M, device=DEVICE)
        # Input eligibility: [batch, hidden, input, M]
        self.elig_u = torch.zeros(batch_size, self.hidden_size, self.input_size, self.M, device=DEVICE)

    def phase_kernel(self, delta_phi, m):
        """Von Mises-like kernel for phase bin m."""
        diff = delta_phi - self.theta_m[m]
        # Wrap to [-pi, pi]
        diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
        return torch.exp(-diff**2 / (2 * self.kappa**2))

    def forward_step(self, x_t):
        """Single forward step. Returns output."""
        # Forward pass
        z = (torch.matmul(self.h, self.W.T) +
             torch.matmul(x_t, self.U.T) + self.b_h)
        h_new = torch.tanh(z)

        # Phase update
        phi_new = self.phi + self.omega.unsqueeze(0) + self.alpha * h_new
        phi_new = torch.remainder(phi_new, 2 * np.pi)

        # Compute phase differences: delta_phi[b,i,j] = phi_i - phi_j
        delta_phi = phi_new.unsqueeze(2) - self.phi.unsqueeze(1)  # [batch, N, N]
        delta_phi_u = phi_new.unsqueeze(2) - torch.zeros(x_t.shape[0], 1, self.input_size, device=DEVICE)  # simplified for input

        # Post-synaptic derivative (locally available)
        post_deriv = 1 - h_new**2  # tanh derivative [batch, N]

        # Pre-synaptic activity
        pre = self.h  # [batch, N] (previous hidden state)
        pre_input = x_t  # [batch, D]

        # Update eligibility bank
        for m in range(self.M):
            G_m = self.phase_kernel(delta_phi, m)  # [batch, N, N]
            # e[b,i,j,m] += G_m[b,i,j] * post_deriv[b,i] * pre[b,j]
            update = G_m * (post_deriv.unsqueeze(2) * pre.unsqueeze(1))
            self.elig[:,:,:,m] = self.lam * self.elig[:,:,:,m] + update

            # Input eligibility
            G_m_u = self.phase_kernel(delta_phi_u, m)  # simplified
            update_u = G_m_u * (post_deriv.unsqueeze(2) * pre_input.unsqueeze(1))
            self.elig_u[:,:,:,m] = self.lam * self.elig_u[:,:,:,m] + update_u

        # Update state
        self.h = h_new
        self.phi = phi_new

        # Output
        y_hat = torch.matmul(self.h, self.V.T) + self.b_o
        return y_hat

    def compute_update(self, error):
        """
        Compute weight updates given output error.
        error: [batch, output_size]
        """
        batch_size = error.shape[0]

        # Learning signal via random feedback
        L = torch.matmul(error, self.B.T)  # [batch, hidden]

        # Phase-selective credit — vectorized over M
        # Compute all phase credits at once
        # phi: [batch, N], theta_m: [M]
        phi_expanded = self.phi.unsqueeze(-1)  # [batch, N, 1]
        theta_expanded = self.theta_m.unsqueeze(0).unsqueeze(0)  # [1, 1, M]
        diff = phi_expanded - theta_expanded
        diff = torch.remainder(diff + np.pi, 2 * np.pi) - np.pi
        all_phase_credits = torch.exp(-diff**2 / (2 * self.kappa**2))  # [batch, N, M]

        # C_m: [batch, N, M]
        C_all = L.unsqueeze(-1) * all_phase_credits  # [batch, N, M]

        # Weight updates via einsum over all M at once
        # elig: [batch, N, N, M], C_all: [batch, N, M]
        dW = torch.einsum('bim,bijm->ij', C_all, self.elig) / batch_size
        dU = torch.einsum('bim,bidm->id', C_all, self.elig_u) / batch_size

        # Update readout weights with standard gradient (biologically plausible for output layer)
        dV = torch.einsum('bo,bh->oh', error, self.h) / batch_size

        return dW, dU, dV

    def update_weights(self, dW, dU, dV):
        """Apply weight updates."""
        self.W += self.lr * dW
        self.U += self.lr * dU
        self.V += self.lr * dV


# ============================================================
# BPTT BASELINE
# ============================================================

class BPTT_RNN(nn.Module):
    """Standard RNN trained with BPTT for baseline comparison."""

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


# ============================================================
# TRAINING LOOPS
# ============================================================

def train_opcr_copy(n_steps=5000, batch_size=32, seq_len=10, delay=10,
                    hidden_size=64, M=8, print_every=500):
    """Train OPCR on copy task."""
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    total_len = seq_len + delay + seq_len + 2

    model = OPCR_RNN(input_size, hidden_size, output_size, M=M,
                     omega_min=0.1, omega_max=0.5, alpha=0.01,
                     lam=0.95, lr=0.001)

    losses = []
    accuracies = []

    for step in range(n_steps):
        x, target, mask = generate_copy_task(batch_size, seq_len, delay, n_symbols)
        model.reset_state(batch_size)

        total_loss = 0
        outputs = []

        for t in range(total_len):
            y_hat = model.forward_step(x[:, t, :])
            outputs.append(y_hat)

            # Compute error and update at each timestep (online learning)
            if mask[:, t].sum() > 0:
                error = target[:, t, :] - y_hat
                masked_error = error * mask[:, t:t+1]
                dW, dU, dV = model.compute_update(masked_error)
                model.update_weights(dW, dU, dV)
                total_loss += (masked_error**2).sum().item()

        # Compute accuracy
        outputs = torch.stack(outputs, dim=1)  # [batch, total_len, output_size]
        output_phase = outputs[:, seq_len + delay + 2:, :]
        target_phase = target[:, seq_len + delay + 2:, :]

        pred_symbols = output_phase.argmax(dim=-1)
        true_symbols = target_phase.argmax(dim=-1)
        valid_mask = target_phase.sum(dim=-1) > 0

        if valid_mask.sum() > 0:
            acc = (pred_symbols[valid_mask] == true_symbols[valid_mask]).float().mean().item()
        else:
            acc = 0.0

        avg_loss = total_loss / (batch_size * seq_len)
        losses.append(avg_loss)
        accuracies.append(acc)

        if (step + 1) % print_every == 0:
            recent_loss = np.mean(losses[-print_every:])
            recent_acc = np.mean(accuracies[-print_every:])
            print(f"  OPCR Copy Step {step+1}/{n_steps}: Loss={recent_loss:.4f}, Acc={recent_acc:.3f}")

    return losses, accuracies


def train_bptt_copy(n_steps=5000, batch_size=32, seq_len=10, delay=10,
                    hidden_size=64, print_every=500):
    """Train BPTT baseline on copy task."""
    n_symbols = 8
    input_size = n_symbols + 2
    output_size = n_symbols
    total_len = seq_len + delay + seq_len + 2

    model = BPTT_RNN(input_size, hidden_size, output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    accuracies = []

    for step in range(n_steps):
        x, target, mask = generate_copy_task(batch_size, seq_len, delay, n_symbols)

        optimizer.zero_grad()
        output = model(x)

        # Masked MSE loss
        loss = ((output - target)**2 * mask.unsqueeze(-1)).sum() / (mask.sum() * output_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accuracy
        output_phase = output[:, seq_len + delay + 2:, :]
        target_phase = target[:, seq_len + delay + 2:, :]
        pred_symbols = output_phase.argmax(dim=-1)
        true_symbols = target_phase.argmax(dim=-1)
        valid_mask = target_phase.sum(dim=-1) > 0

        if valid_mask.sum() > 0:
            acc = (pred_symbols[valid_mask] == true_symbols[valid_mask]).float().mean().item()
        else:
            acc = 0.0

        losses.append(loss.item())
        accuracies.append(acc)

        if (step + 1) % print_every == 0:
            recent_loss = np.mean(losses[-print_every:])
            recent_acc = np.mean(accuracies[-print_every:])
            print(f"  BPTT Copy Step {step+1}/{n_steps}: Loss={recent_loss:.4f}, Acc={recent_acc:.3f}")

    return losses, accuracies


def train_opcr_adding(n_steps=5000, batch_size=32, seq_len=50,
                      hidden_size=64, M=8, print_every=500):
    """Train OPCR on adding problem."""
    input_size = 2
    output_size = 1

    model = OPCR_RNN(input_size, hidden_size, output_size, M=M,
                     omega_min=0.05, omega_max=0.4, alpha=0.005,
                     lam=0.97, lr=0.0005)

    losses = []

    for step in range(n_steps):
        x, target = generate_adding_problem(batch_size, seq_len)
        model.reset_state(batch_size)

        # Process sequence
        for t in range(seq_len):
            y_hat = model.forward_step(x[:, t, :])

        # Compute error at the end
        error = target - y_hat  # [batch, 1]
        dW, dU, dV = model.compute_update(error)
        model.update_weights(dW, dU, dV)

        loss = (error**2).mean().item()
        losses.append(loss)

        if (step + 1) % print_every == 0:
            recent_loss = np.mean(losses[-print_every:])
            print(f"  OPCR Adding Step {step+1}/{n_steps}: MSE={recent_loss:.4f}")

    return losses


def train_bptt_adding(n_steps=5000, batch_size=32, seq_len=50,
                      hidden_size=64, print_every=500):
    """Train BPTT baseline on adding problem."""
    input_size = 2
    output_size = 1

    model = BPTT_RNN(input_size, hidden_size, output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for step in range(n_steps):
        x, target = generate_adding_problem(batch_size, seq_len)

        optimizer.zero_grad()
        output = model(x, return_all=False)
        loss = ((output - target)**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % print_every == 0:
            recent_loss = np.mean(losses[-print_every:])
            print(f"  BPTT Adding Step {step+1}/{n_steps}: MSE={recent_loss:.4f}")

    return losses


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    results = {}

    print("=" * 60)
    print("EXPERIMENT: OPCR vs BPTT on Temporal Tasks")
    print("=" * 60)

    # --- Copy Task ---
    print("\n" + "=" * 60)
    print("TASK 1: Copy Task (seq_len=10, delay=10)")
    print("=" * 60)

    print("\n[1/4] Training BPTT baseline on Copy Task...")
    bptt_copy_losses, bptt_copy_acc = train_bptt_copy(
        n_steps=3000, batch_size=32, seq_len=10, delay=10, hidden_size=64)

    print("\n[2/4] Training OPCR on Copy Task...")
    opcr_copy_losses, opcr_copy_acc = train_opcr_copy(
        n_steps=3000, batch_size=32, seq_len=10, delay=10, hidden_size=64, M=8)

    # --- Adding Problem ---
    print("\n" + "=" * 60)
    print("TASK 2: Adding Problem (seq_len=50)")
    print("=" * 60)

    print("\n[3/4] Training BPTT baseline on Adding Problem...")
    bptt_add_losses = train_bptt_adding(
        n_steps=3000, batch_size=32, seq_len=50, hidden_size=64)

    print("\n[4/4] Training OPCR on Adding Problem...")
    opcr_add_losses = train_opcr_adding(
        n_steps=3000, batch_size=32, seq_len=50, hidden_size=64, M=8)

    # --- Results Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Copy task results
    bptt_copy_final_loss = np.mean(bptt_copy_losses[-500:])
    bptt_copy_final_acc = np.mean(bptt_copy_acc[-500:])
    opcr_copy_final_loss = np.mean(opcr_copy_losses[-500:])
    opcr_copy_final_acc = np.mean(opcr_copy_acc[-500:])

    print(f"\nCopy Task (final 500 steps average):")
    print(f"  BPTT:  Loss={bptt_copy_final_loss:.4f}, Accuracy={bptt_copy_final_acc:.3f}")
    print(f"  OPCR:  Loss={opcr_copy_final_loss:.4f}, Accuracy={opcr_copy_final_acc:.3f}")

    # Adding problem results
    bptt_add_final = np.mean(bptt_add_losses[-500:])
    opcr_add_final = np.mean(opcr_add_losses[-500:])
    baseline_mse = 0.167  # expected MSE for random guessing on adding problem

    print(f"\nAdding Problem (final 500 steps average):")
    print(f"  Random baseline MSE: ~{baseline_mse:.3f}")
    print(f"  BPTT:  MSE={bptt_add_final:.4f}")
    print(f"  OPCR:  MSE={opcr_add_final:.4f}")

    # Assessment
    print(f"\n--- Assessment ---")
    opcr_learns_copy = opcr_copy_final_acc > 0.15  # better than random (1/8 = 0.125)
    opcr_learns_adding = opcr_add_final < baseline_mse * 0.9

    print(f"  OPCR learns on Copy Task: {'YES' if opcr_learns_copy else 'NO'} (acc={opcr_copy_final_acc:.3f} vs random=0.125)")
    print(f"  OPCR learns on Adding Problem: {'YES' if opcr_learns_adding else 'NO'} (mse={opcr_add_final:.4f} vs random={baseline_mse:.3f})")

    if opcr_learns_copy or opcr_learns_adding:
        print("\n  >>> OPCR shows evidence of learning! Algorithm is functional.")
    else:
        print("\n  >>> OPCR does not show clear learning. May need hyperparameter tuning or algorithm revision.")

    # Save results
    results = {
        "copy_task": {
            "bptt_final_loss": float(bptt_copy_final_loss),
            "bptt_final_acc": float(bptt_copy_final_acc),
            "opcr_final_loss": float(opcr_copy_final_loss),
            "opcr_final_acc": float(opcr_copy_final_acc),
        },
        "adding_problem": {
            "bptt_final_mse": float(bptt_add_final),
            "opcr_final_mse": float(opcr_add_final),
            "random_baseline_mse": baseline_mse,
        },
        "hyperparameters": {
            "hidden_size": 64,
            "M_phase_bins": 8,
            "omega_range": [0.1, 0.5],
            "alpha": 0.01,
            "lambda": 0.95,
            "lr": 0.001,
            "n_steps": 3000,
        }
    }

    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to experiment_results.json")
    return results


if __name__ == "__main__":
    main()
