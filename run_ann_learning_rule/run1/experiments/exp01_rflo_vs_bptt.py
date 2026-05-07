"""
Experiment 01: RFLO vs BPTT on Copy Task and Adding Problem
Implements RFLO (Murray, 2019) for vanilla RNN and compares against BPTT.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp01')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Data Generation
# ============================================================

def generate_copy_task(batch_size, seq_len=10, n_symbols=8):
    """Generate copy task data. Input: T symbols, T blanks, 1 go signal. Output: T symbols after go."""
    total_len = 2 * seq_len + 1
    input_dim = n_symbols + 2  # symbols + blank + go

    symbols = torch.randint(0, n_symbols, (batch_size, seq_len))

    inputs = torch.zeros(batch_size, total_len, input_dim)
    for b in range(batch_size):
        for t in range(seq_len):
            inputs[b, t, symbols[b, t]] = 1.0
        inputs[b, seq_len:2*seq_len, n_symbols] = 1.0  # blank
        inputs[b, 2*seq_len, n_symbols + 1] = 1.0  # go signal

    targets = symbols  # shape: (batch, seq_len)
    return inputs.to(DEVICE), targets.to(DEVICE)


def generate_adding_problem(batch_size, seq_len=30):
    """Generate adding problem data. Two marked numbers must be summed."""
    numbers = torch.rand(batch_size, seq_len)
    mask = torch.zeros(batch_size, seq_len)
    for b in range(batch_size):
        indices = torch.randperm(seq_len)[:2]
        mask[b, indices] = 1.0

    inputs = torch.stack([numbers, mask], dim=2)  # (batch, seq_len, 2)
    targets = (numbers * mask).sum(dim=1, keepdim=True)  # (batch, 1)
    return inputs.to(DEVICE), targets.to(DEVICE)


# ============================================================
# Vanilla RNN with BPTT
# ============================================================

class VanillaRNN_BPTT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        self.W_in = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.W_rec = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * (1.0 / np.sqrt(hidden_dim)))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        self.b_out = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, return_all=False):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        hiddens = []
        for t in range(seq_len):
            a = self.W_rec @ h.unsqueeze(-1) + self.W_in @ x[:, t].unsqueeze(-1) + self.b.unsqueeze(-1)
            a = a.squeeze(-1)
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(a)
            hiddens.append(h)

        if return_all:
            hiddens = torch.stack(hiddens, dim=1)  # (batch, seq_len, hidden)
            outputs = (self.W_out @ hiddens.unsqueeze(-1)).squeeze(-1) + self.b_out
            return outputs
        else:
            y = self.W_out @ h.unsqueeze(-1)
            y = y.squeeze(-1) + self.b_out
            return y


# ============================================================
# Vanilla RNN with RFLO
# ============================================================

class VanillaRNN_RFLO:
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.2, lr=0.01):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.lr = lr

        self.W_in = torch.randn(hidden_dim, input_dim, device=DEVICE) * 0.01
        self.W_rec = torch.randn(hidden_dim, hidden_dim, device=DEVICE) * (1.0 / np.sqrt(hidden_dim))
        self.b = torch.zeros(hidden_dim, device=DEVICE)
        self.W_out = torch.randn(output_dim, hidden_dim, device=DEVICE) * 0.01
        self.b_out = torch.zeros(output_dim, device=DEVICE)

        # Random feedback matrix (fixed)
        self.B = torch.randn(hidden_dim, output_dim, device=DEVICE) * (1.0 / np.sqrt(hidden_dim))

    def train_step_copy(self, x, targets, seq_len=10):
        """Train on copy task where output is produced at the last seq_len timesteps."""
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        # Eligibility traces
        e_rec = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim, device=DEVICE)
        e_in = torch.zeros(batch_size, self.hidden_dim, self.input_dim, device=DEVICE)

        # Accumulate weight updates
        dW_rec = torch.zeros_like(self.W_rec)
        dW_in = torch.zeros_like(self.W_in)
        dW_out = torch.zeros_like(self.W_out)
        db_out = torch.zeros_like(self.b_out)
        db = torch.zeros_like(self.b)

        total_loss = 0.0
        correct = 0
        total = 0

        for t in range(total_len):
            # Forward pass
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            h_new = (1 - self.alpha) * h + self.alpha * torch.tanh(a)

            # Update eligibility traces
            phi_prime = self.alpha * (1 - torch.tanh(a) ** 2)  # (batch, hidden)
            # e_rec[b, i, j] = (1-alpha)*e_rec[b,i,j] + phi_prime[b,i] * h[b,j]
            e_rec = (1 - self.alpha) * e_rec + phi_prime.unsqueeze(2) * h.unsqueeze(1)
            e_in = (1 - self.alpha) * e_in + phi_prime.unsqueeze(2) * x[:, t].unsqueeze(1)

            h = h_new

            # Output and learning at the last seq_len steps
            output_step = t - (total_len - seq_len)
            if output_step >= 0 and output_step < seq_len:
                y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out  # (batch, output_dim)

                # Compute error (cross-entropy gradient for softmax)
                probs = torch.softmax(y, dim=-1)
                target_onehot = torch.zeros_like(probs)
                target_onehot.scatter_(1, targets[:, output_step].unsqueeze(1), 1.0)
                delta = target_onehot - probs  # (batch, output_dim)

                # Loss tracking
                log_probs = torch.log_softmax(y, dim=-1)
                loss = -log_probs.gather(1, targets[:, output_step].unsqueeze(1)).mean()
                total_loss += loss.item()

                preds = y.argmax(dim=-1)
                correct += (preds == targets[:, output_step]).sum().item()
                total += batch_size

                # Learning signal via random feedback
                L = (self.B @ delta.unsqueeze(-1)).squeeze(-1)  # (batch, hidden)

                # Weight updates (averaged over batch)
                dW_rec += (L.unsqueeze(2) * e_rec).mean(0)
                dW_in += (L.unsqueeze(2) * e_in).mean(0)
                dW_out += (delta.unsqueeze(2) * h.unsqueeze(1)).mean(0)
                db_out += delta.mean(0)
                db += L.mean(0)

        # Apply updates
        self.W_rec += self.lr * dW_rec
        self.W_in += self.lr * dW_in
        self.W_out += self.lr * dW_out
        self.b_out += self.lr * db_out
        self.b += self.lr * db

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / seq_len
        return avg_loss, accuracy

    def train_step_adding(self, x, targets):
        """Train on adding problem where output is produced at the final timestep."""
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        e_rec = torch.zeros(batch_size, self.hidden_dim, self.hidden_dim, device=DEVICE)
        e_in = torch.zeros(batch_size, self.hidden_dim, self.input_dim, device=DEVICE)

        for t in range(seq_len):
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            h_new = (1 - self.alpha) * h + self.alpha * torch.tanh(a)

            phi_prime = self.alpha * (1 - torch.tanh(a) ** 2)
            e_rec = (1 - self.alpha) * e_rec + phi_prime.unsqueeze(2) * h.unsqueeze(1)
            e_in = (1 - self.alpha) * e_in + phi_prime.unsqueeze(2) * x[:, t].unsqueeze(1)

            h = h_new

        # Output at final step
        y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out  # (batch, 1)
        delta = targets - y  # (batch, 1)
        loss = (delta ** 2).mean().item()

        # Learning signal
        L = (self.B @ delta.unsqueeze(-1)).squeeze(-1)  # (batch, hidden)

        # Weight updates
        self.W_rec += self.lr * (L.unsqueeze(2) * e_rec).mean(0)
        self.W_in += self.lr * (L.unsqueeze(2) * e_in).mean(0)
        self.W_out += self.lr * (delta.unsqueeze(2) * h.unsqueeze(1)).mean(0)
        self.b_out += self.lr * delta.mean(0)
        self.b += self.lr * L.mean(0)

        return loss

    def eval_copy(self, x, targets, seq_len=10):
        """Evaluate on copy task."""
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        correct = 0
        total = 0
        total_loss = 0.0

        for t in range(total_len):
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(a)

            output_step = t - (total_len - seq_len)
            if output_step >= 0 and output_step < seq_len:
                y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out
                log_probs = torch.log_softmax(y, dim=-1)
                loss = -log_probs.gather(1, targets[:, output_step].unsqueeze(1)).mean()
                total_loss += loss.item()
                preds = y.argmax(dim=-1)
                correct += (preds == targets[:, output_step]).sum().item()
                total += batch_size

        return total_loss / seq_len, correct / total

    def eval_adding(self, x, targets):
        """Evaluate on adding problem."""
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        for t in range(seq_len):
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            h = (1 - self.alpha) * h + self.alpha * torch.tanh(a)

        y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out
        loss = ((targets - y) ** 2).mean().item()
        return loss


# ============================================================
# Training Loops
# ============================================================

def train_bptt_copy(n_iters=5000, hidden_dim=128, seq_len=10, lr=1e-3, batch_size=32):
    input_dim = 10  # 8 symbols + blank + go
    output_dim = 8

    model = VanillaRNN_BPTT(input_dim, hidden_dim, output_dim, alpha=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for i in range(n_iters):
        x, targets = generate_copy_task(batch_size, seq_len=seq_len)
        optimizer.zero_grad()

        outputs = model(x, return_all=True)
        # Only take the last seq_len outputs
        outputs = outputs[:, -seq_len:, :]  # (batch, seq_len, output_dim)

        loss = criterion(outputs.reshape(-1, output_dim), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = outputs.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()

        losses.append(loss.item())
        accuracies.append(acc)

        if (i + 1) % 500 == 0:
            print(f"  BPTT Copy iter {i+1}: loss={loss.item():.4f}, acc={acc:.4f}")

    # Final evaluation
    x_test, t_test = generate_copy_task(1000, seq_len=seq_len)
    with torch.no_grad():
        out_test = model(x_test, return_all=True)[:, -seq_len:, :]
        test_acc = (out_test.argmax(dim=-1) == t_test).float().mean().item()
        test_loss = criterion(out_test.reshape(-1, output_dim), t_test.reshape(-1)).item()

    return losses, accuracies, test_loss, test_acc


def train_rflo_copy(n_iters=5000, hidden_dim=128, seq_len=10, lr=0.01, batch_size=32):
    input_dim = 10
    output_dim = 8

    model = VanillaRNN_RFLO(input_dim, hidden_dim, output_dim, alpha=0.2, lr=lr)

    losses = []
    accuracies = []

    for i in range(n_iters):
        x, targets = generate_copy_task(batch_size, seq_len=seq_len)
        loss, acc = model.train_step_copy(x, targets, seq_len=seq_len)
        losses.append(loss)
        accuracies.append(acc)

        if (i + 1) % 500 == 0:
            print(f"  RFLO Copy iter {i+1}: loss={loss:.4f}, acc={acc:.4f}")

    # Final evaluation
    x_test, t_test = generate_copy_task(1000, seq_len=seq_len)
    test_loss, test_acc = model.eval_copy(x_test, t_test, seq_len=seq_len)

    return losses, accuracies, test_loss, test_acc


def train_bptt_adding(n_iters=5000, hidden_dim=128, seq_len=30, lr=1e-3, batch_size=32):
    input_dim = 2
    output_dim = 1

    model = VanillaRNN_BPTT(input_dim, hidden_dim, output_dim, alpha=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for i in range(n_iters):
        x, targets = generate_adding_problem(batch_size, seq_len=seq_len)
        optimizer.zero_grad()

        y = model(x, return_all=False)
        loss = nn.MSELoss()(y, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 500 == 0:
            print(f"  BPTT Adding iter {i+1}: MSE={loss.item():.6f}")

    # Final evaluation
    x_test, t_test = generate_adding_problem(1000, seq_len=seq_len)
    with torch.no_grad():
        y_test = model(x_test, return_all=False)
        test_loss = nn.MSELoss()(y_test, t_test).item()

    return losses, test_loss


def train_rflo_adding(n_iters=5000, hidden_dim=128, seq_len=30, lr=0.01, batch_size=32):
    input_dim = 2
    output_dim = 1

    model = VanillaRNN_RFLO(input_dim, hidden_dim, output_dim, alpha=0.2, lr=lr)

    losses = []

    for i in range(n_iters):
        x, targets = generate_adding_problem(batch_size, seq_len=seq_len)
        loss = model.train_step_adding(x, targets)
        losses.append(loss)

        if (i + 1) % 500 == 0:
            print(f"  RFLO Adding iter {i+1}: MSE={loss:.6f}")

    # Final evaluation
    x_test, t_test = generate_adding_problem(1000, seq_len=seq_len)
    test_loss = model.eval_adding(x_test, t_test)

    return losses, test_loss


# ============================================================
# Main
# ============================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("=" * 60)

    results = {}

    # --- Copy Task ---
    print("\n[1/4] Training BPTT on Copy Task (seq_len=10)...")
    t0 = time.time()
    bptt_copy_losses, bptt_copy_accs, bptt_copy_test_loss, bptt_copy_test_acc = train_bptt_copy()
    bptt_copy_time = time.time() - t0
    print(f"  Final test: loss={bptt_copy_test_loss:.4f}, acc={bptt_copy_test_acc:.4f}, time={bptt_copy_time:.1f}s")

    print("\n[2/4] Training RFLO on Copy Task (seq_len=10)...")
    t0 = time.time()
    rflo_copy_losses, rflo_copy_accs, rflo_copy_test_loss, rflo_copy_test_acc = train_rflo_copy()
    rflo_copy_time = time.time() - t0
    print(f"  Final test: loss={rflo_copy_test_loss:.4f}, acc={rflo_copy_test_acc:.4f}, time={rflo_copy_time:.1f}s")

    # --- Adding Problem ---
    print("\n[3/4] Training BPTT on Adding Problem (seq_len=30)...")
    t0 = time.time()
    bptt_add_losses, bptt_add_test_loss = train_bptt_adding()
    bptt_add_time = time.time() - t0
    print(f"  Final test MSE: {bptt_add_test_loss:.6f}, time={bptt_add_time:.1f}s")

    print("\n[4/4] Training RFLO on Adding Problem (seq_len=30)...")
    t0 = time.time()
    rflo_add_losses, rflo_add_test_loss = train_rflo_adding()
    rflo_add_time = time.time() - t0
    print(f"  Final test MSE: {rflo_add_test_loss:.6f}, time={rflo_add_time:.1f}s")

    # --- Save Results ---
    results = {
        "copy_task": {
            "bptt": {"test_loss": bptt_copy_test_loss, "test_acc": bptt_copy_test_acc, "time_s": bptt_copy_time},
            "rflo": {"test_loss": rflo_copy_test_loss, "test_acc": rflo_copy_test_acc, "time_s": rflo_copy_time},
        },
        "adding_problem": {
            "bptt": {"test_mse": bptt_add_test_loss, "time_s": bptt_add_time},
            "rflo": {"test_mse": rflo_add_test_loss, "time_s": rflo_add_time},
            "random_baseline_mse": 0.167,
        },
        "config": {
            "hidden_dim": 128, "alpha": 0.2, "n_iters": 5000, "batch_size": 32,
            "lr_bptt": 1e-3, "lr_rflo": 1e-2,
            "copy_seq_len": 10, "adding_seq_len": 30,
        }
    }

    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Copy task loss
    window = 50
    ax = axes[0, 0]
    bptt_smooth = np.convolve(bptt_copy_losses, np.ones(window)/window, mode='valid')
    rflo_smooth = np.convolve(rflo_copy_losses, np.ones(window)/window, mode='valid')
    ax.plot(bptt_smooth, label='BPTT', alpha=0.8)
    ax.plot(rflo_smooth, label='RFLO', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Copy Task - Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Copy task accuracy
    ax = axes[0, 1]
    bptt_acc_smooth = np.convolve(bptt_copy_accs, np.ones(window)/window, mode='valid')
    rflo_acc_smooth = np.convolve(rflo_copy_accs, np.ones(window)/window, mode='valid')
    ax.plot(bptt_acc_smooth, label='BPTT', alpha=0.8)
    ax.plot(rflo_acc_smooth, label='RFLO', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Copy Task - Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Adding problem loss
    ax = axes[1, 0]
    bptt_add_smooth = np.convolve(bptt_add_losses, np.ones(window)/window, mode='valid')
    rflo_add_smooth = np.convolve(rflo_add_losses, np.ones(window)/window, mode='valid')
    ax.plot(bptt_add_smooth, label='BPTT', alpha=0.8)
    ax.plot(rflo_add_smooth, label='RFLO', alpha=0.8)
    ax.axhline(y=0.167, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')
    ax.set_title('Adding Problem - MSE (seq_len=30)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Method', 'Copy Acc', 'Copy Loss', 'Adding MSE'],
        ['BPTT', f'{bptt_copy_test_acc:.3f}', f'{bptt_copy_test_loss:.3f}', f'{bptt_add_test_loss:.5f}'],
        ['RFLO', f'{rflo_copy_test_acc:.3f}', f'{rflo_copy_test_loss:.3f}', f'{rflo_add_test_loss:.5f}'],
        ['Random', '0.125', '2.079', '0.167'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    ax.set_title('Final Test Results', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to {RESULTS_DIR}/learning_curves.png")
    print(f"Results saved to {RESULTS_DIR}/results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Copy Task (seq_len=10):")
    print(f"  BPTT:  acc={bptt_copy_test_acc:.3f}, loss={bptt_copy_test_loss:.4f}, time={bptt_copy_time:.1f}s")
    print(f"  RFLO:  acc={rflo_copy_test_acc:.3f}, loss={rflo_copy_test_loss:.4f}, time={rflo_copy_time:.1f}s")
    print(f"Adding Problem (seq_len=30):")
    print(f"  BPTT:  MSE={bptt_add_test_loss:.6f}, time={bptt_add_time:.1f}s")
    print(f"  RFLO:  MSE={rflo_add_test_loss:.6f}, time={rflo_add_time:.1f}s")
    print(f"  Random baseline: MSE=0.167")


if __name__ == '__main__':
    main()
