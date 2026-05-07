"""
Experiment 02: Hyperparameter Sweep for BPTT and RFLO on Copy Task
Focus on finding configurations where both methods perform well.
Key variables: alpha (leak rate), learning rate, training iterations.
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp02')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 10
HIDDEN_DIM = 128
N_SYMBOLS = 8
INPUT_DIM = N_SYMBOLS + 2  # symbols + blank + go
OUTPUT_DIM = N_SYMBOLS
BATCH_SIZE = 64
N_ITERS = 10000
EVAL_EVERY = 200


def generate_copy_task(batch_size, seq_len=SEQ_LEN):
    total_len = 2 * seq_len + 1
    symbols = torch.randint(0, N_SYMBOLS, (batch_size, seq_len))
    inputs = torch.zeros(batch_size, total_len, INPUT_DIM)
    for b in range(batch_size):
        for t in range(seq_len):
            inputs[b, t, symbols[b, t]] = 1.0
        inputs[b, seq_len:2*seq_len, N_SYMBOLS] = 1.0
        inputs[b, 2*seq_len, N_SYMBOLS + 1] = 1.0
    return inputs.to(DEVICE), symbols.to(DEVICE)


# ============================================================
# BPTT Model
# ============================================================

class VanillaRNN_BPTT(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.hidden_dim = HIDDEN_DIM
        self.alpha = alpha
        self.W_in = nn.Parameter(torch.randn(HIDDEN_DIM, INPUT_DIM) * 0.01)
        self.W_rec = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM) * (1.0 / np.sqrt(HIDDEN_DIM)))
        self.b = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.W_out = nn.Parameter(torch.randn(OUTPUT_DIM, HIDDEN_DIM) * 0.01)
        self.b_out = nn.Parameter(torch.zeros(OUTPUT_DIM))

    def forward(self, x):
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, HIDDEN_DIM, device=x.device)

        outputs = []
        for t in range(total_len):
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            if self.alpha < 1.0:
                h = (1 - self.alpha) * h + self.alpha * torch.tanh(a)
            else:
                h = torch.tanh(a)
            outputs.append(h)

        hiddens = torch.stack(outputs[-SEQ_LEN:], dim=1)  # last SEQ_LEN steps
        out = (self.W_out @ hiddens.unsqueeze(-1)).squeeze(-1) + self.b_out
        return out


# ============================================================
# RFLO Model
# ============================================================

class VanillaRNN_RFLO:
    def __init__(self, alpha=0.2, lr=0.01):
        self.alpha = alpha
        self.lr = lr
        self.W_in = torch.randn(HIDDEN_DIM, INPUT_DIM, device=DEVICE) * 0.01
        self.W_rec = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))
        self.b = torch.zeros(HIDDEN_DIM, device=DEVICE)
        self.W_out = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=DEVICE) * 0.01
        self.b_out = torch.zeros(OUTPUT_DIM, device=DEVICE)
        self.B = torch.randn(HIDDEN_DIM, OUTPUT_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))

    def train_step(self, x, targets):
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, HIDDEN_DIM, device=DEVICE)
        e_rec = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, device=DEVICE)
        e_in = torch.zeros(batch_size, HIDDEN_DIM, INPUT_DIM, device=DEVICE)

        dW_rec = torch.zeros_like(self.W_rec)
        dW_in = torch.zeros_like(self.W_in)
        dW_out = torch.zeros_like(self.W_out)
        db_out = torch.zeros_like(self.b_out)
        db = torch.zeros_like(self.b)

        total_loss = 0.0
        correct = 0
        total = 0

        for t in range(total_len):
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b

            if self.alpha < 1.0:
                h_new = (1 - self.alpha) * h + self.alpha * torch.tanh(a)
                phi_prime = self.alpha * (1 - torch.tanh(a) ** 2)
                trace_decay = (1 - self.alpha)
            else:
                h_new = torch.tanh(a)
                phi_prime = 1 - torch.tanh(a) ** 2
                trace_decay = 0.0  # no persistence in discrete case

            # For discrete case (alpha=1), use a modified trace that still retains info
            # through the recurrent connection: e_t = phi'(a_t) * (W_rec @ e_{t-1} + input)
            # But this is non-local! For bio-plausibility, we use the simplified version:
            e_rec = trace_decay * e_rec + phi_prime.unsqueeze(2) * h.unsqueeze(1)
            e_in = trace_decay * e_in + phi_prime.unsqueeze(2) * x[:, t].unsqueeze(1)

            h = h_new

            output_step = t - (total_len - SEQ_LEN)
            if output_step >= 0 and output_step < SEQ_LEN:
                y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out
                probs = torch.softmax(y, dim=-1)
                target_onehot = torch.zeros_like(probs)
                target_onehot.scatter_(1, targets[:, output_step].unsqueeze(1), 1.0)
                delta = target_onehot - probs

                log_probs = torch.log_softmax(y, dim=-1)
                loss = -log_probs.gather(1, targets[:, output_step].unsqueeze(1)).mean()
                total_loss += loss.item()

                preds = y.argmax(dim=-1)
                correct += (preds == targets[:, output_step]).sum().item()
                total += batch_size

                L = (self.B @ delta.unsqueeze(-1)).squeeze(-1)
                dW_rec += (L.unsqueeze(2) * e_rec).mean(0)
                dW_in += (L.unsqueeze(2) * e_in).mean(0)
                dW_out += (delta.unsqueeze(2) * h.unsqueeze(1)).mean(0)
                db_out += delta.mean(0)
                db += L.mean(0)

        self.W_rec += self.lr * dW_rec
        self.W_in += self.lr * dW_in
        self.W_out += self.lr * dW_out
        self.b_out += self.lr * db_out
        self.b += self.lr * db

        return total_loss / SEQ_LEN, correct / total if total > 0 else 0.0

    def evaluate(self, x, targets):
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, HIDDEN_DIM, device=DEVICE)
        correct = 0
        total = 0

        for t in range(total_len):
            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            if self.alpha < 1.0:
                h = (1 - self.alpha) * h + self.alpha * torch.tanh(a)
            else:
                h = torch.tanh(a)

            output_step = t - (total_len - SEQ_LEN)
            if output_step >= 0 and output_step < SEQ_LEN:
                y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out
                preds = y.argmax(dim=-1)
                correct += (preds == targets[:, output_step]).sum().item()
                total += batch_size

        return correct / total


# ============================================================
# Training Functions
# ============================================================

def train_bptt(alpha, lr, n_iters=N_ITERS):
    model = VanillaRNN_BPTT(alpha=alpha).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    eval_accs = []
    for i in range(n_iters):
        x, targets = generate_copy_task(BATCH_SIZE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.reshape(-1, OUTPUT_DIM), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (i + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                x_test, t_test = generate_copy_task(500)
                out_test = model(x_test)
                acc = (out_test.argmax(dim=-1) == t_test).float().mean().item()
                eval_accs.append(acc)

    # Final eval
    with torch.no_grad():
        x_test, t_test = generate_copy_task(2000)
        out_test = model(x_test)
        final_acc = (out_test.argmax(dim=-1) == t_test).float().mean().item()

    return eval_accs, final_acc


def train_rflo(alpha, lr, n_iters=N_ITERS):
    model = VanillaRNN_RFLO(alpha=alpha, lr=lr)

    eval_accs = []
    for i in range(n_iters):
        x, targets = generate_copy_task(BATCH_SIZE)
        model.train_step(x, targets)

        if (i + 1) % EVAL_EVERY == 0:
            x_test, t_test = generate_copy_task(500)
            acc = model.evaluate(x_test, t_test)
            eval_accs.append(acc)

    x_test, t_test = generate_copy_task(2000)
    final_acc = model.evaluate(x_test, t_test)

    return eval_accs, final_acc


# ============================================================
# Main
# ============================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Task: Copy task, seq_len={SEQ_LEN}, hidden={HIDDEN_DIM}")
    print(f"Training: {N_ITERS} iterations, batch_size={BATCH_SIZE}")
    print("=" * 60)

    # BPTT sweep
    bptt_configs = [
        {"alpha": 1.0, "lr": 1e-3},
        {"alpha": 1.0, "lr": 3e-3},
        {"alpha": 0.5, "lr": 1e-3},
        {"alpha": 0.5, "lr": 3e-3},
        {"alpha": 0.2, "lr": 1e-3},
        {"alpha": 0.2, "lr": 3e-3},
    ]

    # RFLO sweep
    rflo_configs = [
        {"alpha": 0.1, "lr": 0.005},
        {"alpha": 0.1, "lr": 0.01},
        {"alpha": 0.1, "lr": 0.02},
        {"alpha": 0.2, "lr": 0.005},
        {"alpha": 0.2, "lr": 0.01},
        {"alpha": 0.2, "lr": 0.02},
        {"alpha": 0.3, "lr": 0.01},
        {"alpha": 0.3, "lr": 0.02},
        {"alpha": 0.5, "lr": 0.01},
        {"alpha": 0.5, "lr": 0.02},
    ]

    bptt_results = {}
    rflo_results = {}

    print("\n--- BPTT Sweep ---")
    for cfg in bptt_configs:
        label = f"alpha={cfg['alpha']}, lr={cfg['lr']}"
        print(f"  Running BPTT: {label}...", end=" ", flush=True)
        t0 = time.time()
        accs, final = train_bptt(**cfg)
        elapsed = time.time() - t0
        bptt_results[label] = {"accs": accs, "final": final, "config": cfg, "time": elapsed}
        print(f"final_acc={final:.3f} ({elapsed:.1f}s)")

    print("\n--- RFLO Sweep ---")
    for cfg in rflo_configs:
        label = f"alpha={cfg['alpha']}, lr={cfg['lr']}"
        print(f"  Running RFLO: {label}...", end=" ", flush=True)
        t0 = time.time()
        accs, final = train_rflo(**cfg)
        elapsed = time.time() - t0
        rflo_results[label] = {"accs": accs, "final": final, "config": cfg, "time": elapsed}
        print(f"final_acc={final:.3f} ({elapsed:.1f}s)")

    # Save results
    results_json = {
        "bptt": {k: {"final_acc": v["final"], "config": v["config"], "time": v["time"]} for k, v in bptt_results.items()},
        "rflo": {k: {"final_acc": v["final"], "config": v["config"], "time": v["time"]} for k, v in rflo_results.items()},
        "best_bptt": max(bptt_results.items(), key=lambda x: x[1]["final"])[0],
        "best_rflo": max(rflo_results.items(), key=lambda x: x[1]["final"])[0],
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title('BPTT - Copy Task Accuracy vs Iteration')
    for label, data in bptt_results.items():
        iters = [(i+1)*EVAL_EVERY for i in range(len(data["accs"]))]
        ax.plot(iters, data["accs"], label=f'{label} (final={data["final"]:.3f})', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.125, color='r', linestyle='--', alpha=0.3, label='random')

    ax = axes[1]
    ax.set_title('RFLO - Copy Task Accuracy vs Iteration')
    for label, data in rflo_results.items():
        iters = [(i+1)*EVAL_EVERY for i in range(len(data["accs"]))]
        ax.plot(iters, data["accs"], label=f'{label} (final={data["final"]:.3f})', alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.125, color='r', linestyle='--', alpha=0.3, label='random')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hyperparam_sweep.png'), dpi=150, bbox_inches='tight')

    # Print summary
    print("\n" + "=" * 60)
    print("BEST RESULTS")
    print("=" * 60)
    best_bptt = max(bptt_results.items(), key=lambda x: x[1]["final"])
    best_rflo = max(rflo_results.items(), key=lambda x: x[1]["final"])
    print(f"Best BPTT: {best_bptt[0]} → acc={best_bptt[1]['final']:.3f}")
    print(f"Best RFLO: {best_rflo[0]} → acc={best_rflo[1]['final']:.3f}")
    print(f"\nPlots saved to: {RESULTS_DIR}/hyperparam_sweep.png")
    print(f"Results saved to: {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()
