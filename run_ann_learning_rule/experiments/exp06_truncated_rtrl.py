"""
Experiment 06: Truncated Forward-RTRL
Test temporal windowing: propagate full Jacobian with decay or periodic reset.
Two approaches:
  1. Exponential decay on full Jacobian (soft window)
  2. Periodic hard reset every K steps
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp06')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 10
HIDDEN_DIM = 64
N_SYMBOLS = 8
INPUT_DIM = N_SYMBOLS + 2
OUTPUT_DIM = N_SYMBOLS
BATCH_SIZE = 32
N_ITERS = 10000
EVAL_EVERY = 500
LR = 0.005


def generate_copy_task(batch_size):
    total_len = 2 * SEQ_LEN + 1
    symbols = torch.randint(0, N_SYMBOLS, (batch_size, SEQ_LEN))
    inputs = torch.zeros(batch_size, total_len, INPUT_DIM)
    for b in range(batch_size):
        for t in range(SEQ_LEN):
            inputs[b, t, symbols[b, t]] = 1.0
        inputs[b, SEQ_LEN:2*SEQ_LEN, N_SYMBOLS] = 1.0
        inputs[b, 2*SEQ_LEN, N_SYMBOLS + 1] = 1.0
    return inputs.to(DEVICE), symbols.to(DEVICE)


class TruncatedRTRL:
    """
    Full RTRL with temporal truncation via either:
    - Exponential decay (trace_decay < 1.0): soft window
    - Periodic reset (reset_period > 0): hard window
    """

    def __init__(self, trace_decay=1.0, reset_period=0, lr=LR):
        self.trace_decay = trace_decay
        self.reset_period = reset_period  # 0 means no reset (use decay only)
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

        J_rec = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, device=DEVICE)
        J_in = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM, device=DEVICE)

        dW_rec = torch.zeros_like(self.W_rec)
        dW_in = torch.zeros_like(self.W_in)
        dW_out = torch.zeros_like(self.W_out)
        db_out = torch.zeros_like(self.b_out)
        db = torch.zeros_like(self.b)

        total_loss = 0.0
        correct = 0
        total = 0

        for t in range(total_len):
            # Hard reset if using periodic reset
            if self.reset_period > 0 and t % self.reset_period == 0:
                J_rec = torch.zeros_like(J_rec)
                J_in = torch.zeros_like(J_in)

            a = (self.W_rec @ h.unsqueeze(-1)).squeeze(-1) + (self.W_in @ x[:, t].unsqueeze(-1)).squeeze(-1) + self.b
            h_new = torch.tanh(a)
            phi_prime = 1 - h_new ** 2

            # Propagate Jacobian (with optional decay)
            J_rec_flat = J_rec.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM * HIDDEN_DIM)
            propagated_rec = torch.bmm(
                self.W_rec.unsqueeze(0).expand(batch_size, -1, -1),
                J_rec_flat
            ).reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM)

            direct_rec = torch.zeros_like(J_rec)
            for i in range(HIDDEN_DIM):
                direct_rec[:, i, i, :] = h

            J_rec = self.trace_decay * phi_prime.unsqueeze(2).unsqueeze(3) * (propagated_rec + direct_rec)

            # W_in Jacobian
            J_in_flat = J_in.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM * INPUT_DIM)
            propagated_in = torch.bmm(
                self.W_rec.unsqueeze(0).expand(batch_size, -1, -1),
                J_in_flat
            ).reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM)

            direct_in = torch.zeros_like(J_in)
            for i in range(HIDDEN_DIM):
                direct_in[:, i, i, :] = x[:, t]

            J_in = self.trace_decay * phi_prime.unsqueeze(2).unsqueeze(3) * (propagated_in + direct_in)

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

                dW_rec += torch.einsum('bi,bijl->jl', L, J_rec) / batch_size
                dW_in += torch.einsum('bi,bijl->jl', L, J_in) / batch_size
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
            h = torch.tanh(a)
            output_step = t - (total_len - SEQ_LEN)
            if output_step >= 0 and output_step < SEQ_LEN:
                y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out
                preds = y.argmax(dim=-1)
                correct += (preds == targets[:, output_step]).sum().item()
                total += batch_size
        return correct / total


def run_model(name, model):
    print(f"  {name}...", end=" ", flush=True)
    eval_accs = []
    t0 = time.time()
    for i in range(N_ITERS):
        x, targets = generate_copy_task(BATCH_SIZE)
        model.train_step(x, targets)
        if (i + 1) % EVAL_EVERY == 0:
            x_test, t_test = generate_copy_task(500)
            acc = model.evaluate(x_test, t_test)
            eval_accs.append(acc)
    elapsed = time.time() - t0
    x_test, t_test = generate_copy_task(2000)
    final_acc = model.evaluate(x_test, t_test)
    print(f"acc={final_acc:.3f} ({elapsed:.1f}s)")
    return eval_accs, final_acc, elapsed


def main():
    print(f"Device: {DEVICE}")
    print(f"Copy task seq_len={SEQ_LEN}, hidden={HIDDEN_DIM}, {N_ITERS} iters, lr={LR}")
    print(f"Total sequence length: {2*SEQ_LEN+1} steps")
    print("=" * 60)

    all_results = {}

    # Part 1: Exponential decay (soft window)
    print("\n--- Part 1: Exponential Decay on Full Jacobian ---")
    for decay in [0.8, 0.9, 0.95, 0.99, 1.0]:
        eff_window = 1.0 / (1.0 - decay) if decay < 1.0 else float('inf')
        name = f"Decay={decay} (eff_win≈{eff_window:.0f})"
        model = TruncatedRTRL(trace_decay=decay, reset_period=0, lr=LR)
        accs, final, elapsed = run_model(name, model)
        all_results[name] = {"accs": accs, "final": final, "time": elapsed, "decay": decay, "type": "decay"}

    # Part 2: Periodic hard reset
    print("\n--- Part 2: Periodic Hard Reset ---")
    for K in [5, 7, 11, 15, 21]:
        name = f"Reset_K={K}"
        model = TruncatedRTRL(trace_decay=1.0, reset_period=K, lr=LR)
        accs, final, elapsed = run_model(name, model)
        all_results[name] = {"accs": accs, "final": final, "time": elapsed, "period": K, "type": "reset"}

    # Save results
    results_json = {k: {"final_acc": v["final"], "time": v["time"]} for k, v in all_results.items()}
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for name, data in all_results.items():
        if data["type"] == "decay":
            iters = [(i+1)*EVAL_EVERY for i in range(len(data["accs"]))]
            ax.plot(iters, data["accs"], label=f'{name} ({data["final"]:.3f})', linewidth=2)
    ax.axhline(y=0.28, color='r', linestyle='--', alpha=0.5, label='Rank-1 (RFLO)')
    ax.set_title('Exponential Decay (Soft Window)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax = axes[1]
    for name, data in all_results.items():
        if data["type"] == "reset":
            iters = [(i+1)*EVAL_EVERY for i in range(len(data["accs"]))]
            ax.plot(iters, data["accs"], label=f'{name} ({data["final"]:.3f})', linewidth=2)
    ax.axhline(y=0.28, color='r', linestyle='--', alpha=0.5, label='Rank-1 (RFLO)')
    ax.set_title('Periodic Hard Reset')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Summary bar chart
    ax = axes[2]
    names = []
    accs_vals = []
    colors = []
    for name, data in sorted(all_results.items(), key=lambda x: -x[1]["final"]):
        short_name = name.split('(')[0].strip() if '(' in name else name
        names.append(short_name)
        accs_vals.append(data["final"])
        colors.append('steelblue' if data["type"] == "decay" else 'coral')
    ax.barh(range(len(names)), accs_vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title('All Conditions')
    ax.axvline(x=0.28, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='g', linestyle='--', alpha=0.5)
    ax.set_xlim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'truncated_rtrl.png'), dpi=150, bbox_inches='tight')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, data in sorted(all_results.items(), key=lambda x: -x[1]["final"]):
        print(f"  {name:<40} acc={data['final']:.3f} ({data['time']:.0f}s)")
    print(f"\nPlots saved to: {RESULTS_DIR}/truncated_rtrl.png")


if __name__ == '__main__':
    main()
