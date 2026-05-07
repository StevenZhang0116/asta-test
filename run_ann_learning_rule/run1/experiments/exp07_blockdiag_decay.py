"""
Experiment 07: Block-Diagonal RTRL + Exponential Decay (Combined)
Test whether spatial (block) and temporal (decay) approximations synergize.
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp07')
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


class BlockDiagDecayRTRL:
    """Block-diagonal RTRL with exponential decay on the Jacobian."""

    def __init__(self, block_size=16, trace_decay=0.9, lr=LR):
        self.block_size = block_size
        self.n_blocks = HIDDEN_DIM // block_size
        self.trace_decay = trace_decay
        self.lr = lr

        self.W_in = torch.randn(HIDDEN_DIM, INPUT_DIM, device=DEVICE) * 0.01
        self.W_rec = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))
        self.b = torch.zeros(HIDDEN_DIM, device=DEVICE)
        self.W_out = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=DEVICE) * 0.01
        self.b_out = torch.zeros(OUTPUT_DIM, device=DEVICE)
        self.B = torch.randn(HIDDEN_DIM, OUTPUT_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))

    def train_step(self, x, targets):
        batch_size, total_len, _ = x.shape
        K = self.block_size
        h = torch.zeros(batch_size, HIDDEN_DIM, device=DEVICE)

        J_rec_blocks = [torch.zeros(batch_size, K, K, K, device=DEVICE) for _ in range(self.n_blocks)]
        J_in_blocks = [torch.zeros(batch_size, K, K, INPUT_DIM, device=DEVICE) for _ in range(self.n_blocks)]

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
            h_new = torch.tanh(a)
            phi_prime = 1 - h_new ** 2

            for b_idx in range(self.n_blocks):
                s = b_idx * K
                e = s + K

                W_block = self.W_rec[s:e, s:e]
                phi_block = phi_prime[:, s:e]
                h_block = h[:, s:e]

                # Propagate with decay
                propagated = torch.einsum('ij,bjkl->bikl', W_block, J_rec_blocks[b_idx])
                direct = torch.zeros_like(J_rec_blocks[b_idx])
                for i in range(K):
                    direct[:, i, i, :] = h_block

                J_rec_blocks[b_idx] = self.trace_decay * phi_block.unsqueeze(2).unsqueeze(3) * (propagated + direct)

                # W_in
                propagated_in = torch.einsum('ij,bjkl->bikl', W_block, J_in_blocks[b_idx])
                direct_in = torch.zeros_like(J_in_blocks[b_idx])
                for i in range(K):
                    direct_in[:, i, i, :] = x[:, t]

                J_in_blocks[b_idx] = self.trace_decay * phi_block.unsqueeze(2).unsqueeze(3) * (propagated_in + direct_in)

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

                for b_idx in range(self.n_blocks):
                    s = b_idx * K
                    e = s + K
                    L_block = L[:, s:e]
                    dW_rec[s:e, s:e] += torch.einsum('bi,bijl->jl', L_block, J_rec_blocks[b_idx]) / batch_size
                    dW_in[s:e, :] += torch.einsum('bi,bijl->jl', L_block, J_in_blocks[b_idx]) / batch_size

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
    print("=" * 60)

    all_results = {}

    configs = [
        # Block size, decay — sweep the 2D grid
        (8, 0.8), (8, 0.9), (8, 0.95),
        (16, 0.8), (16, 0.9), (16, 0.95),
        (32, 0.8), (32, 0.9), (32, 0.95),
    ]

    print("\n--- Block-Diagonal + Decay Grid ---")
    for K, decay in configs:
        name = f"K={K}_decay={decay}"
        model = BlockDiagDecayRTRL(block_size=K, trace_decay=decay, lr=LR)
        accs, final, elapsed = run_model(name, model)
        memory = (HIDDEN_DIM // K) * K * K * K  # = n * K^2
        all_results[name] = {"accs": accs, "final": final, "time": elapsed,
                             "K": K, "decay": decay, "memory": memory}

    # Save results
    results_json = {k: {"final_acc": v["final"], "time": v["time"], "K": v["K"],
                        "decay": v["decay"], "memory": v["memory"]}
                    for k, v in all_results.items()}
    results_json["references"] = {
        "rank1_RFLO": 0.28,
        "full_RTRL_decay0.9": 0.998,
        "full_RTRL_nodecay": 1.0,
        "blockdiag_K8_nodecay": 0.305,
        "blockdiag_K16_nodecay": 0.411,
        "blockdiag_K32_nodecay": 0.788,
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Plot: heatmap of accuracy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Heatmap
    ax = axes[0]
    K_vals = [8, 16, 32]
    decay_vals = [0.8, 0.9, 0.95]
    heatmap_data = np.zeros((len(K_vals), len(decay_vals)))
    for i, K in enumerate(K_vals):
        for j, decay in enumerate(decay_vals):
            name = f"K={K}_decay={decay}"
            heatmap_data[i, j] = all_results[name]["final"]

    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(decay_vals)))
    ax.set_xticklabels([f'{d}' for d in decay_vals])
    ax.set_yticks(range(len(K_vals)))
    ax.set_yticklabels([f'K={K}' for K in K_vals])
    ax.set_xlabel('Trace Decay')
    ax.set_ylabel('Block Size')
    ax.set_title('Accuracy: Block-Diagonal + Decay')
    for i in range(len(K_vals)):
        for j in range(len(decay_vals)):
            ax.text(j, i, f'{heatmap_data[i,j]:.3f}', ha='center', va='center', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax)

    # Bar chart comparison with references
    ax = axes[1]
    comparison = {
        'Rank-1 (RFLO)': 0.28,
        'BlockDiag K=16\n(no decay)': 0.411,
        'BlockDiag K=32\n(no decay)': 0.788,
    }
    # Add best from each K
    for K in K_vals:
        best_decay = max(decay_vals, key=lambda d: all_results[f"K={K}_decay={d}"]["final"])
        best_acc = all_results[f"K={K}_decay={best_decay}"]["final"]
        comparison[f'K={K}\ndecay={best_decay}'] = best_acc
    comparison['Full RTRL\ndecay=0.9'] = 0.998

    names = list(comparison.keys())
    values = list(comparison.values())
    colors = ['gray', 'lightcoral', 'coral'] + ['steelblue'] * 3 + ['green']
    ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison: Block+Decay vs References')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.28, color='r', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'blockdiag_decay.png'), dpi=150, bbox_inches='tight')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Config':<25} {'Accuracy':>8} {'Memory':>10} {'vs no-decay':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*12}")
    ref_nodecay = {8: 0.305, 16: 0.411, 32: 0.788}
    for name, data in sorted(all_results.items(), key=lambda x: -x[1]["final"]):
        improvement = data["final"] - ref_nodecay[data["K"]]
        print(f"  {name:<25} {data['final']:>8.3f} {data['memory']:>10,} {improvement:>+11.3f}")
    print(f"\n  References: Rank-1=0.280, Full_RTRL_decay0.9=0.998")
    print(f"\nPlots saved to: {RESULTS_DIR}/blockdiag_decay.png")


if __name__ == '__main__':
    main()
