"""
Experiment 04: Debug C4 + Test Optimizer Effect
Part 1: Sweep lr for C4 (exact feedback + full RTRL) to fix anomalous 13.8% result
Part 2: Test all conditions with Adam-style local updates
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp04')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 10
HIDDEN_DIM = 64
N_SYMBOLS = 8
INPUT_DIM = N_SYMBOLS + 2
OUTPUT_DIM = N_SYMBOLS
BATCH_SIZE = 32
N_ITERS = 10000
EVAL_EVERY = 500
TRACE_DECAY = 0.9


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


class LocalRNN:
    """RNN with local learning rule, supporting different feedback/trace/optimizer combos."""

    def __init__(self, use_exact_feedback=False, use_full_rtrl=False,
                 lr=0.01, trace_decay=TRACE_DECAY, optimizer='sgd', momentum=0.9,
                 adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
        self.use_exact_feedback = use_exact_feedback
        self.use_full_rtrl = use_full_rtrl
        self.lr = lr
        self.trace_decay = trace_decay
        self.optimizer_type = optimizer
        self.momentum = momentum
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        self.W_in = torch.randn(HIDDEN_DIM, INPUT_DIM, device=DEVICE) * 0.01
        self.W_rec = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))
        self.b = torch.zeros(HIDDEN_DIM, device=DEVICE)
        self.W_out = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=DEVICE) * 0.01
        self.b_out = torch.zeros(OUTPUT_DIM, device=DEVICE)

        if not use_exact_feedback:
            self.B = torch.randn(HIDDEN_DIM, OUTPUT_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))

        # Optimizer state
        self.params = ['W_in', 'W_rec', 'b', 'W_out', 'b_out']
        self.t_step = 0
        if optimizer == 'momentum':
            self.vel = {name: torch.zeros_like(getattr(self, name)) for name in self.params}
        elif optimizer == 'adam':
            self.m = {name: torch.zeros_like(getattr(self, name)) for name in self.params}
            self.v = {name: torch.zeros_like(getattr(self, name)) for name in self.params}

    def _apply_update(self, name, grad):
        """Apply optimizer update to parameter."""
        if self.optimizer_type == 'sgd':
            param = getattr(self, name)
            param += self.lr * grad
        elif self.optimizer_type == 'momentum':
            self.vel[name] = self.momentum * self.vel[name] + grad
            param = getattr(self, name)
            param += self.lr * self.vel[name]
        elif self.optimizer_type == 'adam':
            self.m[name] = self.adam_beta1 * self.m[name] + (1 - self.adam_beta1) * grad
            self.v[name] = self.adam_beta2 * self.v[name] + (1 - self.adam_beta2) * grad ** 2
            m_hat = self.m[name] / (1 - self.adam_beta1 ** self.t_step)
            v_hat = self.v[name] / (1 - self.adam_beta2 ** self.t_step)
            param = getattr(self, name)
            param += self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_eps)

    def train_step(self, x, targets):
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, HIDDEN_DIM, device=DEVICE)
        self.t_step += 1

        if self.use_full_rtrl:
            J_rec = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, device=DEVICE)
            J_in = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM, device=DEVICE)
        else:
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
            h_new = torch.tanh(a)
            phi_prime = 1 - h_new ** 2

            if self.use_full_rtrl:
                J_rec_flat = J_rec.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM * HIDDEN_DIM)
                propagated_rec = torch.bmm(
                    self.W_rec.unsqueeze(0).expand(batch_size, -1, -1),
                    J_rec_flat
                ).reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM)

                direct_rec = torch.zeros_like(J_rec)
                for i in range(HIDDEN_DIM):
                    direct_rec[:, i, i, :] = h

                J_rec = phi_prime.unsqueeze(2).unsqueeze(3) * (propagated_rec + direct_rec)

                J_in_flat = J_in.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM * INPUT_DIM)
                propagated_in = torch.bmm(
                    self.W_rec.unsqueeze(0).expand(batch_size, -1, -1),
                    J_in_flat
                ).reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM)

                direct_in = torch.zeros_like(J_in)
                for i in range(HIDDEN_DIM):
                    direct_in[:, i, i, :] = x[:, t]

                J_in = phi_prime.unsqueeze(2).unsqueeze(3) * (propagated_in + direct_in)
            else:
                e_rec = self.trace_decay * e_rec + phi_prime.unsqueeze(2) * h.unsqueeze(1)
                e_in = self.trace_decay * e_in + phi_prime.unsqueeze(2) * x[:, t].unsqueeze(1)

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

                if self.use_exact_feedback:
                    L = (self.W_out.T @ delta.unsqueeze(-1)).squeeze(-1)
                else:
                    L = (self.B @ delta.unsqueeze(-1)).squeeze(-1)

                if self.use_full_rtrl:
                    dW_rec += torch.einsum('bi,bijl->jl', L, J_rec) / batch_size
                    dW_in += torch.einsum('bi,bijl->jl', L, J_in) / batch_size
                else:
                    dW_rec += (L.unsqueeze(2) * e_rec).mean(0)
                    dW_in += (L.unsqueeze(2) * e_in).mean(0)

                dW_out += (delta.unsqueeze(2) * h.unsqueeze(1)).mean(0)
                db_out += delta.mean(0)
                db += L.mean(0)

        # Apply updates with optimizer
        self._apply_update('W_rec', dW_rec)
        self._apply_update('W_in', dW_in)
        self._apply_update('W_out', dW_out)
        self._apply_update('b_out', db_out)
        self._apply_update('b', db)

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


def run_experiment(name, **kwargs):
    print(f"  {name}...", end=" ", flush=True)
    model = LocalRNN(**kwargs)
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
    print(f"Copy task seq_len={SEQ_LEN}, hidden={HIDDEN_DIM}, {N_ITERS} iters")
    print("=" * 60)

    all_results = {}

    # Part 1: C4 lr sweep to debug anomalous result
    print("\n--- Part 1: C4 (Exact FB + Full RTRL) LR Sweep ---")
    c4_lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    for lr in c4_lrs:
        name = f"C4_lr={lr}"
        accs, final, elapsed = run_experiment(
            name, use_exact_feedback=True, use_full_rtrl=True, lr=lr, optimizer='sgd'
        )
        all_results[name] = {"accs": accs, "final": final, "time": elapsed}

    # Part 2: Test optimizer effect with SGD, Momentum, Adam on key conditions
    print("\n--- Part 2: Optimizer Effect (Rank-1 Trace + Random FB) ---")
    for opt in ['sgd', 'momentum', 'adam']:
        lr = 0.02 if opt == 'sgd' else (0.02 if opt == 'momentum' else 0.001)
        name = f"Rank1_RandomFB_{opt}_lr={lr}"
        accs, final, elapsed = run_experiment(
            name, use_exact_feedback=False, use_full_rtrl=False,
            lr=lr, optimizer=opt
        )
        all_results[name] = {"accs": accs, "final": final, "time": elapsed}

    print("\n--- Part 3: Optimizer Effect (Full RTRL + Random FB) ---")
    for opt in ['sgd', 'momentum', 'adam']:
        lr = 0.005 if opt == 'sgd' else (0.005 if opt == 'momentum' else 0.001)
        name = f"RTRL_RandomFB_{opt}_lr={lr}"
        accs, final, elapsed = run_experiment(
            name, use_exact_feedback=False, use_full_rtrl=True,
            lr=lr, optimizer=opt
        )
        all_results[name] = {"accs": accs, "final": final, "time": elapsed}

    print("\n--- Part 4: Full RTRL + Exact FB + Adam (should approach BPTT) ---")
    for lr in [0.0005, 0.001, 0.002]:
        name = f"RTRL_ExactFB_adam_lr={lr}"
        accs, final, elapsed = run_experiment(
            name, use_exact_feedback=True, use_full_rtrl=True,
            lr=lr, optimizer='adam'
        )
        all_results[name] = {"accs": accs, "final": final, "time": elapsed}

    # Save results
    results_json = {k: {"final_acc": v["final"], "time": v["time"]} for k, v in all_results.items()}
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Part 1: C4 lr sweep
    ax = axes[0, 0]
    for lr in c4_lrs:
        name = f"C4_lr={lr}"
        iters = [(i+1)*EVAL_EVERY for i in range(len(all_results[name]["accs"]))]
        ax.plot(iters, all_results[name]["accs"], label=f'lr={lr} ({all_results[name]["final"]:.3f})')
    ax.set_title('Part 1: C4 (Exact FB + RTRL) LR Sweep')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Part 2: Optimizer on rank-1
    ax = axes[0, 1]
    for opt in ['sgd', 'momentum', 'adam']:
        lr = 0.02 if opt == 'sgd' else (0.02 if opt == 'momentum' else 0.001)
        name = f"Rank1_RandomFB_{opt}_lr={lr}"
        iters = [(i+1)*EVAL_EVERY for i in range(len(all_results[name]["accs"]))]
        ax.plot(iters, all_results[name]["accs"], label=f'{opt} ({all_results[name]["final"]:.3f})', linewidth=2)
    ax.set_title('Part 2: Rank-1 Trace + Random FB\n(Optimizer Comparison)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Part 3: Optimizer on RTRL
    ax = axes[1, 0]
    for opt in ['sgd', 'momentum', 'adam']:
        lr = 0.005 if opt == 'sgd' else (0.005 if opt == 'momentum' else 0.001)
        name = f"RTRL_RandomFB_{opt}_lr={lr}"
        iters = [(i+1)*EVAL_EVERY for i in range(len(all_results[name]["accs"]))]
        ax.plot(iters, all_results[name]["accs"], label=f'{opt} ({all_results[name]["final"]:.3f})', linewidth=2)
    ax.set_title('Part 3: Full RTRL + Random FB\n(Optimizer Comparison)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Part 4: RTRL + exact + Adam
    ax = axes[1, 1]
    for lr in [0.0005, 0.001, 0.002]:
        name = f"RTRL_ExactFB_adam_lr={lr}"
        iters = [(i+1)*EVAL_EVERY for i in range(len(all_results[name]["accs"]))]
        ax.plot(iters, all_results[name]["accs"], label=f'lr={lr} ({all_results[name]["final"]:.3f})', linewidth=2)
    ax.set_title('Part 4: Full RTRL + Exact FB + Adam\n(Should approach BPTT)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'optimizer_effect.png'), dpi=150, bbox_inches='tight')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["final"]):
        print(f"  {name}: {r['final']:.3f}")

    print(f"\nPlots saved to: {RESULTS_DIR}/optimizer_effect.png")


if __name__ == '__main__':
    main()
