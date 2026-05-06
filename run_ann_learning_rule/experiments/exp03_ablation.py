"""
Experiment 03: Ablation — Random Feedback vs Trace Approximation
Determine whether RFLO's bottleneck is:
  (a) Random feedback matrix B, or
  (b) Rank-1 eligibility trace approximation

4 Conditions:
  1. RFLO standard (random B + rank-1 trace)
  2. Exact feedback (W_out^T) + rank-1 trace
  3. Random B + full RTRL (exact influence matrix)
  4. Exact feedback + full RTRL (upper bound, ~BPTT)
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp03')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 10
HIDDEN_DIM = 64  # smaller for RTRL feasibility
N_SYMBOLS = 8
INPUT_DIM = N_SYMBOLS + 2
OUTPUT_DIM = N_SYMBOLS
BATCH_SIZE = 32
N_ITERS = 10000
EVAL_EVERY = 200
TRACE_DECAY = 0.9  # separate from network dynamics


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


class AblationRNN:
    """
    Unified RNN class supporting 4 ablation conditions.
    Network: h_t = tanh(W_rec @ h_{t-1} + W_in @ x_t + b)  [standard discrete RNN]
    """

    def __init__(self, use_exact_feedback=False, use_full_rtrl=False, lr=0.01, trace_decay=TRACE_DECAY):
        self.use_exact_feedback = use_exact_feedback
        self.use_full_rtrl = use_full_rtrl
        self.lr = lr
        self.trace_decay = trace_decay

        self.W_in = torch.randn(HIDDEN_DIM, INPUT_DIM, device=DEVICE) * 0.01
        self.W_rec = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))
        self.b = torch.zeros(HIDDEN_DIM, device=DEVICE)
        self.W_out = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=DEVICE) * 0.01
        self.b_out = torch.zeros(OUTPUT_DIM, device=DEVICE)

        if not use_exact_feedback:
            self.B = torch.randn(HIDDEN_DIM, OUTPUT_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))

    def get_feedback_matrix(self):
        if self.use_exact_feedback:
            return self.W_out.T.clone()  # (hidden, output)
        else:
            return self.B

    def train_step(self, x, targets):
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, HIDDEN_DIM, device=DEVICE)

        if self.use_full_rtrl:
            # Full RTRL: maintain dh/dW_rec of shape (batch, hidden, hidden, hidden)
            # dh_t^i / dW_rec^{j,k} — but we only need the influence on output
            # More memory-efficient: maintain dh/dW_rec as (batch, hidden, hidden*hidden)
            # Actually for weight update we need: sum over output of (error * W_out) * dh/dW
            # Let's maintain dh_t/dW_rec[i,:] for each neuron i
            # Shape: (batch, hidden_i, hidden_j, hidden_k) = dh_t^i / dW_rec^{j,k}
            # This is O(n^3) per batch element

            # For W_rec: dh_t^i / dW_rec^{a,b}
            # = phi'(a_t^i) * [W_rec^{i,:} @ dh_{t-1}/dW_rec^{a,b} + delta_{i=a} * h_{t-1}^b]
            # We store J_rec[b, i, a, b_idx] = dh_t^i / dW_rec^{a, b_idx}
            J_rec = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, device=DEVICE)

            # For W_in: dh_t^i / dW_in^{a,b}
            # = phi'(a_t^i) * [W_rec^{i,:} @ dh_{t-1}/dW_in^{a,b} + delta_{i=a} * x_t^b]
            J_in = torch.zeros(batch_size, HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM, device=DEVICE)
        else:
            # Rank-1 trace
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
            phi_prime = 1 - h_new ** 2  # (batch, hidden)

            if self.use_full_rtrl:
                # Update Jacobian for W_rec
                # J_rec_new[b, i, a, b_idx] = phi'[b,i] * (sum_k W_rec[i,k]*J_rec[b,k,a,b_idx] + delta(i==a)*h[b,b_idx])
                # Vectorized: for each i, sum over k
                # W_rec[i,:] @ J_rec[b,:,a,b_idx] for all a,b_idx
                # Shape manipulation: J_rec is (batch, hidden, hidden, hidden)
                # W_rec @ J_rec along dim 1

                # J_rec[:, :, a, b_idx] represents dh/dW_rec[a, b_idx] for all neurons
                # New: phi'[b,i] * (W_rec[i,:] @ J_rec[b,:,a,b_idx] + delta(i==a)*h[b,b_idx])

                # Efficient computation:
                # propagated = einsum('ik,bkab2->biab2', W_rec, J_rec) but that's (batch,H,H,H)
                # Let's reshape: J_rec_flat = (batch, H, H*H), W_rec is (H, H)
                J_rec_flat = J_rec.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM * HIDDEN_DIM)
                propagated_rec = torch.bmm(
                    self.W_rec.unsqueeze(0).expand(batch_size, -1, -1),
                    J_rec_flat
                )  # (batch, H, H*H)
                propagated_rec = propagated_rec.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM)

                # Add direct contribution: delta(i==a) * h[b, b_idx]
                direct_rec = torch.zeros_like(J_rec)
                # direct_rec[b, i, i, :] = h[b, :]
                for i in range(HIDDEN_DIM):
                    direct_rec[:, i, i, :] = h

                J_rec = phi_prime.unsqueeze(2).unsqueeze(3) * (propagated_rec + direct_rec)

                # Update Jacobian for W_in
                J_in_flat = J_in.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM * INPUT_DIM)
                propagated_in = torch.bmm(
                    self.W_rec.unsqueeze(0).expand(batch_size, -1, -1),
                    J_in_flat
                )
                propagated_in = propagated_in.reshape(batch_size, HIDDEN_DIM, HIDDEN_DIM, INPUT_DIM)

                direct_in = torch.zeros_like(J_in)
                for i in range(HIDDEN_DIM):
                    direct_in[:, i, i, :] = x[:, t]

                J_in = phi_prime.unsqueeze(2).unsqueeze(3) * (propagated_in + direct_in)

            else:
                # Rank-1 trace update (decoupled from network alpha)
                e_rec = self.trace_decay * e_rec + phi_prime.unsqueeze(2) * h.unsqueeze(1)
                e_in = self.trace_decay * e_in + phi_prime.unsqueeze(2) * x[:, t].unsqueeze(1)

            h = h_new

            # Output at the recall phase
            output_step = t - (total_len - SEQ_LEN)
            if output_step >= 0 and output_step < SEQ_LEN:
                y = (self.W_out @ h.unsqueeze(-1)).squeeze(-1) + self.b_out
                probs = torch.softmax(y, dim=-1)
                target_onehot = torch.zeros_like(probs)
                target_onehot.scatter_(1, targets[:, output_step].unsqueeze(1), 1.0)
                delta = target_onehot - probs  # (batch, output_dim)

                log_probs = torch.log_softmax(y, dim=-1)
                loss = -log_probs.gather(1, targets[:, output_step].unsqueeze(1)).mean()
                total_loss += loss.item()

                preds = y.argmax(dim=-1)
                correct += (preds == targets[:, output_step]).sum().item()
                total += batch_size

                # Feedback: either random B or exact W_out^T
                B = self.get_feedback_matrix()
                L = (B @ delta.unsqueeze(-1)).squeeze(-1)  # (batch, hidden)

                if self.use_full_rtrl:
                    # Weight update: dW_rec[a,b] += L^i * J_rec[b,i,a,b]
                    # = sum_i L[b,i] * J_rec[b,i,a,b_idx]
                    # (batch, hidden, hidden) = einsum('bi,biab->bab', L, J_rec)
                    dW_rec += torch.einsum('bi,bijl->jl', L, J_rec).mean(0) if batch_size > 0 else dW_rec
                    dW_in += torch.einsum('bi,bijl->jl', L, J_in).mean(0) if batch_size > 0 else dW_in
                else:
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


def run_condition(name, use_exact_feedback, use_full_rtrl, lr):
    print(f"\n  [{name}] exact_fb={use_exact_feedback}, full_rtrl={use_full_rtrl}, lr={lr}")
    model = AblationRNN(
        use_exact_feedback=use_exact_feedback,
        use_full_rtrl=use_full_rtrl,
        lr=lr,
        trace_decay=TRACE_DECAY
    )

    eval_accs = []
    t0 = time.time()

    for i in range(N_ITERS):
        x, targets = generate_copy_task(BATCH_SIZE)
        loss, acc = model.train_step(x, targets)

        if (i + 1) % EVAL_EVERY == 0:
            x_test, t_test = generate_copy_task(500)
            test_acc = model.evaluate(x_test, t_test)
            eval_accs.append(test_acc)
            if (i + 1) % 2000 == 0:
                print(f"    iter {i+1}: acc={test_acc:.3f}")

    elapsed = time.time() - t0

    # Final eval
    x_test, t_test = generate_copy_task(2000)
    final_acc = model.evaluate(x_test, t_test)
    print(f"    FINAL: acc={final_acc:.3f} ({elapsed:.1f}s)")

    return eval_accs, final_acc, elapsed


def main():
    print(f"Device: {DEVICE}")
    print(f"Task: Copy task, seq_len={SEQ_LEN}, hidden={HIDDEN_DIM}")
    print(f"Training: {N_ITERS} iterations, batch={BATCH_SIZE}, trace_decay={TRACE_DECAY}")
    print("=" * 60)

    conditions = {
        "C1: Random FB + Rank-1 Trace (RFLO)": {
            "use_exact_feedback": False, "use_full_rtrl": False, "lr": 0.02
        },
        "C2: Exact FB + Rank-1 Trace": {
            "use_exact_feedback": True, "use_full_rtrl": False, "lr": 0.02
        },
        "C3: Random FB + Full RTRL": {
            "use_exact_feedback": False, "use_full_rtrl": True, "lr": 0.005
        },
        "C4: Exact FB + Full RTRL": {
            "use_exact_feedback": True, "use_full_rtrl": True, "lr": 0.005
        },
    }

    results = {}
    all_accs = {}

    for name, cfg in conditions.items():
        accs, final, elapsed = run_condition(name, **cfg)
        results[name] = {"final_acc": final, "time": elapsed, "config": cfg}
        all_accs[name] = accs

    # Save results
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name, accs in all_accs.items():
        iters = [(i+1)*EVAL_EVERY for i in range(len(accs))]
        ax.plot(iters, accs, label=f'{name} (final={results[name]["final_acc"]:.3f})', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Ablation: Random Feedback vs Trace Approximation\n(Copy Task, seq_len=10, hidden=64)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.125, color='r', linestyle='--', alpha=0.3, label='random baseline')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ablation.png'), dpi=150, bbox_inches='tight')

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: acc={r['final_acc']:.3f} ({r['time']:.1f}s)")

    print("\n--- Interpretation ---")
    c1 = results["C1: Random FB + Rank-1 Trace (RFLO)"]["final_acc"]
    c2 = results["C2: Exact FB + Rank-1 Trace"]["final_acc"]
    c3 = results["C3: Random FB + Full RTRL"]["final_acc"]
    c4 = results["C4: Exact FB + Full RTRL"]["final_acc"]

    print(f"  Effect of exact feedback (C2 - C1): {c2 - c1:+.3f}")
    print(f"  Effect of full RTRL (C3 - C1):      {c3 - c1:+.3f}")
    print(f"  Combined (C4 - C1):                 {c4 - c1:+.3f}")

    if c3 - c1 > c2 - c1 + 0.05:
        print("\n  → TRACE APPROXIMATION is the primary bottleneck")
    elif c2 - c1 > c3 - c1 + 0.05:
        print("\n  → RANDOM FEEDBACK is the primary bottleneck")
    else:
        print("\n  → BOTH contribute significantly to the bottleneck")

    print(f"\nPlots saved to: {RESULTS_DIR}/ablation.png")
    print(f"Results saved to: {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()
