"""
Experiment 05: Approximate RTRL — Block-Diagonal and Random Projection
Find the minimum trace richness needed to solve copy task.
Compare: rank-1 (baseline) < approx RTRL < full RTRL (ceiling)
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp05')
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


class BlockDiagonalRTRL:
    """
    Block-diagonal approximation to RTRL.
    Partition n neurons into n/K blocks of size K.
    Only track within-block Jacobians, ignoring cross-block influence propagation.
    Forward pass uses full W_rec (connections span blocks).
    """

    def __init__(self, block_size=8, lr=LR):
        self.block_size = block_size
        self.n_blocks = HIDDEN_DIM // block_size
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

        # Block-diagonal Jacobians for W_rec
        # For block b: J_b[batch, K, K, K] = dh[block_b_neurons] / dW_rec[block_b_rows, block_b_cols]
        # Shape per block: (batch, K, K, K)
        # Total blocks: n_blocks
        J_rec_blocks = [torch.zeros(batch_size, K, K, K, device=DEVICE) for _ in range(self.n_blocks)]

        # For W_in: block-diagonal means dh[block_b] / dW_in[block_b, :]
        # Shape per block: (batch, K, K, INPUT_DIM)
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
            phi_prime = 1 - h_new ** 2  # (batch, n)

            # Update block Jacobians
            for b_idx in range(self.n_blocks):
                s = b_idx * K
                e = s + K

                # Within-block recurrent weight sub-matrix
                W_block = self.W_rec[s:e, s:e]  # (K, K)
                phi_block = phi_prime[:, s:e]  # (batch, K)

                # Propagate: J_new[i,j,k] = phi'[i] * (W_block[i,:] @ J_old[:,j,k] + delta(i==j)*h[k])
                # J_rec_blocks[b_idx] shape: (batch, K, K, K)
                # W_block @ J: einsum('ij, bjkl -> bikl', W_block, J)
                propagated = torch.einsum('ij,bjkl->bikl', W_block, J_rec_blocks[b_idx])

                # Direct contribution: delta(i==j) * h_{t-1}[s:e] (within block)
                h_block = h[:, s:e]  # (batch, K)
                direct = torch.zeros_like(J_rec_blocks[b_idx])
                for i in range(K):
                    direct[:, i, i, :] = h_block

                J_rec_blocks[b_idx] = phi_block.unsqueeze(2).unsqueeze(3) * (propagated + direct)

                # W_in Jacobian update
                propagated_in = torch.einsum('ij,bjkl->bikl', W_block, J_in_blocks[b_idx])
                direct_in = torch.zeros_like(J_in_blocks[b_idx])
                for i in range(K):
                    direct_in[:, i, i, :] = x[:, t]

                J_in_blocks[b_idx] = phi_block.unsqueeze(2).unsqueeze(3) * (propagated_in + direct_in)

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

                L = (self.B @ delta.unsqueeze(-1)).squeeze(-1)  # (batch, n)

                # Compute weight updates using block Jacobians
                for b_idx in range(self.n_blocks):
                    s = b_idx * K
                    e = s + K
                    L_block = L[:, s:e]  # (batch, K)

                    # dW_rec[s:e, s:e] += sum_i L[i] * J[i, j, k]
                    dW_rec[s:e, s:e] += torch.einsum('bi,bijl->jl', L_block, J_rec_blocks[b_idx]) / batch_size

                    # dW_in[s:e, :] += sum_i L[i] * J_in[i, j, k]
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


class RandomProjectionRTRL:
    """
    Random projection approximation to RTRL.
    Maintain R independent projected traces instead of full n-dimensional Jacobian.
    Each trace compresses the Jacobian along the output dimension using a fixed random projection.

    Key idea: Instead of J[i,j,k] (n×n×n), maintain Jc[r,j,k] = sum_i Q[r,i] * J[i,j,k]
    where Q ∈ R^{R×n} is a fixed random projection matrix.
    Memory: R × n × n (for W_rec) instead of n × n × n.
    """

    def __init__(self, n_projections=4, lr=LR):
        self.R = n_projections
        self.lr = lr

        self.W_in = torch.randn(HIDDEN_DIM, INPUT_DIM, device=DEVICE) * 0.01
        self.W_rec = torch.randn(HIDDEN_DIM, HIDDEN_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))
        self.b = torch.zeros(HIDDEN_DIM, device=DEVICE)
        self.W_out = torch.randn(OUTPUT_DIM, HIDDEN_DIM, device=DEVICE) * 0.01
        self.b_out = torch.zeros(OUTPUT_DIM, device=DEVICE)
        self.B_fb = torch.randn(HIDDEN_DIM, OUTPUT_DIM, device=DEVICE) * (1.0 / np.sqrt(HIDDEN_DIM))

        # Fixed random projection: Q ∈ R^{R×n}
        self.Q = torch.randn(self.R, HIDDEN_DIM, device=DEVICE) / np.sqrt(self.R)

    def train_step(self, x, targets):
        batch_size, total_len, _ = x.shape
        h = torch.zeros(batch_size, HIDDEN_DIM, device=DEVICE)

        # Compressed Jacobians: Jc_rec[batch, R, n_j, n_k]
        Jc_rec = torch.zeros(batch_size, self.R, HIDDEN_DIM, HIDDEN_DIM, device=DEVICE)
        Jc_in = torch.zeros(batch_size, self.R, HIDDEN_DIM, INPUT_DIM, device=DEVICE)

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
            phi_prime = 1 - h_new ** 2  # (batch, n)

            # Update compressed Jacobian for W_rec
            # Full RTRL: J_new[i,j,k] = phi'[i] * (sum_l W_rec[i,l] * J[l,j,k] + delta(i==j)*h[k])
            # Projected: Jc_new[r,j,k] = sum_i Q[r,i] * phi'[i] * (sum_l W_rec[i,l] * J[l,j,k] + delta(i==j)*h[k])
            # = sum_i Q[r,i]*phi'[i] * sum_l W_rec[i,l] * J[l,j,k]  +  Q[r,j]*phi'[j]*h[k]
            #
            # First term: need sum_l (sum_i Q[r,i]*phi'[i]*W_rec[i,l]) * J[l,j,k]
            # But J[l,j,k] is not available! We only have Jc[r,j,k] = sum_l Q[r,l] * J[l,j,k]
            #
            # This is the fundamental issue with random projection RTRL.
            # Approximation: assume J[l,j,k] ≈ Q^T @ Jc[:,j,k] (pseudoinverse reconstruction)
            # Then: first term ≈ sum_l (A[r,l]) * (sum_r2 Q^T[l,r2] * Jc[r2,j,k])
            # where A[r,l] = sum_i Q[r,i]*phi'[i]*W_rec[i,l]
            #
            # Actually a cleaner approach: use the APPROXIMATION that the projection commutes
            # Jc_new[r,j,k] ≈ sum_l M[r,l] * Jc_old[l,j,k] + Q[r,j]*phi'[j]*h[k]
            # where M[r,l] = sum_i Q[r,i] * phi'[i] * W_rec[i,l] * Q_pinv[l, ???]
            #
            # Simplest viable approximation (from SnAp-R literature):
            # Maintain R independent rank-1 traces, each with its own random direction.
            # e_r_rec[j,k] tracks sensitivity along direction Q[r,:]
            # e_r_new[j,k] = sum_l (Q[r,:] · phi' * W_rec[:,l]) * ... this gets circular.
            #
            # Let's use a different approach: DIRECT PROJECTED TRACES
            # For each projection direction r, maintain a trace:
            #   Jc_rec[r, j, k] evolves with a simplified recursion:
            #   Jc_rec_new[r,j,k] = sum_l W_eff[r,l] * Jc_rec_old[l,j,k] + Q[r,j]*phi'[j]*h[k]
            # where W_eff[r,l] = Q[r,:] @ diag(phi') @ W_rec @ Q^T  ... still complex

            # PRAGMATIC APPROACH: Use trace_decay-based persistence + projected direct contribution
            # This is a multi-trace RFLO with R different projection directions:
            # Jc_rec[r,j,k] = trace_decay * Jc_rec[r,j,k] + Q[r,j] * phi'[j] * h[k]
            # This gives R rank-1 traces, each measuring sensitivity in direction Q[r,:]
            # The key difference from single RFLO: we have R diverse measurements

            # Actually let's do this properly using the correct recursion but approximated:
            # Use the factored form. The full recursion for the projected trace is:
            # Jc_new = (Q @ diag(phi') @ W_rec @ Q^+) @ Jc_old + direct
            # With Q^+ = Q^T (Q @ Q^T)^{-1} ≈ Q^T (since Q is approx orthogonal for large n)
            # So: effective_W = Q @ diag(phi') @ W_rec @ Q^T / R  (R×R matrix)
            # Then: Jc_new[r, j, k] = sum_r2 effective_W[r,r2] * Jc_old[r2,j,k] + Q[r,j]*phi'[j]*h[k]

            # Compute effective transition matrix for projections
            # effective_W[r1, r2] = sum_i Q[r1,i] * phi'[i] * (sum_l W_rec[i,l] * Q[r2,l]) / something
            # = Q @ diag(phi') @ W_rec @ Q^T  (shape: R×R)
            phi_diag = phi_prime  # (batch, n)
            # For each batch: eff_W = Q @ diag(phi'[b]) @ W_rec @ Q^T
            # Q is (R, n), phi is (batch, n), W_rec is (n, n)
            # Vectorized: (Q * phi[b]) @ W_rec @ Q^T
            QP = self.Q.unsqueeze(0) * phi_prime.unsqueeze(1)  # (batch, R, n)
            eff_W = torch.bmm(QP @ self.W_rec, self.Q.T.unsqueeze(0).expand(batch_size, -1, -1))  # (batch, R, R)

            # Propagate: Jc_new = eff_W @ Jc_old + direct
            # Jc_rec shape: (batch, R, n, n) — reshape for bmm
            Jc_rec_flat = Jc_rec.reshape(batch_size, self.R, HIDDEN_DIM * HIDDEN_DIM)
            propagated = torch.bmm(eff_W, Jc_rec_flat)  # (batch, R, n*n)
            propagated = propagated.reshape(batch_size, self.R, HIDDEN_DIM, HIDDEN_DIM)

            # Direct: Q[r,j] * phi'[j] * h[k]
            direct = torch.einsum('rj,bj,bk->brjk', self.Q, phi_prime, h)
            Jc_rec = propagated + direct

            # Same for W_in
            Jc_in_flat = Jc_in.reshape(batch_size, self.R, HIDDEN_DIM * INPUT_DIM)
            propagated_in = torch.bmm(eff_W, Jc_in_flat).reshape(batch_size, self.R, HIDDEN_DIM, INPUT_DIM)
            direct_in = torch.einsum('rj,bj,bk->brjk', self.Q, phi_prime, x[:, t])
            Jc_in = propagated_in + direct_in

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

                L = (self.B_fb @ delta.unsqueeze(-1)).squeeze(-1)  # (batch, n)

                # Weight update: dW[j,k] = sum_i L[i] * J[i,j,k]
                # ≈ sum_r (Q^T @ L)[r] * Jc[r,j,k] = sum_r (L @ Q^T)[r] * Jc[r,j,k]
                # projected_L[r] = sum_i Q[r,i] * L[i]
                projected_L = torch.einsum('ri,bi->br', self.Q, L)  # (batch, R)
                dW_rec += torch.einsum('br,brjk->jk', projected_L, Jc_rec) / batch_size
                dW_in += torch.einsum('br,brjk->jk', projected_L, Jc_in) / batch_size

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


def run_model(name, model, n_iters=N_ITERS):
    print(f"  {name}...", end=" ", flush=True)
    eval_accs = []
    t0 = time.time()

    for i in range(n_iters):
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

    # Block-diagonal RTRL
    print("\n--- Block-Diagonal RTRL ---")
    for K in [4, 8, 16, 32]:
        name = f"BlockDiag_K={K}"
        model = BlockDiagonalRTRL(block_size=K, lr=LR)
        accs, final, elapsed = run_model(name, model)
        memory = HIDDEN_DIM * K * K  # approximate memory in trace entries
        all_results[name] = {"accs": accs, "final": final, "time": elapsed, "memory": memory, "type": "block"}

    # Random Projection RTRL
    print("\n--- Random Projection RTRL ---")
    for R in [2, 4, 8, 16, 32]:
        name = f"RandProj_R={R}"
        model = RandomProjectionRTRL(n_projections=R, lr=LR)
        accs, final, elapsed = run_model(name, model)
        memory = R * HIDDEN_DIM * HIDDEN_DIM
        all_results[name] = {"accs": accs, "final": final, "time": elapsed, "memory": memory, "type": "proj"}

    # Save results
    results_json = {k: {"final_acc": v["final"], "time": v["time"], "memory": v["memory"]}
                    for k, v in all_results.items()}
    results_json["reference"] = {
        "rank1_RFLO": 0.28,
        "full_RTRL": 1.00,
        "full_RTRL_memory": HIDDEN_DIM ** 3,
    }
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Learning curves - Block diagonal
    ax = axes[0]
    for name, data in all_results.items():
        if data["type"] == "block":
            iters = [(i+1)*EVAL_EVERY for i in range(len(data["accs"]))]
            ax.plot(iters, data["accs"], label=f'{name} ({data["final"]:.3f})', linewidth=2)
    ax.axhline(y=0.28, color='r', linestyle='--', alpha=0.5, label='Rank-1 (RFLO)')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Full RTRL')
    ax.set_title('Block-Diagonal RTRL')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Learning curves - Random projection
    ax = axes[1]
    for name, data in all_results.items():
        if data["type"] == "proj":
            iters = [(i+1)*EVAL_EVERY for i in range(len(data["accs"]))]
            ax.plot(iters, data["accs"], label=f'{name} ({data["final"]:.3f})', linewidth=2)
    ax.axhline(y=0.28, color='r', linestyle='--', alpha=0.5, label='Rank-1 (RFLO)')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Full RTRL')
    ax.set_title('Random Projection RTRL')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Accuracy vs Memory tradeoff
    ax = axes[2]
    for name, data in all_results.items():
        marker = 's' if data["type"] == "block" else 'o'
        color = 'blue' if data["type"] == "block" else 'red'
        ax.scatter(data["memory"], data["final"], marker=marker, color=color, s=100, zorder=5)
        ax.annotate(name.split('_')[1], (data["memory"], data["final"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(y=0.28, color='gray', linestyle='--', alpha=0.5, label='Rank-1')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Full RTRL')
    ax.axvline(x=HIDDEN_DIM**3, color='green', linestyle=':', alpha=0.5)
    ax.set_title('Accuracy vs Memory')
    ax.set_xlabel('Memory (# trace entries)')
    ax.set_ylabel('Accuracy')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'approx_rtrl.png'), dpi=150, bbox_inches='tight')

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by accuracy)")
    print("=" * 60)
    print(f"  {'Method':<25} {'Accuracy':>8} {'Memory':>10} {'Time':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
    for name, data in sorted(all_results.items(), key=lambda x: -x[1]["final"]):
        print(f"  {name:<25} {data['final']:>8.3f} {data['memory']:>10,} {data['time']:>7.1f}s")
    print(f"  {'Rank-1 (RFLO)':<25} {'0.280':>8} {HIDDEN_DIM**2:>10,} {'~115':>8}")
    print(f"  {'Full RTRL':<25} {'1.000':>8} {HIDDEN_DIM**3:>10,} {'~550':>8}")
    print(f"\nPlots saved to: {RESULTS_DIR}/approx_rtrl.png")


if __name__ == '__main__':
    main()
