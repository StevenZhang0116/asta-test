"""
Step 3 experiment — copy task with BPTT / RFLO / STC-Prop.

Comparison of three recurrent learning rules on a small copy task:
  - BPTT: PyTorch autograd reference.
  - RFLO: Murray 2019 local online rule (per-synapse eligibility + random feedback).
  - STC-Prop: the candidate rule from design_step2.md — two-timescale tagging
    with per-neuron discrete capture driven by local surprise, plus a no-gate
    ablation that sets c_i(t) ≡ 1.

This file is intentionally a single self-contained script so Step 4 can point
at it as a reproducible harness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Task: copy task
# -----------------------------------------------------------------------------

def make_copy_batch(batch_size: int, S: int, L: int, K: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Generate a batched copy task.

    K tokens total. Token 0 = blank, tokens 1..K-2 = data symbols, K-1 = go.
    Sequence layout: [S data tokens] [L blanks] [1 go token] [S blanks].
    Target:          [S blanks]      [L blanks] [1 blank]   [S data tokens].
    Returns (x, y, T) with x, y ∈ Long[T, batch].
    """
    num_data = K - 2
    T = S + L + 1 + S
    data = torch.randint(1, 1 + num_data, (S, batch_size), device=device)
    x = torch.zeros(T, batch_size, dtype=torch.long, device=device)
    x[:S] = data
    x[S + L] = K - 1  # go token
    y = torch.zeros(T, batch_size, dtype=torch.long, device=device)
    y[S + L + 1:] = data
    return x, y, T


def one_hot(tokens: torch.Tensor, K: int) -> torch.Tensor:
    return F.one_hot(tokens, num_classes=K).float()


# -----------------------------------------------------------------------------
# Shared model primitives
# -----------------------------------------------------------------------------

@dataclass
class RNNParams:
    N: int
    K: int
    W_in: torch.Tensor
    W_rec: torch.Tensor
    b: torch.Tensor
    W_out: torch.Tensor
    b_out: torch.Tensor

    @staticmethod
    def init(N: int, K: int, device) -> "RNNParams":
        g = 1.0 / math.sqrt(N)
        W_in = torch.randn(N, K, device=device) * (1.0 / math.sqrt(K))
        W_rec = torch.randn(N, N, device=device) * g
        b = torch.zeros(N, device=device)
        W_out = torch.randn(K, N, device=device) * g
        b_out = torch.zeros(K, device=device)
        return RNNParams(N, K, W_in, W_rec, b, W_out, b_out)

    def clone(self) -> "RNNParams":
        return RNNParams(self.N, self.K,
                         self.W_in.clone(), self.W_rec.clone(), self.b.clone(),
                         self.W_out.clone(), self.b_out.clone())


def rnn_step(h: torch.Tensor, x_onehot: torch.Tensor, p: RNNParams) -> tuple[torch.Tensor, torch.Tensor]:
    """One RNN step. x_onehot: [B, K], h: [B, N]. Returns (h_new, u_pre)."""
    u = x_onehot @ p.W_in.T + h @ p.W_rec.T + p.b
    h_new = torch.tanh(u)
    return h_new, u


def rnn_output(h: torch.Tensor, p: RNNParams) -> torch.Tensor:
    return h @ p.W_out.T + p.b_out


def seq_loss_and_acc(logits: torch.Tensor, y: torch.Tensor, S: int, L: int) -> tuple[torch.Tensor, float]:
    """Cross-entropy over all timesteps + accuracy on the copy window only."""
    T, B, K = logits.shape
    loss = F.cross_entropy(logits.reshape(T * B, K), y.reshape(T * B), reduction="mean")
    # Accuracy: compare predicted token to target on the output window only.
    pred = logits.argmax(-1)
    copy_window = slice(S + L + 1, T)
    correct = (pred[copy_window] == y[copy_window]).float().mean().item()
    return loss, correct


# -----------------------------------------------------------------------------
# BPTT
# -----------------------------------------------------------------------------

def train_bptt(p: RNNParams, cfg, device) -> dict:
    """Train with BPTT via PyTorch autograd + Adam."""
    params = [p.W_in, p.W_rec, p.b, p.W_out, p.b_out]
    for t in params:
        t.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.lr)
    N, K = p.N, p.K

    losses, accs, times = [], [], []
    t_start = time.time()
    for it in range(cfg.num_iters):
        x, y, T = make_copy_batch(cfg.batch_size, cfg.S, cfg.L, K, device)
        x_oh = one_hot(x, K)  # [T, B, K]
        h = torch.zeros(cfg.batch_size, N, device=device)
        logits_list = []
        for t in range(T):
            h, _ = rnn_step(h, x_oh[t], p)
            logits_list.append(rnn_output(h, p))
        logits = torch.stack(logits_list, dim=0)
        loss, acc = seq_loss_and_acc(logits, y, cfg.S, cfg.L)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (it + 1) % cfg.log_every == 0 or it == 0:
            losses.append((it + 1, loss.item()))
            accs.append((it + 1, acc))
            times.append((it + 1, time.time() - t_start))
    for t in params:
        t.requires_grad_(False)
    return {"losses": losses, "accs": accs, "times": times, "final_loss": losses[-1][1], "final_acc": accs[-1][1], "wall_clock": time.time() - t_start}


# -----------------------------------------------------------------------------
# RFLO (Murray 2019)
# -----------------------------------------------------------------------------

def train_rflo(p: RNNParams, cfg, device) -> dict:
    """Murray 2019 RFLO: per-synapse eligibility + random-feedback projection."""
    N, K = p.N, p.K
    B = cfg.batch_size
    # Fixed random feedback matrix for the recurrent-layer credit signal.
    B_fb = torch.randn(N, K, device=device) * (1.0 / math.sqrt(K))

    losses, accs, times = [], [], []
    t_start = time.time()
    for it in range(cfg.num_iters):
        x, y, T = make_copy_batch(B, cfg.S, cfg.L, K, device)
        x_oh = one_hot(x, K)
        h = torch.zeros(B, N, device=device)
        e_rec = torch.zeros(B, N, N, device=device)  # eligibility for W_rec
        e_in  = torch.zeros(B, N, K, device=device)  # eligibility for W_in

        dWrec = torch.zeros_like(p.W_rec)
        dWin  = torch.zeros_like(p.W_in)
        dWout = torch.zeros_like(p.W_out)
        dbout = torch.zeros_like(p.b_out)

        total_loss = 0.0
        correct_out = 0.0
        out_steps = 0
        for t in range(T):
            h_prev = h
            h, u = rnn_step(h, x_oh[t], p)
            logits = rnn_output(h, p)
            # Cross-entropy on this timestep.
            probs = F.softmax(logits, dim=-1)
            y_t = y[t]
            loss_t = F.cross_entropy(logits, y_t, reduction="mean")
            total_loss += loss_t.item()
            if t >= cfg.S + cfg.L + 1:
                correct_out += (logits.argmax(-1) == y_t).float().mean().item()
                out_steps += 1
            # ---- eligibility update ----
            phi_prime = 1.0 - h * h               # tanh' from pre-activation via h
            e_rec = cfg.alpha * e_rec + phi_prime.unsqueeze(2) * h_prev.unsqueeze(1)
            e_in  = cfg.alpha * e_in  + phi_prime.unsqueeze(2) * x_oh[t].unsqueeze(1)
            # ---- local per-neuron modulator via random feedback ----
            err = probs - F.one_hot(y_t, K).float()   # [B, K]
            M = err @ B_fb.T                           # [B, N]
            # ---- updates ----
            dWrec = dWrec + cfg.lr_local * (M.unsqueeze(2) * e_rec).mean(dim=0)
            dWin  = dWin  + cfg.lr_local * (M.unsqueeze(2) * e_in ).mean(dim=0)
            # Output layer is trained normally: true-gradient local update.
            dWout = dWout + cfg.lr_local * (err.unsqueeze(2) * h.unsqueeze(1)).mean(dim=0)
            dbout = dbout + cfg.lr_local * err.mean(dim=0)

        # Apply the accumulated updates once per sequence.
        p.W_rec = p.W_rec - dWrec
        p.W_in  = p.W_in  - dWin
        p.W_out = p.W_out - dWout
        p.b_out = p.b_out - dbout

        if (it + 1) % cfg.log_every == 0 or it == 0:
            mean_loss = total_loss / T
            acc = correct_out / max(out_steps, 1)
            losses.append((it + 1, mean_loss))
            accs.append((it + 1, acc))
            times.append((it + 1, time.time() - t_start))

    return {"losses": losses, "accs": accs, "times": times, "final_loss": losses[-1][1], "final_acc": accs[-1][1], "wall_clock": time.time() - t_start}


# -----------------------------------------------------------------------------
# STC-Prop (the candidate)
# -----------------------------------------------------------------------------

def train_stcprop(p: RNNParams, cfg, device, no_gate: bool = False) -> dict:
    """STC-Prop. If no_gate=True, c_i(t) ≡ 1 (ablation)."""
    N, K = p.N, p.K
    B = cfg.batch_size
    B_fb = torch.randn(N, K, device=device) * (1.0 / math.sqrt(K))

    losses, accs, times = [], [], []
    capture_rate_log = []
    t_start = time.time()

    theta = torch.zeros(B, N, device=device)    # homeostatic threshold
    h_hat = torch.zeros(B, N, device=device)    # local predictor state (persists across sequences for simplicity)

    for it in range(cfg.num_iters):
        x, y, T = make_copy_batch(B, cfg.S, cfg.L, K, device)
        x_oh = one_hot(x, K)
        h = torch.zeros(B, N, device=device)
        e_rec = torch.zeros(B, N, N, device=device)
        e_in  = torch.zeros(B, N, K, device=device)
        s_rec = torch.zeros(B, N, N, device=device)
        s_in  = torch.zeros(B, N, K, device=device)

        dWrec = torch.zeros_like(p.W_rec)
        dWin  = torch.zeros_like(p.W_in)
        dWout = torch.zeros_like(p.W_out)
        dbout = torch.zeros_like(p.b_out)

        total_loss = 0.0
        correct_out = 0.0
        out_steps = 0
        captures_fired = 0
        captures_denom = 0
        for t in range(T):
            h_prev = h
            h, u = rnn_step(h, x_oh[t], p)
            logits = rnn_output(h, p)
            probs = F.softmax(logits, dim=-1)
            y_t = y[t]
            loss_t = F.cross_entropy(logits, y_t, reduction="mean")
            total_loss += loss_t.item()
            if t >= cfg.S + cfg.L + 1:
                correct_out += (logits.argmax(-1) == y_t).float().mean().item()
                out_steps += 1

            # ---- fast tag update (RFLO-style) ----
            phi_prime = 1.0 - h * h
            e_rec = cfg.alpha * e_rec + phi_prime.unsqueeze(2) * h_prev.unsqueeze(1)
            e_in  = cfg.alpha * e_in  + phi_prime.unsqueeze(2) * x_oh[t].unsqueeze(1)

            # ---- local surprise signal ----
            surprise = (h - h_hat).abs()             # [B, N]
            # Update the local predictor toward the previous activation, to avoid self-referential surprise.
            h_hat = cfg.gamma * h_hat + (1.0 - cfg.gamma) * h_prev

            # ---- capture event ----
            if no_gate:
                c = torch.ones_like(surprise)
            else:
                c = (surprise > theta).float()
                # Homeostatic threshold update (per-neuron per-batch element).
                theta = theta + cfg.eta_theta * (c - cfg.rho)
            captures_fired += c.mean().item()
            captures_denom += 1

            # ---- slow commit trace integrates tag at captures ----
            s_rec = cfg.beta * s_rec + c.unsqueeze(2) * e_rec
            s_in  = cfg.beta * s_in  + c.unsqueeze(2) * e_in

            # ---- sparse updates gated by capture ----
            err = probs - F.one_hot(y_t, K).float()     # [B, K]
            M = err @ B_fb.T                             # [B, N]
            gate = (c * M).unsqueeze(2)                  # [B, N, 1]
            dWrec = dWrec + cfg.lr_local * (gate * s_rec).mean(dim=0)
            dWin  = dWin  + cfg.lr_local * (gate * s_in ).mean(dim=0)
            # Output layer trained normally (true-gradient local update).
            dWout = dWout + cfg.lr_local * (err.unsqueeze(2) * h.unsqueeze(1)).mean(dim=0)
            dbout = dbout + cfg.lr_local * err.mean(dim=0)

        # Apply updates once per sequence.
        p.W_rec = p.W_rec - dWrec
        p.W_in  = p.W_in  - dWin
        p.W_out = p.W_out - dWout
        p.b_out = p.b_out - dbout

        if (it + 1) % cfg.log_every == 0 or it == 0:
            mean_loss = total_loss / T
            acc = correct_out / max(out_steps, 1)
            cap_rate = captures_fired / max(captures_denom, 1)
            losses.append((it + 1, mean_loss))
            accs.append((it + 1, acc))
            times.append((it + 1, time.time() - t_start))
            capture_rate_log.append((it + 1, cap_rate))

    return {"losses": losses, "accs": accs, "times": times,
            "capture_rate": capture_rate_log,
            "final_loss": losses[-1][1], "final_acc": accs[-1][1],
            "wall_clock": time.time() - t_start}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

@dataclass
class Cfg:
    N: int = 64
    K: int = 8
    S: int = 3
    L: int = 5
    batch_size: int = 64
    num_iters: int = 1000
    log_every: int = 50
    lr: float = 1e-3               # BPTT optimizer LR (Adam)
    lr_local: float = 5e-3         # per-step SGD rate for local rules
    alpha: float = 0.8
    beta: float = 0.95
    gamma: float = 0.9
    rho: float = 0.1
    eta_theta: float = 0.01
    seed: int = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="step3_results.json")
    parser.add_argument("--fig", default="step3_learning_curves.png")
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Cfg(num_iters=args.num_iters, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"config = {cfg}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Shared initialization — each rule starts from its own independent copy.
    p0 = RNNParams.init(cfg.N, cfg.K, device)

    results = {}
    rules = [
        ("bptt",         lambda p: train_bptt(p, cfg, device)),
        ("rflo",         lambda p: train_rflo(p, cfg, device)),
        ("stcprop",      lambda p: train_stcprop(p, cfg, device, no_gate=False)),
        ("stcprop_nogate", lambda p: train_stcprop(p, cfg, device, no_gate=True)),
    ]
    for name, fn in rules:
        print(f"--- training {name} ---")
        # Same initial parameters for fair comparison.
        p = p0.clone()
        torch.manual_seed(cfg.seed)   # reset RNG so batches are reproducible
        r = fn(p)
        print(f"  final loss={r['final_loss']:.4f}  final_copy_acc={r['final_acc']:.4f}  wall={r['wall_clock']:.1f}s")
        results[name] = r

    # Save JSON
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2)
    print(f"wrote {args.out}")

    # Plot learning curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, r in results.items():
            xs = [it for it, _ in r["losses"]]
            ys = [v  for _, v  in r["losses"]]
            axes[0].plot(xs, ys, label=name)
            xs = [it for it, _ in r["accs"]]
            ys = [v  for _, v  in r["accs"]]
            axes[1].plot(xs, ys, label=name)
        axes[0].set_xlabel("iteration"); axes[0].set_ylabel("mean cross-entropy loss"); axes[0].legend(); axes[0].set_title("training loss")
        axes[1].set_xlabel("iteration"); axes[1].set_ylabel("copy-window accuracy"); axes[1].legend(); axes[1].set_title("copy accuracy")
        fig.tight_layout()
        fig.savefig(args.fig, dpi=140)
        print(f"wrote {args.fig}")
    except Exception as e:
        print(f"plot failed: {e}")

    # Print summary table
    print("\n=== SUMMARY ===")
    print(f"{'rule':<20}{'final loss':>14}{'final acc':>14}{'wall (s)':>14}")
    for name, r in results.items():
        print(f"{name:<20}{r['final_loss']:>14.4f}{r['final_acc']:>14.4f}{r['wall_clock']:>14.2f}")


if __name__ == "__main__":
    main()
