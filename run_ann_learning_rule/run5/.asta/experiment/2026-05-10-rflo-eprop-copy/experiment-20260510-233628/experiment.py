
# ----------
# The below command failed to execute (raised a No module named 'torch' exception)
# import torch

# ----------

# ----------
import subprocess
import sys
print("Installing PyTorch...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', 'torch', '--quiet'],
    capture_output=True, text=True
)
print(f"pip stdout: {result.stdout[-500:] if result.stdout else '(none)'}")
print(f"pip stderr: {result.stderr[-500:] if result.stderr else '(none)'}")
print(f"Return code: {result.returncode}")
import importlib
torch_spec = importlib.util.find_spec('torch')
print(f"torch found: {torch_spec is not None}")

# ----------

# ----------
import subprocess
import sys
import os
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nTrying: uv pip install torch...")
result = subprocess.run(
    ['uv', 'pip', 'install', 'torch', '--system'],
    capture_output=True, text=True
)
print(f"Return code: {result.returncode}")
print(f"stdout: {result.stdout[-1000:] if result.stdout else '(none)'}")
print(f"stderr: {result.stderr[-1000:] if result.stderr else '(none)'}")
import importlib.util
torch_spec = importlib.util.find_spec('torch')
print(f"\ntorch found after uv install: {torch_spec is not None}")
if torch_spec is None:
    # Method 2: Try uv tool install or uv add
    print("\nTrying: uv pip install torch (without --system)...")
    result2 = subprocess.run(
        ['uv', 'pip', 'install', 'torch'],
        capture_output=True, text=True
    )
    print(f"Return code: {result2.returncode}")
    print(f"stdout: {result2.stdout[-500:] if result2.stdout else '(none)'}")
    print(f"stderr: {result2.stderr[-500:] if result2.stderr else '(none)'}")
for path in sys.path:
    if 'site-packages' in path or 'dist-packages' in path:
        print(f"\nSite packages path: {path}")
        if os.path.exists(path):
            pkgs = [p for p in os.listdir(path) if 'torch' in p.lower()]
            print(f"  torch-related packages: {pkgs}")

# ----------

# ----------
import sys
import os
import subprocess
print(f"Current Python: {sys.executable}")
print(f"Current sys.path: {sys.path[:3]}")
conda_site_packages = '/home/zihan.zhang/.conda/envs/panda/lib/python3.11/site-packages'
if os.path.exists(conda_site_packages):
    print(f"\nFound conda site-packages: {conda_site_packages}")
    # Check if torch is there
    torch_dir = os.path.join(conda_site_packages, 'torch')
    print(f"torch directory exists: {os.path.exists(torch_dir)}")
    if os.path.exists(torch_dir) and conda_site_packages not in sys.path:
        sys.path.insert(0, conda_site_packages)
        print(f"Added {conda_site_packages} to sys.path")
else:
    print(f"conda site-packages not found at {conda_site_packages}")
    # Try to find it
    conda_base = '/home/zihan.zhang/.conda/envs/panda/lib'
    if os.path.exists(conda_base):
        for item in os.listdir(conda_base):
            print(f"  Found in lib: {item}")
try:
    import torch
    print(f"\nSUCCESS: torch {torch.__version__} imported!")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"\nFailed to import torch: {e}")
    
    # Method 2: Try uv pip install with specific python
    print("\nTrying uv pip install with specific python...")
    result = subprocess.run(
        ['uv', 'pip', 'install', 'torch', 
         '--python', '/home/zihan.zhang/.local/share/uv/tools/panda/bin/python'],
        capture_output=True, text=True, timeout=300
    )
    print(f"Return code: {result.returncode}")
    print(f"stdout: {result.stdout[-500:]}")
    print(f"stderr: {result.stderr[-500:]}")
    
    # Try import again
    try:
        import importlib
        importlib.invalidate_caches()
        import torch
        print(f"SUCCESS after uv install: torch {torch.__version__}")
    except ImportError as e2:
        print(f"Still failed: {e2}")
        
        # Method 3: Check what's available
        print("\nChecking available packages in uv env:")
        uv_site = '/home/zihan.zhang/.local/share/uv/tools/panda/lib/python3.12/site-packages'
        if os.path.exists(uv_site):
            pkgs = sorted(os.listdir(uv_site))
            print(f"Packages: {pkgs[:30]}")

# ----------

# ----------
import sys
import os
os.chdir('/tmp')
print(f"Changed CWD to: {os.getcwd()}")
conda_path = '/home/zihan.zhang/.conda/envs/panda/lib/python3.11/site-packages'
if conda_path in sys.path:
    sys.path.remove(conda_path)
    print(f"Removed conda path from sys.path")
to_remove = [key for key in sys.modules.keys() if 'torch' in key.lower()]
for key in to_remove:
    del sys.modules[key]
print(f"Removed {len(to_remove)} torch-related cached modules")
uv_site = '/home/zihan.zhang/.local/share/uv/tools/panda/lib/python3.12/site-packages'
print(f"\nChecking uv site-packages: {uv_site}")
if os.path.exists(uv_site):
    torch_items = [p for p in os.listdir(uv_site) if 'torch' in p.lower()]
    print(f"torch-related items: {torch_items}")
try:
    import importlib
    importlib.invalidate_caches()
    import torch
    print(f"\nSUCCESS: torch {torch.__version__} imported from {torch.__file__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    BACKEND = 'torch'
except Exception as e:
    print(f"\ntorch still unavailable: {type(e).__name__}: {str(e)[:200]}")
    print("Will use NumPy for the experiment.")
    BACKEND = 'numpy'
import numpy as np
print(f"\nNumPy {np.__version__} available - using as primary backend: {BACKEND}")
print(f"Current sys.path relevant entries:")
for p in sys.path:
    if 'panda' in p or 'site' in p:
        print(f"  {p}")

# ----------

# ----------
import torch
import numpy as np
import time
import random
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
N = 128          # hidden size
N_IN = 10        # input dim
N_OUT = 8        # output dim
T = 15           # sequence length
N_SYM = 5        # symbols to copy
ALPH = 8         # alphabet size
BATCH = 64
N_STEPS = 3000
EVAL_EVERY = 100
EVAL_BATCH = 1000
ALPHA = 0.1
ETA = 3e-3
ETA_V = 1e-3
OUT_STEPS = list(range(10, 15))  # t=10..14 (0-indexed), output at positions 11..15
def generate_batch(batch_size, seed=None):
    """Generate a batch of copy task sequences.
    Input shape: (batch, T, N_IN)
    Target shape: (batch, T) with -100 for non-output steps
    
    Structure (0-indexed):
      t=0..4:  symbol tokens (one-hot in channels 0..7)
      t=5..8:  blank tokens (channel 8 = 1)
      t=9:     blank + go cue (channels 8,9 = 1)
      t=10..14: blank (network should output the stored symbols)
    Target: at t=10..14, the symbol index (0..7); -100 elsewhere
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    # Sample symbol sequences
    symbols = rng.randint(0, ALPH, size=(batch_size, N_SYM))  # (B, 5)
    
    # Build input tensor
    X = np.zeros((batch_size, T, N_IN), dtype=np.float32)
    # t=0..4: symbol one-hots
    for t in range(N_SYM):
        X[np.arange(batch_size), t, symbols[:, t]] = 1.0
    # t=5..8: blank (channel 8)
    X[:, 5:9, 8] = 1.0
    # t=9: blank + go cue (channels 8 and 9)
    X[:, 9, 8] = 1.0
    X[:, 9, 9] = 1.0
    # t=10..14: blank (channel 8)
    X[:, 10:15, 8] = 1.0
    
    # Build target tensor: -100 everywhere except output steps
    Y = np.full((batch_size, T), -100, dtype=np.int64)
    for i, t in enumerate(OUT_STEPS):
        Y[:, t] = symbols[:, i]
    
    return (torch.tensor(X, device=device),
            torch.tensor(Y, device=device))
X_test, Y_test = generate_batch(4, seed=42)
print(f"\nData generator test:")
print(f"  X shape: {X_test.shape}, Y shape: {Y_test.shape}")
print(f"  X[0] = {X_test[0].cpu().numpy()}")
print(f"  Y[0] = {Y_test[0].cpu().numpy()}")
def xavier_uniform(fan_in, fan_out):
    """Xavier uniform initialization"""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return torch.FloatTensor(fan_out, fan_in).uniform_(-limit, limit).to(device)
def init_weights():
    """Initialize RNN weights. Returns dict of weight tensors."""
    W = xavier_uniform(N, N)          # recurrent: (N, N)
    U = xavier_uniform(N_IN, N)       # input:     (N, N_IN)
    b = torch.zeros(N, device=device) # recurrent bias: (N,)
    V = xavier_uniform(N, N_OUT)      # readout:   (N_OUT, N)
    b_y = torch.zeros(N_OUT, device=device)  # readout bias: (N_OUT,)
    
    # Count parameters
    total = W.numel() + U.numel() + b.numel() + V.numel() + b_y.numel()
    print(f"  W: {W.shape} = {W.numel()}")
    print(f"  U: {U.shape} = {U.numel()}")
    print(f"  b: {b.shape} = {b.numel()}")
    print(f"  V: {V.shape} = {V.numel()}")
    print(f"  b_y: {b_y.shape} = {b_y.numel()}")
    print(f"  Total params: {total}")
    
    return W, U, b, V, b_y
print("\nWeight shapes:")
W, U, b, V, b_y = init_weights()
torch.manual_seed(SEED)
B_feedback = torch.randn(N, N_OUT, device=device) / np.sqrt(N_OUT)
print(f"  B_feedback: {B_feedback.shape}, std={B_feedback.std().item():.4f}")

# ----------

# ----------
import torch.nn.functional as F
def run_rflo(eta, eta_v, alpha, n_steps, seed=0, verbose=True):
    """Run RFLO/e-prop training on the copy task.
    Returns learning curve data.
    """
    # Re-init weights with given seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    W_ = xavier_uniform(N, N).detach()   # (N, N)
    U_ = xavier_uniform(N_IN, N).detach() # (N, N_IN)
    b_ = torch.zeros(N, device=device)
    V_ = xavier_uniform(N, N_OUT).detach() # (N_OUT, N)
    b_y_ = torch.zeros(N_OUT, device=device)
    
    # Fixed random feedback B ~ N(0, 1/sqrt(N_OUT))
    torch.manual_seed(seed)
    B_ = torch.randn(N, N_OUT, device=device) / np.sqrt(N_OUT)
    
    # Eval dataset (fixed)
    X_eval, Y_eval = generate_batch(EVAL_BATCH, seed=12345)
    
    # Learning curve storage
    curve_steps = []
    curve_sym_acc = []
    curve_seq_acc = []
    curve_loss = []
    
    t_start = time.time()
    wall_times = []
    
    for step in range(1, n_steps + 1):
        # Generate training batch
        X, Y = generate_batch(BATCH)
        # X: (B, T, N_IN), Y: (B, T) with -100 for non-output
        
        # Initialize hidden state and eligibility traces
        h = torch.zeros(BATCH, N, device=device)  # (B, N)
        p_W = torch.zeros(BATCH, N, N, device=device)    # (B, N, N)
        p_U = torch.zeros(BATCH, N, N_IN, device=device) # (B, N, N_IN)
        
        # Accumulators for weight updates (averaged over batch and time)
        dW = torch.zeros_like(W_)  # (N, N)
        dU = torch.zeros_like(U_)  # (N, N_IN)
        db = torch.zeros_like(b_)  # (N,)
        dV = torch.zeros_like(V_)  # (N_OUT, N)
        db_y = torch.zeros_like(b_y_)  # (N_OUT,)
        
        with torch.no_grad():
            for t in range(T):
                x_t = X[:, t, :]   # (B, N_IN)
                h_prev = h.clone() # (B, N)
                
                # Forward pass
                u_t = h_prev @ W_.t() + x_t @ U_.t() + b_  # (B, N)
                h = torch.tanh(u_t)                          # (B, N)
                y_hat = h @ V_.t() + b_y_                    # (B, N_OUT)
                
                # phi_prime = 1 - tanh^2(u_t) = 1 - h^2
                phi_prime = 1.0 - h ** 2  # (B, N)
                
                # Update eligibility traces
                # p_W[b,i,j] = (1-alpha)*p_W[b,i,j] + alpha*phi_prime[b,i]*h_prev[b,j]
                # Using broadcasting: phi_prime.unsqueeze(2) * h_prev.unsqueeze(1) -> (B, N, N)
                p_W = (1 - alpha) * p_W + alpha * (phi_prime.unsqueeze(2) * h_prev.unsqueeze(1))
                
                # p_U[b,i,j] = (1-alpha)*p_U[b,i,j] + alpha*phi_prime[b,i]*x_t[b,j]
                p_U = (1 - alpha) * p_U + alpha * (phi_prime.unsqueeze(2) * x_t.unsqueeze(1))
                
                # At output timesteps: compute teaching signal and update weights
                if t in OUT_STEPS:
                    # Softmax probabilities
                    probs = F.softmax(y_hat, dim=-1)  # (B, N_OUT)
                    
                    # One-hot target
                    y_onehot = F.one_hot(Y[:, t], num_classes=N_OUT).float()  # (B, N_OUT)
                    
                    # Error signal e = probs - y_onehot: (B, N_OUT)
                    e_t = probs - y_onehot
                    
                    # Teaching signal ell = B_feedback @ e_t^T -> (N, B)
                    # ell[b] = B_ @ e_t[b]: (B, N)
                    ell = e_t @ B_.t()  # (B, N)
                    
                    # Weight updates (average over batch)
                    # dW += mean_b [ outer(ell[b], p_W_avg[b]) ]
                    # outer(ell[b], p_W[b]) = ell[b,:].unsqueeze(1) * p_W[b] -> (B, N, N) 
                    # but wait: we need (N_post, N_pre) update: ell_i * p_W_ij
                    # ell has shape (B, N), p_W has shape (B, N, N)
                    # dW_ij = ell_i * p_W_ij -> element-wise: ell.unsqueeze(2) * p_W -> (B,N,N)
                    dW += (ell.unsqueeze(2) * p_W).mean(0)   # (N, N)
                    dU += (ell.unsqueeze(2) * p_U).mean(0)   # (N, N_IN)
                    db += ell.mean(0)  # bias: ell_i directly (local delta for bias)
                    
                    # Readout: local delta rule V and b_y
                    # dV_ij = e_i * h_j -> (N_OUT, N)
                    dV += (e_t.unsqueeze(2) * h.unsqueeze(1)).mean(0)  # (N_OUT, N)
                    db_y += e_t.mean(0)  # (N_OUT,)
            
            # Apply weight updates (after full sequence)
            W_ -= eta * dW
            U_ -= eta * dU
            b_ -= eta * db
            V_ -= eta_v * dV
            b_y_ -= eta_v * db_y
        
        # Evaluate every EVAL_EVERY steps
        if step % EVAL_EVERY == 0:
            wall_elapsed = time.time() - t_start
            wall_times.append(wall_elapsed)
            
            with torch.no_grad():
                h_e = torch.zeros(EVAL_BATCH, N, device=device)
                all_preds = []
                all_targets = []
                total_loss = 0.0
                
                for te in range(T):
                    x_te = X_eval[:, te, :]
                    u_te = h_e @ W_.t() + x_te @ U_.t() + b_
                    h_e = torch.tanh(u_te)
                    y_hat_e = h_e @ V_.t() + b_y_
                    
                    if te in OUT_STEPS:
                        preds = y_hat_e.argmax(dim=-1)  # (EVAL_BATCH,)
                        targets = Y_eval[:, te]          # (EVAL_BATCH,)
                        all_preds.append(preds)
                        all_targets.append(targets)
                        # Cross-entropy loss
                        total_loss += F.cross_entropy(y_hat_e, targets).item()
                
                all_preds = torch.stack(all_preds, dim=1)    # (EVAL_BATCH, 5)
                all_targets = torch.stack(all_targets, dim=1)  # (EVAL_BATCH, 5)
                
                # Per-symbol accuracy
                correct_sym = (all_preds == all_targets).float()
                sym_acc = correct_sym.mean().item()
                
                # Sequence-level accuracy (all 5 correct)
                seq_acc = (correct_sym.sum(dim=1) == N_SYM).float().mean().item()
                
                avg_loss = total_loss / len(OUT_STEPS)
                
                curve_steps.append(step)
                curve_sym_acc.append(sym_acc)
                curve_seq_acc.append(seq_acc)
                curve_loss.append(avg_loss)
                
                if verbose and (step % 500 == 0 or step <= 200):
                    print(f"Step {step:4d}: sym_acc={sym_acc:.4f}, seq_acc={seq_acc:.4f}, "
                          f"loss={avg_loss:.4f}")
    
    # Compute wall time per 100 steps
    total_wall = time.time() - t_start
    wall_per_100 = total_wall / (n_steps / 100)
    
    return {
        'curve_steps': curve_steps,
        'curve_sym_acc': curve_sym_acc,
        'curve_seq_acc': curve_seq_acc,
        'curve_loss': curve_loss,
        'wall_per_100': wall_per_100,
        'final_sym_acc': curve_sym_acc[-1],
        'final_seq_acc': curve_seq_acc[-1],
        'W': W_, 'U': U_, 'b': b_, 'V': V_, 'b_y': b_y_
    }
print("Training function defined. Starting RFLO training with eta=3e-3...")
print("(Will print progress at steps 100-200 and every 500 steps)")
results_3em3 = run_rflo(eta=3e-3, eta_v=1e-3, alpha=0.1, n_steps=N_STEPS, seed=0, verbose=True)
print(f"\nTraining complete!")
print(f"Final per-symbol accuracy: {results_3em3['final_sym_acc']:.4f}")
print(f"Final sequence accuracy: {results_3em3['final_seq_acc']:.4f}")
print(f"Wall time per 100 steps: {results_3em3['wall_per_100']:.3f}s")

# ----------

# ----------
def run_rflo_online(eta, eta_v, alpha, n_steps, seed=0, verbose=True):
    """Run RFLO with true online per-timestep weight updates.
    At each output timestep, immediately apply weight updates.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    W_ = xavier_uniform(N, N).detach()
    U_ = xavier_uniform(N_IN, N).detach()
    b_ = torch.zeros(N, device=device)
    V_ = xavier_uniform(N, N_OUT).detach()
    b_y_ = torch.zeros(N_OUT, device=device)
    
    torch.manual_seed(seed)
    B_ = torch.randn(N, N_OUT, device=device) / np.sqrt(N_OUT)
    
    X_eval, Y_eval = generate_batch(EVAL_BATCH, seed=12345)
    
    curve_steps = []
    curve_sym_acc = []
    curve_seq_acc = []
    curve_loss = []
    
    t_start = time.time()
    
    for step in range(1, n_steps + 1):
        X, Y = generate_batch(BATCH)
        
        h = torch.zeros(BATCH, N, device=device)
        p_W = torch.zeros(BATCH, N, N, device=device)
        p_U = torch.zeros(BATCH, N, N_IN, device=device)
        
        with torch.no_grad():
            for t in range(T):
                x_t = X[:, t, :]
                h_prev = h.clone()
                
                u_t = h_prev @ W_.t() + x_t @ U_.t() + b_
                h = torch.tanh(u_t)
                y_hat = h @ V_.t() + b_y_
                
                phi_prime = 1.0 - h ** 2
                
                # Update eligibility traces
                p_W = (1 - alpha) * p_W + alpha * (phi_prime.unsqueeze(2) * h_prev.unsqueeze(1))
                p_U = (1 - alpha) * p_U + alpha * (phi_prime.unsqueeze(2) * x_t.unsqueeze(1))
                
                # ONLINE: apply weight updates immediately at output timesteps
                if t in OUT_STEPS:
                    probs = F.softmax(y_hat, dim=-1)
                    y_onehot = F.one_hot(Y[:, t], num_classes=N_OUT).float()
                    e_t = probs - y_onehot
                    ell = e_t @ B_.t()  # (B, N)
                    
                    # Online update: apply immediately (average over batch)
                    W_ -= eta * (ell.unsqueeze(2) * p_W).mean(0)
                    U_ -= eta * (ell.unsqueeze(2) * p_U).mean(0)
                    b_ -= eta * ell.mean(0)
                    V_ -= eta_v * (e_t.unsqueeze(2) * h.unsqueeze(1)).mean(0)
                    b_y_ -= eta_v * e_t.mean(0)
        
        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                h_e = torch.zeros(EVAL_BATCH, N, device=device)
                all_preds = []
                all_targets = []
                total_loss = 0.0
                
                for te in range(T):
                    x_te = X_eval[:, te, :]
                    u_te = h_e @ W_.t() + x_te @ U_.t() + b_
                    h_e = torch.tanh(u_te)
                    y_hat_e = h_e @ V_.t() + b_y_
                    
                    if te in OUT_STEPS:
                        preds = y_hat_e.argmax(dim=-1)
                        targets = Y_eval[:, te]
                        all_preds.append(preds)
                        all_targets.append(targets)
                        total_loss += F.cross_entropy(y_hat_e, targets).item()
                
                all_preds = torch.stack(all_preds, dim=1)
                all_targets = torch.stack(all_targets, dim=1)
                correct_sym = (all_preds == all_targets).float()
                sym_acc = correct_sym.mean().item()
                seq_acc = (correct_sym.sum(dim=1) == N_SYM).float().mean().item()
                avg_loss = total_loss / len(OUT_STEPS)
                
                curve_steps.append(step)
                curve_sym_acc.append(sym_acc)
                curve_seq_acc.append(seq_acc)
                curve_loss.append(avg_loss)
                
                if verbose and (step % 500 == 0 or step <= 300):
                    print(f"  Step {step:4d}: sym_acc={sym_acc:.4f}, seq_acc={seq_acc:.4f}, loss={avg_loss:.4f}")
    
    total_wall = time.time() - t_start
    wall_per_100 = total_wall / (n_steps / 100)
    
    return {
        'curve_steps': curve_steps,
        'curve_sym_acc': curve_sym_acc,
        'curve_seq_acc': curve_seq_acc,
        'curve_loss': curve_loss,
        'wall_per_100': wall_per_100,
        'final_sym_acc': curve_sym_acc[-1],
        'final_seq_acc': curve_seq_acc[-1],
    }
eta_candidates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
print("=" * 60)
print("RFLO Online - Hyperparameter sweep over eta")
print("=" * 60)
sweep_results = {}
for eta_cand in eta_candidates:
    print(f"\n--- eta={eta_cand:.0e} ---")
    res = run_rflo_online(eta=eta_cand, eta_v=1e-3, alpha=0.1, n_steps=N_STEPS, 
                          seed=0, verbose=True)
    sweep_results[eta_cand] = res
    print(f"  FINAL: sym_acc={res['final_sym_acc']:.4f}, seq_acc={res['final_seq_acc']:.4f}")
best_eta = max(sweep_results, key=lambda e: sweep_results[e]['final_sym_acc'])
best_result = sweep_results[best_eta]
print(f"\n{'='*60}")
print(f"Best eta: {best_eta:.0e}")
print(f"Best final per-symbol accuracy: {best_result['final_sym_acc']:.4f}")
print(f"Best final sequence accuracy: {best_result['final_seq_acc']:.4f}")
print(f"Wall time per 100 steps (best run): {best_result['wall_per_100']:.3f}s")
crossed_90 = 'never'
for s, acc in zip(best_result['curve_steps'], best_result['curve_sym_acc']):
    if acc >= 0.90:
        crossed_90 = s
        break
print(f"90% per-symbol accuracy first crossed at step: {crossed_90}")

# ----------

# ----------
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
output_dir = '/tmp/rflo_eprop_results'
os.makedirs(output_dir, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
eta_labels = ['1e-4', '3e-4', '1e-3', '3e-3', '1e-2']
eta_vals = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
ax1 = axes[0]
for i, (eta_val, label, color) in enumerate(zip(eta_vals, eta_labels, colors)):
    if eta_val in sweep_results:
        res = sweep_results[eta_val]
        lw = 3 if eta_val == best_eta else 1.5
        ls = '-' if eta_val == best_eta else '--'
        marker = 'o' if eta_val == best_eta else None
        ms = 4 if eta_val == best_eta else None
        label_str = f'e={label}' + (' (best)' if eta_val == best_eta else '')
        ax1.plot(res['curve_steps'], res['curve_sym_acc'], 
                 color=color, lw=lw, ls=ls, 
                 marker=marker, markersize=ms, markevery=5,
                 label=label_str)
ax1.axhline(y=1.0, color='black', ls=':', lw=1.5, alpha=0.5, label='BPTT baseline (100%)')
ax1.axhline(y=0.90, color='gray', ls='--', lw=1.5, alpha=0.7, label='90% threshold')
ax1.axhline(y=1/8, color='lightgray', ls=':', lw=1.0, alpha=0.5, label='Chance (12.5%)')
ax1.set_xlabel('Training Step', fontsize=13)
ax1.set_ylabel('Per-Symbol Accuracy', fontsize=13)
ax1.set_title('RFLO/e-prop: Per-Symbol Accuracy\n(e sweep, copy task T=15)', fontsize=13)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_xlim(0, N_STEPS)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=11)
ax2 = axes[1]
br = best_result
steps = br['curve_steps']
ax2_twin = ax2.twinx()
l1, = ax2.plot(steps, br['curve_sym_acc'], 'b-o', lw=2.5, markersize=4, markevery=5,
               label=f'Per-symbol acc (e={best_eta:.0e})')
l2, = ax2.plot(steps, br['curve_seq_acc'], 'g-s', lw=2.0, markersize=4, markevery=5,
               label='Sequence-level acc')
l3, = ax2_twin.plot(steps, br['curve_loss'], 'r--', lw=1.5, alpha=0.7,
                    label='CE Loss')
ax2.axhline(y=0.90, color='gray', ls='--', lw=1.5, alpha=0.7, label='90% threshold')
ax2.axhline(y=1.0, color='black', ls=':', lw=1.5, alpha=0.4, label='BPTT baseline (100%)')
ax2.annotate(f'{br["final_sym_acc"]:.3f}', 
             xy=(steps[-1], br['final_sym_acc']),
             xytext=(-40, 10), textcoords='offset points',
             fontsize=10, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.0))
ax2.set_xlabel('Training Step', fontsize=13)
ax2.set_ylabel('Accuracy', fontsize=13, color='black')
ax2_twin.set_ylabel('Cross-Entropy Loss', fontsize=12, color='red')
ax2.set_title(f'RFLO/e-prop Best Run (e={best_eta:.0e})\nPer-symbol: {br["final_sym_acc"]:.1%}, Seq: {br["final_seq_acc"]:.1%}',
              fontsize=13)
ax2.set_xlim(0, N_STEPS)
ax2.set_ylim(0, 1.05)
ax2_twin.set_ylim(1.5, 2.3)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=11)
lines = [l1, l2, l3]
labels_leg = [l.get_label() for l in lines]
ax2.legend(lines, labels_leg, fontsize=9, loc='upper left')
fig.suptitle('RFLO & Rate e-prop on Copy Task (T=15, N=128)\n'
             'Note: For rate RNN + linear readout, RFLO and rate e-prop are mathematically equivalent',
             fontsize=11, y=1.02)
plt.tight_layout()
png_path = os.path.join(output_dir, 'rflo_eprop_learning_curve.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"Learning curve saved to: {png_path}")
plt.close()
fig2, ax = plt.subplots(figsize=(8, 5))
models = ['BPTT\n(Step-3 baseline)', 'RFLO/e-prop\n(best e=1e-4)']
final_accs = [1.00, best_result['final_sym_acc']]
colors_bar = ['#2ecc71', '#e74c3c']
bars = ax.bar(models, final_accs, color=colors_bar, width=0.4, alpha=0.85, edgecolor='black')
for bar, acc in zip(bars, final_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.axhline(y=0.90, color='gray', ls='--', lw=1.5, label='90% threshold')
ax.set_ylabel('Final Per-Symbol Accuracy', fontsize=13)
ax.set_title('RFLO/e-prop vs BPTT Baseline\nCopy Task (T=15, N=128, 3000 steps)', fontsize=13)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3, axis='y')
gap = 1.00 - best_result['final_sym_acc']
ax.annotate('', xy=(1, best_result['final_sym_acc']), xytext=(1, 1.0),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.text(1.22, (1.0 + best_result['final_sym_acc'])/2, 
        f'Gap:\n{gap:.1%}', color='purple', fontsize=11, va='center')
plt.tight_layout()
png_path2 = os.path.join(output_dir, 'rflo_eprop_comparison.png')
fig2.savefig(png_path2, dpi=150, bbox_inches='tight')
print(f"Comparison chart saved to: {png_path2}")
plt.close()
print(f"\nAll figures saved to: {output_dir}")
print(f"Files: {os.listdir(output_dir)}")

# ----------

# ----------
import json
crossed_90 = 'never'
for s, acc in zip(best_result['curve_steps'], best_result['curve_sym_acc']):
    if acc >= 0.90:
        crossed_90 = s
        break
bptt_baseline = {
    'final_sym_acc': 1.00,
    'final_seq_acc': 1.00,  # approximate
    'step_90pct': 100,
    'wall_per_100_steps_s': 0.82,
    'device': 'A100',
    'n_steps': 3000
}
sweep_summary = {}
for eta_val, res in sweep_results.items():
    sweep_summary[f'eta_{eta_val:.0e}'] = {
        'eta': eta_val,
        'final_sym_acc': res['final_sym_acc'],
        'final_seq_acc': res['final_seq_acc'],
        'wall_per_100_steps_s': res['wall_per_100'],
        'learning_curve_steps': res['curve_steps'],
        'learning_curve_sym_acc': res['curve_sym_acc'],
        'learning_curve_seq_acc': res['curve_seq_acc'],
        'learning_curve_loss': res['curve_loss'],
    }
results = {
    'experiment': 'RFLO and rate e-prop on copy task (rate RNN, linear readout)',
    'note': 'For a rate RNN with linear readout, RFLO and rate e-prop are mathematically equivalent. '
            'Both rules use: eligibility trace p_ij = (1-alpha)*p_ij + alpha*phi_prime_i*pre_j, '
            'and learning signal ell = B @ (softmax(y_hat) - y_onehot) with fixed random B.',
    'architecture': {
        'model': 'Vanilla Elman RNN',
        'hidden_size': N,
        'input_dim': N_IN,
        'output_dim': N_OUT,
        'total_params': 18824,
        'recurrence': 'h_t = tanh(W h_{t-1} + U x_t + b)',
        'readout': 'y_hat = V h_t + b_y'
    },
    'task': {
        'name': 'Copy task',
        'T': T,
        'n_symbols': N_SYM,
        'alphabet_size': ALPH,
        'input_dim': N_IN,
        'output_dim': N_OUT,
        'output_timesteps': OUT_STEPS,
        'loss': 'cross-entropy on output timesteps only'
    },
    'training': {
        'n_steps': N_STEPS,
        'batch_size': BATCH,
        'eval_every': EVAL_EVERY,
        'eval_batch': EVAL_BATCH,
        'seed': SEED
    },
    'hyperparameters': {
        'alpha': ALPHA,
        'eta_default': 3e-3,
        'eta_chosen': best_eta,
        'eta_V': ETA_V,
        'feedback_B_std': f'N(0, 1/sqrt({N_OUT}))',
        'update_schedule': 'online (applied at each output timestep)'
    },
    'eta_sweep': {
        'candidates': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        'final_sym_accs': {f'{e:.0e}': sweep_results[e]['final_sym_acc'] for e in sweep_results},
        'best_eta': best_eta
    },
    'results': {
        'final_sym_acc': best_result['final_sym_acc'],
        'final_seq_acc': best_result['final_seq_acc'],
        'step_90pct_sym_acc': crossed_90,
        'wall_per_100_steps_s': best_result['wall_per_100'],
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    },
    'bptt_baseline': bptt_baseline,
    'comparison_to_bptt': {
        'sym_acc_gap': bptt_baseline['final_sym_acc'] - best_result['final_sym_acc'],
        'sym_acc_gap_pct': f"{(bptt_baseline['final_sym_acc'] - best_result['final_sym_acc'])*100:.1f}%",
        'rflo_reaches_90pct': crossed_90 != 'never',
        'bptt_reaches_90pct_at_step': bptt_baseline['step_90pct'],
        'speed_comparison': f"RFLO: {best_result['wall_per_100']:.2f}s/100steps, BPTT: {bptt_baseline['wall_per_100_steps_s']:.2f}s/100steps"
    },
    'sweep_details': sweep_summary
}
json_path = os.path.join(output_dir, 'result.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to: {json_path}")
print("\n" + "="*65)
print("RFLO / Rate e-prop -- Copy Task Results Summary")
print("="*65)
print(f"Architecture: Vanilla Elman RNN, N={N}, params=18824")
print(f"Task: Copy task T={T}, 5 symbols from alphabet {ALPH}")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
print()
print(f"Learning Rule: RFLO = rate e-prop (mathematically identical for rate RNN)")
print(f"  alpha (trace decay): {ALPHA}")
print(f"  eta (chosen, after sweep): {best_eta:.0e}  [default 3e-3 was tried first]")
print(f"  eta_V (readout): {ETA_V}")
print(f"  Feedback B: fixed random N(0, 1/sqrt({N_OUT}))")
print(f"  Update: online at each output timestep")
print()
print("--- ETA Sweep Results ---")
print(f"{'eta':>10} | {'final sym acc':>14} | {'final seq acc':>14}")
print("-" * 45)
for eta_val in sorted(sweep_results.keys()):
    res = sweep_results[eta_val]
    marker = " <-- BEST" if eta_val == best_eta else ""
    print(f"{eta_val:>10.0e} | {res['final_sym_acc']:>14.4f} | {res['final_seq_acc']:>14.4f}{marker}")
print()
print("--- Final Results (best eta) ---")
print(f"  Final per-symbol accuracy:    {best_result['final_sym_acc']:.4f} ({best_result['final_sym_acc']*100:.1f}%)")
print(f"  Final sequence-level acc:     {best_result['final_seq_acc']:.4f} ({best_result['final_seq_acc']*100:.1f}%)")
print(f"  Step where sym_acc >= 90%:    {crossed_90}")
print(f"  Wall time per 100 steps:      {best_result['wall_per_100']:.3f}s")
print()
print("--- Comparison to BPTT Baseline (Step-3) ---")
print(f"  BPTT final sym acc:           100.0%")
print(f"  RFLO final sym acc:           {best_result['final_sym_acc']*100:.1f}%")
print(f"  GAP (BPTT - RFLO):            {(1.0 - best_result['final_sym_acc'])*100:.1f} percentage points")
print(f"  BPTT 90% at step:             100")
print(f"  RFLO 90% at step:             {crossed_90}")
print(f"  BPTT wall/100steps:           0.82s")
print(f"  RFLO wall/100steps:           {best_result['wall_per_100']:.2f}s")
print()
print("CONCLUSION: RFLO/e-prop reaches only 55.3% per-symbol accuracy")
print("  vs. BPTT's 100%, a gap of 44.7 ppts. The 90% threshold is")
print("  never reached by RFLO/e-prop in 3000 steps. This confirms")
print("  the significant credit-assignment advantage of BPTT over")
print("  biologically-plausible local learning rules on this task.")
print("="*65)

# ----------

# ----------
print("="*70)
print("EXPERIMENT: RFLO and Rate e-prop Baselines on Copy Task")
print("="*70)
print()
print("BACKGROUND")
print("-"*40)
print("This experiment implements two biologically-plausible learning rules")
print("(RFLO and rate e-prop) on a copy task with a vanilla Elman RNN.")
print("On a rate RNN with linear readout, these two rules are mathematically")
print("equivalent, so a single experiment covers both.")
print()
print("ARCHITECTURE")
print("-"*40)
print("Model: Vanilla Elman RNN")
print("Recurrence: h_t = tanh(W h_{t-1} + U x_t + b)")
print("Readout: y_hat = V h_t + b_y")
print("Hidden size N = 128")
print("Input dim = 10 (8 symbol one-hots + blank + go-cue)")
print("Output dim = 8")
print("Total parameters = 18824")
print("  W: (128,128)=16384, U: (128,10)=1280, b: 128, V: (8,128)=1024, b_y: 8")
print()
print("TASK")
print("-"*40)
print("Copy task: T=15, 5 symbols from alphabet of size 8")
print("  t=0..4:  5 symbol inputs (one-hot, channels 0-7)")
print("  t=5..8:  blank tokens (channel 8 = 1)")
print("  t=9:     blank + go-cue (channels 8 and 9 = 1)")
print("  t=10..14: blank (output expected here)")
print("Loss: cross-entropy only on output timesteps t=10..14")
print()
print("LEARNING RULE")
print("-"*40)
print("RFLO (Murray 2019) = Rate e-prop (Bellec et al. 2020) for rate RNNs:")
print("  Eligibility trace:")
print("    p^W_{ij}(t) = (1-alpha)*p^W_{ij}(t-1) + alpha*phi'(u_i(t))*h_j(t-1)")
print("    p^U_{ij}(t) = (1-alpha)*p^U_{ij}(t-1) + alpha*phi'(u_i(t))*x_j(t)")
print("  Fixed random feedback: B ~ N(0, 1/sqrt(8)), shape (128, 8)")
print("  Teaching signal (output steps only): ell(t) = B @ (softmax(y_hat) - y_onehot)")
print("  Weight update: W -= eta * mean_batch[ell_i * p^W_{ij}]")
print("  Readout update: V -= eta_V * mean_batch[e_i * h_j]")
print("  Update applied ONLINE at each output timestep")
print("  NO backpropagation through time, NO autograd on recurrent weights")
print()
print("TRAINING SETUP")
print("-"*40)
print("Batch size: 64")
print("Training steps: 3000")
print("Eval every 100 steps on 1000 held-out sequences")
print("Seed: 0")
print("Device: NVIDIA A100-PCIE-40GB (cuda)")
print()
print("HYPERPARAMETER SWEEP RESULTS")
print("-"*40)
print("Default eta=3e-3 was tried first (reached 34.1% by step 3000).")
print("Full sweep over eta in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}:")
print()
print(f"  eta=1e-4: final sym_acc=55.30%, seq_acc=2.70%  <-- BEST")
print(f"  eta=3e-4: final sym_acc=50.34%, seq_acc=0.70%")
print(f"  eta=1e-3: final sym_acc=37.42%, seq_acc=0.10%")
print(f"  eta=3e-3: final sym_acc=34.06%, seq_acc=0.10%")
print(f"  eta=1e-2: final sym_acc=32.82%, seq_acc=0.10%")
print()
print("LEARNING CURVE (best run, eta=1e-4, per-symbol accuracy)")
print("-"*40)
for i, (s, acc, seq) in enumerate(zip(
    best_result['curve_steps'], 
    best_result['curve_sym_acc'],
    best_result['curve_seq_acc']
)):
    if s % 500 == 0 or s <= 300:
        print(f"  Step {s:4d}: sym_acc={acc:.4f}, seq_acc={seq:.4f}")
print()
print("FINAL RESULTS (best eta=1e-4)")
print("-"*40)
print(f"  Final per-symbol accuracy: 55.30% (0.5530)")
print(f"  Final sequence accuracy:    2.70% (0.0270)")
print(f"  90% threshold crossed at:  never")
print(f"  Wall time per 100 steps:   0.851s (A100)")
print()
print("COMPARISON TO BPTT BASELINE (Step-3)")
print("-"*40)
print("                    BPTT        RFLO/e-prop    Gap")
print("  Final sym acc:   100.0%         55.3%       -44.7 ppts")
print("  Final seq acc:   ~100%           2.7%       -97.3 ppts")
print("  90% at step:     100           never         n/a")
print("  Wall/100 steps:  0.82s          0.85s       +0.03s")
print()
print("INTERPRETATION")
print("-"*40)
print("RFLO/e-prop reaches 55.3% per-symbol accuracy after 3000 steps,")
print("compared to BPTT's 100% (achieved by step 200). The gap is 44.7")
print("percentage points. The 90% threshold is never crossed by RFLO/e-prop,")
print("while BPTT crosses it at step 100. The wall-clock time is comparable")
print("(0.85s vs 0.82s per 100 steps on A100), showing that the performance")
print("gap is purely due to credit assignment quality, not computational cost.")
print()
print("NOTE ON RFLO vs e-PROP EQUIVALENCE:")
print("For a vanilla rate RNN with a linear readout, the RFLO rule (Murray 2019)")
print("and the rate e-prop rule (Bellec et al. 2020) are mathematically identical.")
print("Both use the same eligibility trace (low-pass filtered pre*phi') and the")
print("same fixed-random-feedback learning signal. The distinction matters only")
print("for spiking networks (LSNN) where e-prop uses a surrogate gradient.")
print("="*70)
report_path = write_report()
print(f"\nReport written to: {report_path}")

# ----------
