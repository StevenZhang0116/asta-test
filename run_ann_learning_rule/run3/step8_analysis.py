import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"

# Load results
with open(os.path.join(WORK_DIR, "full_training_results.json"), "r") as f:
    results = json.load(f)

methods = {
    "bptt": "BPTT (gold standard)",
    "psc_nogate": "PSC-NoGate (ours)",
    "psc_osc": "PSC-Osc (ours)",
    "fa": "FA (baseline)"
}
colors = {
    "bptt": "blue",
    "psc_nogate": "green",
    "psc_osc": "orange",
    "fa": "red"
}

# Reconstruct full histories from sampled data
LOG_INTERVAL = 100
iters = list(range(LOG_INTERVAL, 5001, LOG_INTERVAL))

# ============================================================
# FIGURE 1: Learning curves (MSE)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for key, label in methods.items():
    mse_h = results[key]["mse_history"]
    x = iters[:len(mse_h)]
    ax.plot(x, mse_h, label=label, color=colors[key], linewidth=2)
ax.set_xlabel("Training Iterations", fontsize=12)
ax.set_ylabel("MSE", fontsize=12)
ax.set_title("Learning Curves: MSE", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.02, 0.65])

# Zoomed version (excluding BPTT for clarity)
ax2 = axes[1]
for key, label in methods.items():
    if key == "bptt": continue
    mse_h = results[key]["mse_history"]
    x = iters[:len(mse_h)]
    ax2.plot(x, mse_h, label=label, color=colors[key], linewidth=2)
ax2.set_xlabel("Training Iterations", fontsize=12)
ax2.set_ylabel("MSE", fontsize=12)
ax2.set_title("Learning Curves: MSE (Biologically-Plausible Methods)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "learning_curves_mse.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: learning_curves_mse.png")

# ============================================================
# FIGURE 2: Accuracy curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for key, label in methods.items():
    acc_h = results[key]["acc_history"]
    x = iters[:len(acc_h)]
    ax.plot(x, [a*100 for a in acc_h], label=label, color=colors[key], linewidth=2)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
ax.set_xlabel("Training Iterations", fontsize=12)
ax.set_ylabel("Bit Accuracy (%)", fontsize=12)
ax.set_title("Learning Curves: Bit Accuracy", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
for key, label in methods.items():
    if key == "bptt": continue
    acc_h = results[key]["acc_history"]
    x = iters[:len(acc_h)]
    ax2.plot(x, [a*100 for a in acc_h], label=label, color=colors[key], linewidth=2)
ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
ax2.set_xlabel("Training Iterations", fontsize=12)
ax2.set_ylabel("Bit Accuracy (%)", fontsize=12)
ax2.set_title("Accuracy (Biologically-Plausible Methods)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "learning_curves_acc.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: learning_curves_acc.png")

# ============================================================
# FIGURE 3: Final comparison bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

method_names = ["BPTT", "PSC-NoGate", "PSC-Osc", "FA"]
keys_order = ["bptt", "psc_nogate", "psc_osc", "fa"]
bar_colors = [colors[k] for k in keys_order]
final_mse = [results[k]["final_mse"] for k in keys_order]
final_acc = [results[k]["final_acc"]*100 for k in keys_order]

ax = axes[0]
bars = ax.bar(method_names, final_mse, color=bar_colors, alpha=0.8, edgecolor="black")
ax.set_ylabel("Final MSE", fontsize=12)
ax.set_title("Final MSE Comparison (5000 iters)", fontsize=13)
for bar, val in zip(bars, final_mse):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

ax2 = axes[1]
bars2 = ax2.bar(method_names, final_acc, color=bar_colors, alpha=0.8, edgecolor="black")
ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.7, label="Random")
ax2.set_ylabel("Bit Accuracy (%)", fontsize=12)
ax2.set_title("Final Bit Accuracy (5000 iters)", fontsize=13)
for bar, val in zip(bars2, final_acc):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax2.set_ylim([0, 110])
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, "final_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: final_comparison.png")

# ============================================================
# CONVERGENCE SPEED ANALYSIS
# ============================================================
print("")
print("=== Convergence Speed Analysis ===")
thresholds = [0.6, 0.7, 0.75, 0.8, 0.9]
convergence = {}
for key in keys_order:
    acc_h = results[key]["acc_history"]
    convergence[key] = {}
    for thresh in thresholds:
        reached = None
        for i, acc in enumerate(acc_h):
            if acc >= thresh:
                reached = (i+1) * LOG_INTERVAL
                break
        convergence[key][thresh] = reached

print("  Method         | 60%   | 70%   | 75%   | 80%   | 90%")
print("  " + "-"*65)
for key, name in zip(keys_order, method_names):
    row = "  {:15s}|".format(name)
    for t in thresholds:
        v = convergence[key][t]
        row += " {:5s} |".format(str(v) if v else "N/A")
    print(row)

# ============================================================
# PRINT SUMMARY TABLE
# ============================================================
print("")
print("=== Final Results Summary ===")
print("  {:20s} | {:10s} | {:10s}".format("Method", "Final MSE", "Final Acc"))
print("  " + "-"*47)
for key, name in zip(keys_order, method_names):
    mse = results[key]["final_mse"]
    acc = results[key]["final_acc"]
    print("  {:20s} | {:10.4f} | {:10.2%}".format(name, mse, acc))

# Save analysis to JSON
analysis = {
    "final_comparison": {
        k: {"final_mse": results[k]["final_mse"], "final_acc": results[k]["final_acc"]}
        for k in keys_order
    },
    "convergence_speed": {
        k: {str(t): v for t, v in convergence[k].items()}
        for k in keys_order
    }
}
with open(os.path.join(WORK_DIR, "analysis_results.json"), "w") as f:
    json.dump(analysis, f, indent=2)
print("")
print("Analysis saved to analysis_results.json")
print("=== Step 8: Analysis COMPLETE ===")