import json
import os
import matplotlib.pyplot as plt

# =========================
# File Paths
# =========================
separate_path = "../results/averaged_top_n.json"
concat_path = "/workspace/projects/MixFL_concatenate/results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LLM ID Mapping
# =========================
LLM_IDS = {
    "1": "CodeBERT",
    "2": "CodeGen",
    "3": "CodeT5",
    "4": "GraphCodeBERT",
    "5": "InCoder",
    "6": "UniXcoder",
}

# =========================
# Load Data
# =========================
with open(separate_path, "r") as f:
    separate_data = json.load(f)

with open(concat_path, "r") as f:
    concat_data = json.load(f)

sep_overall = separate_data["Overall"]
con_overall = concat_data["Overall"]

# =========================
# Plot: Single CodeLMs (6개 한 번에)
# =========================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

single_combs = ["1", "2", "3", "4", "5", "6"]

for idx, comb_str in enumerate(single_combs):
    ax = axes[idx]
    title = LLM_IDS[comb_str]
    comb_num = "1"  # Single LLMs are under "1"
    
    sep_values = sep_overall.get(comb_num, {}).get(comb_str)
    con_values = con_overall.get(comb_num, {}).get(comb_str)

    if sep_values is None or con_values is None:
        print(f"[⚠] Missing data for {title}, skipping subplot.")
        ax.axis("off")
        continue

    N = [1, 2, 3, 4, 5]

    # Plot curves
    ax.plot(N, sep_values, marker='o', color='blue', linewidth=2.2, label='Separate Embedding')
    ax.plot(N, con_values, marker='o', color='green', linewidth=2.2, label='Concatenated Embedding')

    # Style
    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Top-N", fontsize=12)
    ax.set_xticks(N)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.8)

# Layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Separate vs Concatenated Embedding Performance (6 Single CodeLMs)", fontsize=18, y=0.98)

# Save
output_path_singles = os.path.join(output_dir, "ablation2_single.png")
plt.savefig(output_path_singles, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved single-model combined figure: {output_path_singles}\n")
