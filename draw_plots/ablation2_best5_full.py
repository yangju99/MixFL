import json
import os
import matplotlib.pyplot as plt

# =========================
# File Paths
# =========================
best5_path = "../results/best5_comb_results.json"
separate_path = "../results/averaged_top_n.json"
concat_path = "/workspace/projects/MixFL_concatenate/results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LLM ID Mapping
# =========================
ID_TO_NAME = {
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
with open(best5_path, "r") as f:
    best5_data = json.load(f)

with open(separate_path, "r") as f:
    separate_data = json.load(f)

with open(concat_path, "r") as f:
    concat_data = json.load(f)

sep_overall = separate_data["Overall"]
con_overall = concat_data["Overall"]

# =========================
# Figure Setup (2 x 3 grid)
# =========================
# plt.close('all')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# =========================
# Plot Best 5 Combinations
# =========================
for idx, entry in enumerate(best5_data[:5]):  # top 5
    ax = axes[idx]
    rank = entry["rank"]
    comb_num = entry["comb_num"]
    comb_str = entry["comb_str"]

    # === Convert comb_str (e.g., "135") → LLM names (e.g., "CodeBERT + CodeT5 + InCoder")
    llm_names = " + ".join([ID_TO_NAME[c] for c in comb_str])

    sep_values = entry["overall_topn"]
    con_values = con_overall.get(comb_num, {}).get(comb_str, None)

    title = f"Rank {rank}: {llm_names}"

    if sep_values is None or con_values is None:
        print(f"[⚠] Missing data for comb {comb_num}-{comb_str}")
        ax.axis("off")
        continue

    N = [1, 2, 3, 4, 5]

    # Plot both
    ax.plot(N, sep_values, marker='o', color='blue', linewidth=2.5, label='Separate Embedding')
    ax.plot(N, con_values, marker='o', color='green', linewidth=2.5, label='Concatenated Embedding')

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Top-N", fontsize=12)
    ax.set_xticks(N)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)

# =========================
# Plot Full Combination (6th)
# =========================
ax = axes[5]
comb_num = "6"
comb_str = "123456"
title = "Full Combination (6 CodeLMs)"

sep_values = sep_overall.get(comb_num, {}).get(comb_str)
con_values = con_overall.get(comb_num, {}).get(comb_str)

if sep_values is not None and con_values is not None:
    N = [1, 2, 3, 4, 5]
    ax.plot(N, sep_values, marker='o', color='blue', linewidth=2.5, label='Separate Embedding')
    ax.plot(N, con_values, marker='o', color='green', linewidth=2.5, label='Concatenated Embedding')

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Top-N", fontsize=12)
    ax.set_xticks(N)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
else:
    ax.axis("off")
    print("[⚠] Missing full combination data.")

# =========================
# Final Layout
# =========================
plt.tight_layout(rect=[0, 0, 1, 1])
# fig.suptitle("Separate vs Concatenated Embedding (Best 5 + Full Combination)", fontsize=18, y=0.98)

# Save Figure
output_path = os.path.join(output_dir, "ablation2_best5_full.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Combined figure saved with simplified full title: {output_path}\n")
