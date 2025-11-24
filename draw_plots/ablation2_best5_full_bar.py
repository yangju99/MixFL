import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

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
# Setup
# =========================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

N = [1, 3, 5]
index_map = [0, 2, 4]
sep_color = '#4C72B0'
con_color = '#DD8452'

# =========================
# Compute global y-axis range (공통 y축 범위)
# =========================
all_sep_vals, all_con_vals = [], []

# top 5 combinations
for entry in best5_data[:5]:
    sep_full = entry["overall_topn"]
    con_full = con_overall.get(entry["comb_num"], {}).get(entry["comb_str"], None)
    if sep_full and con_full:
        all_sep_vals.extend([sep_full[i] for i in index_map])
        all_con_vals.extend([con_full[i] for i in index_map])

# full combination (6)
sep_full = sep_overall.get("6", {}).get("123456")
con_full = con_overall.get("6", {}).get("123456")
if sep_full and con_full:
    all_sep_vals.extend([sep_full[i] for i in index_map])
    all_con_vals.extend([con_full[i] for i in index_map])

GLOBAL_YMIN = min(min(all_sep_vals), min(all_con_vals))
GLOBAL_YMAX = max(max(all_sep_vals), max(all_con_vals))
MARGIN = (GLOBAL_YMAX - GLOBAL_YMIN) * 0.2  # 강조 위해 여유 추가

# =========================
# Helper for consistent axis style
# =========================
def style_axes(ax, title):
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Top-N", fontsize=12)
    ax.set_xticks(np.arange(len(N)))
    ax.set_xticklabels(N, fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # ✅ 모든 subplot 동일한 y축 범위
    ax.set_ylim(GLOBAL_YMIN - MARGIN, GLOBAL_YMAX + MARGIN * 0.3)

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc="upper left", frameon=True, framealpha=0.9, fancybox=True, edgecolor='black')

# =========================
# Plot Best 5
# =========================
for idx, entry in enumerate(best5_data[:5]):
    ax = axes[idx]
    rank = entry["rank"]
    comb_num = entry["comb_num"]
    comb_str = entry["comb_str"]

    llm_names = " + ".join([ID_TO_NAME[c] for c in comb_str])

    sep_values_full = entry["overall_topn"]
    con_values_full = con_overall.get(comb_num, {}).get(comb_str, None)
    if sep_values_full is None or con_values_full is None:
        print(f"[⚠] Missing data for comb {comb_num}-{comb_str}")
        ax.axis("off")
        continue

    sep_values = [sep_values_full[i] for i in index_map]
    con_values = [con_values_full[i] for i in index_map]
    x = np.arange(len(N))
    bar_width = 0.4

    sep_bars = ax.bar(x - bar_width/2, sep_values, width=bar_width,
                      color=sep_color, edgecolor='black', linewidth=0.8, alpha=0.9, label='Separate')
    con_bars = ax.bar(x + bar_width/2, con_values, width=bar_width,
                      color=con_color, edgecolor='black', linewidth=0.8, alpha=0.9, label='Concatenated')

    # ✅ 정수형 라벨 표시
    for bars in [sep_bars, con_bars]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.4, f"{int(round(h))}",
                    ha='center', va='bottom', fontsize=10, fontweight='medium')

    style_axes(ax, f"Rank {rank}: {llm_names}")

# =========================
# Full Combination (6)
# =========================
ax = axes[5]
comb_num = "6"
comb_str = "123456"
title = "Full Combination (6 CodeLMs)"

sep_values_full = sep_overall.get(comb_num, {}).get(comb_str)
con_values_full = con_overall.get(comb_num, {}).get(comb_str)

if sep_values_full and con_values_full:
    sep_values = [sep_values_full[i] for i in index_map]
    con_values = [con_values_full[i] for i in index_map]
    x = np.arange(len(N))
    bar_width = 0.4

    sep_bars = ax.bar(x - bar_width/2, sep_values, width=bar_width,
                      color=sep_color, edgecolor='black', linewidth=0.8, alpha=0.9, label='Separate')
    con_bars = ax.bar(x + bar_width/2, con_values, width=bar_width,
                      color=con_color, edgecolor='black', linewidth=0.8, alpha=0.9, label='Concatenated')

    for bars in [sep_bars, con_bars]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.4, f"{int(round(h))}",
                    ha='center', va='bottom', fontsize=10, fontweight='medium')

    style_axes(ax, title)
else:
    ax.axis("off")
    print("[⚠] Missing full combination data.")

# =========================
# Save
# =========================
plt.tight_layout()
output_path = os.path.join(output_dir, "ablation2_best5_full_bar.png")
plt.savefig(output_path, dpi=400, bbox_inches='tight')
plt.close()

print(f"\n✅ Figure saved with uniform y-axis and integer labels: {output_path}\n")
