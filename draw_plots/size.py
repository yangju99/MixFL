import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

input_path = "../results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "size.png")

with open(input_path, "r") as f:
    data = json.load(f)

overall_data = data.get("Overall", {})

color_map = {
    1: 'red',
    2: 'orange',
    3: 'goldenrod',
    4: 'green',
    5: 'blue',
    6: 'purple'
}

marker_map = {
    1: 'o',  # circle
    2: 's',  # square
    3: '^',  # triangle
    4: 'D',  # diamond
    5: '*',  # star
    6: 'X',  # X
}

target_indices = [0, 2, 4]  # index for Top-1, Top-3, Top-5
base_x = {0: 1, 2: 3, 4: 5}  # X-axis labels

points_by_comb = {k: [] for k in range(1, 7)}
mean_by_topn = {1: [], 3: [], 5: []}  # for average dashed lines

# Collect all points by combination
for comb_num_str, comb_results in overall_data.items():
    comb_num = int(comb_num_str)
    for comb_str, top_n in comb_results.items():
        if not isinstance(top_n, list) or len(top_n) < 5:
            continue
        for idx in target_indices:
            offset = (comb_num - 3.5) * 0.12
            x = base_x[idx] + offset
            y = top_n[idx]
            points_by_comb[comb_num].append((x, y))

# Plot setup
plt.figure(figsize=(10, 6))

# Scatter points by combination
for comb_num in range(1, 7):
    x_vals = [x for x, y in points_by_comb[comb_num]]
    y_vals = [y for x, y in points_by_comb[comb_num]]
    marker = marker_map[comb_num]
    color = color_map[comb_num]
    size = 100 if marker in ['*', 'X'] else 60

    plt.scatter(
        x_vals,
        y_vals,
        c=color,
        marker=marker,
        s=size,
        alpha=0.7,
        label=f"{comb_num} CodeLM{'s' if comb_num > 1 else ''}"
    )

# Compute and plot dashed average lines for Top-1, 3, 5
for idx in target_indices:
    topn_label = base_x[idx]  # 1, 3, 5
    x_vals = []
    y_vals = []
    for comb_num in range(1, 7):
        x_target = base_x[idx] + (comb_num - 3.5) * 0.12
        y_targets = [y for (x, y) in points_by_comb[comb_num] if abs(x - x_target) < 1e-3]
        if y_targets:
            mean_y = np.mean(y_targets)
            x_vals.append(x_target)
            y_vals.append(mean_y)
            mean_by_topn[topn_label].append((x_target, mean_y))

    if x_vals and y_vals:
        plt.plot(
            x_vals,
            y_vals,
            linestyle='dashed',
            color='black',
            linewidth=1.5,
            label=f"Top-{topn_label} Mean Line" if topn_label == 1 else None  # prevent duplicate legend
        )

# Axis & Labels
plt.xticks([1, 3, 5], ["1", "3", "5"])
plt.xlabel("N", fontsize=13)
plt.ylabel("Top-N", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='1 CodeLM', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='2 CodeLMs', markerfacecolor='orange', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='3 CodeLMs', markerfacecolor='goldenrod', markersize=10),
    Line2D([0], [0], marker='D', color='w', label='4 CodeLMs', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='*', color='w', label='5 CodeLMs', markerfacecolor='blue', markersize=14),
    Line2D([0], [0], marker='X', color='w', label='6 CodeLMs', markerfacecolor='purple', markersize=14),
    Line2D([0], [0], linestyle='dashed', color='black', label='Mean Line')
]
plt.legend(handles=legend_elements, title="Number of CodeLMs", loc="best", fontsize=11, title_fontsize=12)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"✅ Figure saved to: {output_path}")
plt.show()