import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LLM_ARCH = {
    1: "encoder",
    2: "decoder",
    3: "enc-dec",
    4: "encoder",
    5: "decoder",
    6: "enc-dec"
}


arch_color_map = {
    1: "red",
    2: "green",
    3: "blue"
}

arch_marker_map = {
    1: "o",  # circle
    2: "s",  # square
    3: "^",  # triangle
}

x_offsets = {
    1: -0.2,
    2: 0.0,
    3: 0.2
}

input_path = "../results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "architecture_diversity.png")

with open(input_path, "r") as f:
    data = json.load(f)

overall_data = data.get("Overall", {})

# Top-1, 3, 5
target_indices = [0, 2, 4]
target_labels = [1, 3, 5]


points_by_arch = {1: [], 2: [], 3: []}

for _, comb_results in overall_data.items():
    for comb_str, topn in comb_results.items():
        if not isinstance(topn, list) or len(topn) < 5:
            continue
        llm_ids = [int(c) for c in comb_str]
        arch_count = len(set(LLM_ARCH[i] for i in llm_ids))

        for idx in target_indices:
            x = target_labels[target_indices.index(idx)] + x_offsets[arch_count]
            y = topn[idx]
            points_by_arch[arch_count].append((x, y))

import numpy as np

plt.figure(figsize=(10, 6))


mean_by_topn = {1: [], 3: [], 5: []}


for arch_count in [1, 2, 3]:
    x_vals = [x for x, y in points_by_arch[arch_count]]
    y_vals = [y for x, y in points_by_arch[arch_count]]
    color = arch_color_map[arch_count]
    marker = arch_marker_map[arch_count]
    
    plt.scatter(
        x_vals,
        y_vals,
        c=color,
        marker=marker,
        alpha=0.7,
        label=f"{arch_count} Architecture Type{'s' if arch_count > 1 else ''}"
    )


    for i, topn_x in enumerate(target_labels):  # [1, 3, 5]
        x_target = topn_x + x_offsets[arch_count]
        y_target_vals = [y for (x, y) in points_by_arch[arch_count] if abs(x - x_target) < 1e-3]
        if y_target_vals:
            mean_y = np.mean(y_target_vals)
            mean_by_topn[topn_x].append((arch_count, mean_y))  # e.g., (1, 212.5)


for topn_x in target_labels:
    if len(mean_by_topn[topn_x]) == 3:
        sorted_points = sorted(mean_by_topn[topn_x], key=lambda x: x[0])
        x_vals = [topn_x + x_offsets[arch_count] for arch_count, _ in sorted_points]
        y_vals = [mean for _, mean in sorted_points]

        plt.plot(
            x_vals, y_vals,
            linestyle='dashed',
            color='black',
            linewidth=1.5,
            label=f"Top-{topn_x} Mean Line" if topn_x == 1 else None 
        )


plt.xticks(target_labels)
plt.xlabel("N", fontsize=13)
plt.ylabel("Top-N", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='1 Architecture Type', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='2 Architecture Types', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='3 Architecture Types', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], linestyle='dashed', color='black', label='Mean Line')
]
plt.legend(handles=legend_elements, title="Architecture Diversity", loc="best", fontsize=11, title_fontsize=12)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"✅ Figure saved to: {output_path}")
plt.show()
