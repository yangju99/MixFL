import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # ← 추가
from matplotlib.lines import Line2D
import numpy as np

input_path = "../results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "individual.png")

with open(input_path, "r") as f:
    data = json.load(f)

overall_data = data.get("Overall", {})

LLM_MAP = {
    1: "CodeBERT",
    2: "CodeGen",
    3: "CodeT5",
    4: "GraphCodeBERT",
    5: "InCoder",
    6: "UniXcoder",
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

# 색상 결정 함수
def get_color_for_combination(comb_str):
    """
    CodeBERT(1) → red
    UniXcoder(6) → blue
    둘 다 포함 → green
    둘 다 없음 → gray
    """
    try:
        included = set(map(int, comb_str.strip()))
    except Exception as e:
        print(f"Parsing error for comb_str={comb_str}: {e}")
        return "gray"

    has_codebert = 1 in included
    has_unixcoder = 6 in included

    if has_codebert and has_unixcoder:
        return "green"
    elif has_codebert:
        return "red"
    elif has_unixcoder:
        return "blue"
    else:
        return "gray"

# Collect all points by combination
for comb_num_str, comb_results in overall_data.items():
    comb_num = int(comb_num_str)
    for comb_str, top_n in comb_results.items():
        if not isinstance(top_n, list) or len(top_n) < 5:
            continue
        color = get_color_for_combination(comb_str)
        for idx in target_indices:
            offset = (comb_num - 3.5) * 0.12
            x = base_x[idx] + offset
            y = top_n[idx]
            points_by_comb[comb_num].append((x, y, color))

# Plot setup
plt.figure(figsize=(10, 6))

# Scatter points by combination
for comb_num in range(1, 7):
    marker = marker_map[comb_num]
    size = 100 if marker in ['*', 'X'] else 60

    for (x, y, color) in points_by_comb[comb_num]:
        plt.scatter(
            x,
            y,
            c=color,
            marker=marker,
            s=size,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.3
        )

# Compute and plot dashed average lines for Top-1, 3, 5
for idx in target_indices:
    topn_label = base_x[idx]  # 1, 3, 5
    x_vals = []
    y_vals = []
    for comb_num in range(1, 7):
        x_target = base_x[idx] + (comb_num - 3.5) * 0.12
        y_targets = [y for (x, y, c) in points_by_comb[comb_num] if abs(x - x_target) < 1e-3]
        if y_targets:
            mean_y = np.mean(y_targets)
            x_vals.append(x_target)
            y_vals.append(mean_y)
            mean_by_topn[topn_label].append((x_target, mean_y))

# Axis & Labels
plt.xticks([1, 3, 5], ["1", "3", "5"])
plt.xlabel("N", fontsize=13)
plt.ylabel("Top-N", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.5)

# ✅ Legend (사각형 patch로 변경, 텍스트는 그대로)
# legend_elements = [
#     Patch(facecolor='red', edgecolor='black', label='CodeBERT'),
#     Patch(facecolor='blue', edgecolor='black', label='UniXcoder'),
#     Patch(facecolor='green', edgecolor='black', label='Both'),
#     Patch(facecolor='gray', edgecolor='black', label='Neither'),
# ]

legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='CodeBERT', alpha=0.7),
    Patch(facecolor='blue', edgecolor='black', label='UniXcoder', alpha=0.7),
    Patch(facecolor='green', edgecolor='black', label='Both', alpha=0.7),
    Patch(facecolor='gray', edgecolor='black', label='Neither', alpha=0.7),
]

plt.legend(handles=legend_elements, title="Combination Includes", loc="best", fontsize=11, title_fontsize=12)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"✅ Figure saved to: {output_path}")
plt.show()
