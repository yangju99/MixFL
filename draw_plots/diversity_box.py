import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# ======== 기본 설정 ========
LLM_MAP = {
    1: "CodeBERT",
    2: "CodeGen",
    3: "CodeT5",
    4: "GraphCodeBERT",
    5: "InCoder",
    6: "UniXcoder",
}

LLM_ARCH = {
    1: "encoder",
    2: "decoder",
    3: "enc-dec",
    4: "encoder",
    5: "decoder",
    6: "enc-dec"
}

# 세련된 색상 팔레트 (논문 친화적)
arch_color_map = {
    1: "#D95F02",  # reddish orange
    2: "#1B9E77",  # green teal
    3: "#7570B3"   # muted purple
}

# ======== 경로 ========
input_path = "../results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "architecture_diversity_box.png")

# ======== 데이터 로드 ========
with open(input_path, "r") as f:
    data = json.load(f)

overall_data = data.get("Overall", {})

# ======== 목표 인덱스 ========
target_indices = [0, 2, 4]
target_labels = [1, 3, 5]

# ======== 데이터 수집 ========
points_by_arch_topn = {1: {1: [], 3: [], 5: []},
                       2: {1: [], 3: [], 5: []},
                       3: {1: [], 3: [], 5: []}}

for _, comb_results in overall_data.items():
    for comb_str, topn in comb_results.items():
        if not isinstance(topn, list) or len(topn) < 5:
            continue
        llm_ids = [int(c) for c in comb_str]
        arch_count = len(set(LLM_ARCH[i] for i in llm_ids))
        for idx, label in zip(target_indices, target_labels):
            points_by_arch_topn[arch_count][label].append(topn[idx])

# ======== Boxplot 생성 ========
plt.figure(figsize=(10, 6))
positions, box_data, colors = [], [], []
width = 0.25

for i, topn_label in enumerate(target_labels):
    base_x = i + 1
    for arch_count in [1, 2, 3]:
        data_points = points_by_arch_topn[arch_count][topn_label]
        if not data_points:
            continue
        pos = base_x + (arch_count - 2) * width
        positions.append(pos)
        box_data.append(data_points)
        colors.append(arch_color_map[arch_count])

bplots = plt.boxplot(
    box_data,
    positions=positions,
    widths=0.22,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=1.5),
    boxprops=dict(linewidth=1.2, edgecolor='black'),
    whiskerprops=dict(color='gray', linewidth=1),
    capprops=dict(color='gray', linewidth=1),
    flierprops=dict(marker='o', color='gray', markersize=3, alpha=0.5),
    meanprops=dict(marker='X', markerfacecolor='black', markeredgecolor='black', markersize=6)
)

# 색상 적용
for patch, color in zip(bplots['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

# ======== 시각적 세부 조정 ========
plt.xticks(range(1, len(target_labels) + 1), [f"{t}" for t in target_labels], fontsize=11)
plt.xlabel("N", fontsize=13, labelpad=5)
plt.ylabel("Top-N", fontsize=13, labelpad=5)
plt.grid(True, linestyle=':', alpha=0.4, color='gray', zorder=0)

plt.tick_params(axis='both', which='major', labelsize=10)
# ======== 범례 (패치 스타일) ========

legend_elements = [
    Patch(facecolor=arch_color_map[1], edgecolor='black', label='1 Architecture Type', alpha=0.65),
    Patch(facecolor=arch_color_map[2], edgecolor='black', label='2 Architecture Types', alpha=0.65),
    Patch(facecolor=arch_color_map[3], edgecolor='black', label='3 Architecture Types', alpha=0.65)
]

plt.legend(handles=legend_elements, title="Architecture Diversity", loc="best",
           fontsize=11, title_fontsize=12, frameon=True, edgecolor='gray')

plt.tight_layout()
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"✅ Figure saved to: {output_path}")
plt.show()
