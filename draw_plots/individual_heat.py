import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ======== 기본 설정 ========
input_path = "../results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "individual_heatmap_masked.png")

# ======== 데이터 로드 ========
with open(input_path, "r") as f:
    data = json.load(f)

overall_data = data.get("Overall", {})

# ======== 모델 이름 & 순서 정의 ========
LLM_ORDER = ["CodeBERT", "GraphCodeBERT", "CodeGen", "CodeT5", "InCoder", "UniXcoder"]
LLM_MAP = {
    1: "CodeBERT",
    2: "CodeGen",
    3: "CodeT5",
    4: "GraphCodeBERT",
    5: "InCoder",
    6: "UniXcoder",
}
LLM_INV = {v: k for k, v in LLM_MAP.items()}

# ======== Heatmap 데이터 초기화 ========
heat_data = pd.DataFrame(np.nan, index=LLM_ORDER, columns=LLM_ORDER)

# ======== Helper: 조합 문자열을 숫자 리스트로 변환 ========
def parse_comb_str(comb_str):
    try:
        return sorted(list(map(int, comb_str.strip())))
    except:
        return []

# ======== 데이터 채우기 ========
for comb_num_str, comb_results in overall_data.items():
    comb_num = int(comb_num_str)
    for comb_str, top_n in comb_results.items():
        if not isinstance(top_n, list) or len(top_n) < 5:
            continue
        included = parse_comb_str(comb_str)

        # 1개짜리 조합 → 대각선
        if len(included) == 1:
            llm_name = LLM_MAP[included[0]]
            heat_data.loc[llm_name, llm_name] = int(round(top_n[0]))
        # 2개짜리 조합 → off-diagonal
        elif len(included) == 2:
            a, b = included
            name_a, name_b = LLM_MAP[a], LLM_MAP[b]
            val = int(round(top_n[0]))
            heat_data.loc[name_a, name_b] = val
            heat_data.loc[name_b, name_a] = val

# ======== 상삼각 마스크 생성 (대각선 제외) ========
# mask = np.triu(np.ones_like(heat_data, dtype=bool), k=1)

mask = np.triu(np.ones_like(heat_data, dtype=bool))

# k=1 → 대각선보다 한 칸 위부터 마스킹 (즉, diagonal은 False)

# ======== 시각화 ========
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    heat_data,
    mask=mask,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Top-1"},
)

# ======== 테두리 추가 ========
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Heatmap saved to: {output_path}")
plt.show()
