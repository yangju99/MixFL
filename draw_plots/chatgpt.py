import os
import matplotlib.pyplot as plt

# ===========================
# Output directory & file
# ===========================
output_dir = "./results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "param_size_logscale.png")

# ===========================
# Model sizes (in millions)
# ===========================
models = [
    "CodeBERT",
    "GraphCodeBERT",
    "CodeT5",
    "UniXcoder",
    "CodeGen",
    "InCoder",
    "ChatGPT-3"
]

sizes_millions = [
    125,      # CodeBERT
    125,      # GraphCodeBERT
    215,      # CodeT5
    125,      # UniXcoder
    350,      # CodeGen
    1300,     # InCoder (1.3B)
    175000    # ChatGPT (175B)
]

# ===========================
# Colors: CodeLMs = green, ChatGPT = red
# ===========================
colors = ["green"] * 6 + ["red"]

# ===========================
# Plot
# ===========================
plt.figure(figsize=(30, 7))

bars = plt.barh(models, sizes_millions, color=colors)

plt.xscale("log")   # 로그 스케일
plt.xlabel("Number of Parameters (Millions, log scale)", fontsize=20)
# plt.title("Parameter Size Comparison: CodeLMs vs ChatGPT (log scale)")

plt.yticks(fontsize=20)

plt.tick_params(axis='x', labelsize=20)

# Annotate bars
for bar, size in zip(bars, sizes_millions):
    plt.text(
        size,
        bar.get_y() + bar.get_height() / 2,
        f"{size:,}",
        va='center',
        ha='left',
        fontsize=20
    )

# Legend
plt.legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color='green', label='CodeLMs'),
        plt.Rectangle((0, 0), 1, 1, color='red', label='LLM (ChatGPT)')
    ],
    loc='lower right',
    fontsize=20,          # 🔥 legend 글자 크기 확대
    handlelength=2.5,     # 막대(아이콘) 길이 확대
    handleheight=2.0,     # 아이콘 높이 확대
    borderpad=1.2,        # legend 박스 내부 여백 증가
    labelspacing=1.0      # 항목 간 간격 증가
)

plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Figure saved to: {output_path}")
