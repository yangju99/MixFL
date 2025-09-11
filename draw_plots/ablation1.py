import json
import os
import matplotlib.pyplot as plt


input_path = "../results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ablation1.png")


LLM_IDS = {
    "CodeBERT": "1",
    "CodeGen": "2",
    "CodeT5": "3",
    "GraphCodeBERT": "4",
    "InCoder": "5",
    "UniXcoder": "6",
}

# Equal Weight 
equal_weight_topn = [192, 228, 251, 268, 280]  


with open(input_path, "r") as f:
    data = json.load(f)

overall = data["Overall"]


individual_results = {}
for name, id_ in LLM_IDS.items():
    result = overall.get("1", {}).get(id_)
    if result is not None and len(result) >= 5:
        individual_results[name] = result[:5]  

# Adaptive Embedding Mixing 
adaptive_result = overall.get("6", {}).get("123456")
if adaptive_result:
    adaptive_result = adaptive_result[:5]  

plt.figure(figsize=(12, 8))

# Adaptive
if adaptive_result:
    plt.plot(
        range(1, len(adaptive_result) + 1),
        adaptive_result,
        marker='o',
        label="Adaptive Embedding Mixing",
        color='blue',
        linewidth=2.5
    )

# Equal Weight
plt.plot(
    range(1, len(equal_weight_topn) + 1),
    equal_weight_topn,
    marker='o',
    label="Equal Weight",
    color='red',
    linewidth=2.5
)


color_map = {
    "CodeBERT": "pink",
    "CodeGen": "orange",
    "GraphCodeBERT": "purple",
    "UniXcoder": "green",
    "CodeT5": "gold",
    "InCoder": "black",
}
linestyle_map = {name: "--" for name in LLM_IDS}

for model_name, topn_scores in individual_results.items():
    plt.plot(
        range(1, len(topn_scores) + 1),
        topn_scores,
        label=model_name,
        color=color_map.get(model_name, 'gray'),
        linestyle=linestyle_map.get(model_name, '-'),
        linewidth=2
    )


plt.xlabel("N", fontsize=16)
plt.ylabel("Top-N", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(range(1, 6))
plt.legend(fontsize=15)
plt.tight_layout()


plt.savefig(output_path, dpi=300)
print(f"✅ Figure saved to: {output_path}")
plt.show()
