import json
import os
import matplotlib.pyplot as plt

# =========================
# File Paths
# =========================
best5_path = "../results/best5_comb_results.json"
averaged_path = "../results/averaged_top_n.json"  # for individual LLMs
equal_weight_path = "/workspace/projects/MixFL_equal_weights/results/averaged_top_n.json"
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LLM ID Mapping
# =========================
LLM_IDS = {
    "CodeBERT": "1",
    "CodeGen": "2",
    "CodeT5": "3",
    "GraphCodeBERT": "4",
    "InCoder": "5",
    "UniXcoder": "6",
}
ID_TO_NAME = {v: k for k, v in LLM_IDS.items()}

# =========================
# Load Data
# =========================
with open(best5_path, "r") as f:
    best5_data = json.load(f)

with open(averaged_path, "r") as f:
    averaged_data = json.load(f)

with open(equal_weight_path, "r") as f:
    equal_weight_data = json.load(f)

overall = averaged_data["Overall"]
equal_overall = equal_weight_data["Overall"]

# =========================
# Create Combined Figure (2×3 grid)
# =========================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# =========================
# Plot Best 5 Combinations
# =========================
for idx, rank_entry in enumerate(best5_data[:5]):  # top 5 combos
    ax = axes[idx]
    rank = rank_entry["rank"]
    comb_str = rank_entry["comb_str"]
    comb_num = rank_entry["comb_num"]
    overall_topn = rank_entry["overall_topn"]

    llm_ids = list(comb_str)
    llm_names = [ID_TO_NAME[i] for i in llm_ids]

    print(f"[✔] Rank {rank} | Combo {comb_str} ({llm_names})")

    # Individual LLM Results
    individual_results = {}
    for llm_id in llm_ids:
        name = ID_TO_NAME[llm_id]
        result = overall.get("1", {}).get(llm_id)
        if result is not None and len(result) >= 5:
            individual_results[name] = result[:5]

    # Equal-weight Results
    eq_weight = equal_overall.get(comb_num, {}).get(comb_str)
    if eq_weight is None:
        print(f"[⚠] Equal-weight result not found for comb {comb_num}-{comb_str}. Using zeros.")
        eq_weight = [0, 0, 0, 0, 0]

    # Plot
    ax.plot(range(1, 6), overall_topn, marker='o', label="Adaptive Mixing", color='blue', linewidth=2.5)
    ax.plot(range(1, 6), eq_weight, marker='o', label="Uniform Mixing", color='red', linewidth=2.5)

    color_list = ['green', 'purple', 'orange', 'brown', 'gray']
    for i, (model_name, scores) in enumerate(individual_results.items()):
        ax.plot(range(1, 6), scores, label=model_name,
                color=color_list[i % len(color_list)], linestyle='--', linewidth=1.8)

    ax.set_title(f"Rank {rank}: {' + '.join(llm_names)}", fontsize=14, pad=10)
    ax.set_xlabel("N", fontsize=12)
    ax.set_ylabel("Top-N", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xticks(range(1, 6))
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)

# =========================
# Plot Bottom-right (6th) — Full 6-LLM Combination
# =========================
ax = axes[5]
comb_num = "6"
comb_str = "123456"
full_llm_names = [ID_TO_NAME[i] for i in list(comb_str)]

# Load adaptive and equal-weight results
adaptive_full = overall.get(comb_num, {}).get(comb_str)
equal_full = equal_overall.get(comb_num, {}).get(comb_str)

if adaptive_full is None:
    print(f"[⚠] Adaptive full-combination result not found for {comb_str}.")
    adaptive_full = [0, 0, 0, 0, 0]
if equal_full is None:
    print(f"[⚠] Equal-weight full-combination result not found for {comb_str}.")
    equal_full = [0, 0, 0, 0, 0]

# Individual LLM results
individual_results = {}
for llm_id in list(comb_str):
    name = ID_TO_NAME[llm_id]
    result = overall.get("1", {}).get(llm_id)
    if result is not None and len(result) >= 5:
        individual_results[name] = result[:5]

# Plot full 6-LLM
ax.plot(range(1, 6), adaptive_full, marker='o', label="Adaptive Mixing", color='blue', linewidth=2.5)
ax.plot(range(1, 6), equal_full, marker='o', label="Uniform Mixing", color='red', linewidth=2.5)

color_list = ['green', 'purple', 'orange', 'brown', 'gray', 'pink']
for i, (model_name, scores) in enumerate(individual_results.items()):
    ax.plot(range(1, 6), scores, label=model_name,
            color=color_list[i % len(color_list)], linestyle='--', linewidth=1.8)

ax.set_title("Full Combination (6 CodeLMs)", fontsize=14, pad=10)
ax.set_xlabel("N", fontsize=12)
ax.set_ylabel("Top-N", fontsize=12)
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_xticks(range(1, 6))
ax.tick_params(axis='both', labelsize=10)
ax.legend(fontsize=9, loc="lower right", framealpha=0.8)

# =========================
# Finalize and Save
# =========================
plt.tight_layout(rect=[0, 0, 1, 1])
# fig.suptitle("Adaptive Mixing vs Equal Weight Mixing vs Single CodeLM (Best 5 + Full Combination)", fontsize=20)


output_path = os.path.join(output_dir, "ablation1_best5_full.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\n✅ Combined figure saved (5 best + full combination): {output_path}\n")
