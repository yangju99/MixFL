import json
import os
import numpy as np

input_path = "../results/averaged_top_n.json"

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

# 단일 모델만 포함된 조합 찾기 (예: "1", "2", ..., "6")
single_llm_results = {}

for comb_num_str, comb_results in overall_data.items():
    for comb_str, top_n in comb_results.items():
        # 조합이 숫자 한 개만 포함된 경우
        if len(comb_str.strip()) == 1 and isinstance(top_n, list):
            llm_id = int(comb_str)
            # Top-1 성능 사용 (필요시 Top-3, Top-5로 바꿔도 됨)
            top1_score = top_n[0]
            single_llm_results[LLM_MAP[llm_id]] = top1_score

# 성능 순위 계산 (내림차순)
sorted_llms = sorted(single_llm_results.items(), key=lambda x: x[1], reverse=True)

print("=== Individual CodeLM Top-1 Performance Ranking ===")
for rank, (name, score) in enumerate(sorted_llms, start=1):
    print(f"{rank}. {name:<15} : {score:.4f}")

print("\n(LLM 이름순 정렬도 함께 출력)")
for name in sorted(single_llm_results.keys()):
    print(f"{name:<15} : {single_llm_results[name]:.4f}")