import os
import json

INPUT_JSON = "../results/averaged_top_n.json"
OUTPUT_JSON = "../results/best5_comb_results.json"


def get_top_k_overall_combs(data, k=5):
    """
    Overall 섹션에서 Top-1 → Top-2 → Top-3 → Top-4 → Top-5 순으로
    lexicographical 정렬하여 상위 k개 조합 반환
    """
    if "Overall" not in data:
        raise ValueError("[❌] 'Overall' key is missing in input data.")

    comb_list = []

    for comb_num, combs in data["Overall"].items():
        for comb_str, top_n_list in combs.items():
            if not isinstance(top_n_list, list) or len(top_n_list) == 0:
                continue
            comb_list.append(((comb_num, comb_str), top_n_list))

    if not comb_list:
        raise RuntimeError("[❌] No valid combination found in Overall section.")

    # Top-1~Top-5 순서로 lexicographical 정렬 (내림차순)
    def sort_key(item):
        top_n = item[1]
        return tuple(top_n[i] if i < len(top_n) else -1 for i in range(5))

    comb_list.sort(key=sort_key, reverse=True)

    return comb_list[:k]


def extract_results_for_comb(data, comb_num, comb_str):
    """특정 조합에 대한 프로젝트별 성능을 추출"""
    result = {}

    for project in data:
        if project == "Overall":
            continue

        project_data = data[project]
        project_result = project_data.get(comb_num, {}).get(comb_str, None)

        if project_result is not None:
            result[project] = project_result
        else:
            print(f"[!] Missing result for project={project}, comb={comb_str}")

    return result


def main():
    if not os.path.isfile(INPUT_JSON):
        print(f"[!] File not found: {INPUT_JSON}")
        return

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    top_k_combs = get_top_k_overall_combs(data, k=5)
    print(f"✅ Found Top-{len(top_k_combs)} Overall combinations (ranked by Top-1 → Top-5)")

    results_to_save = []

    for rank, ((comb_num, comb_str), overall_topn) in enumerate(top_k_combs, start=1):
        display_topn = ", ".join(f"{x:.4f}" for x in overall_topn[:5])
        print(f"#{rank}: Comb={comb_num}/{comb_str}, Top-1~5=({display_topn})")
        project_results = extract_results_for_comb(data, comb_num, comb_str)

        results_to_save.append({
            "rank": rank,
            "comb_num": comb_num,
            "comb_str": comb_str,
            "overall_topn": overall_topn,
            "project_results": project_results
        })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"✅ Top-{len(top_k_combs)} results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
