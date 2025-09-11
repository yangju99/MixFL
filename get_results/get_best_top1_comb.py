import os
import json

INPUT_JSON = "../results/averaged_top_n.json"
OUTPUT_JSON = "../results/top1_best_overall_comb_results.json"

def get_best_overall_comb(data):
    if "Overall" not in data:
        raise ValueError("[❌] 'Overall' key is missing in input data.")

    best_top1 = -1
    best_comb = None
    best_overall_topn = None

    for comb_num, combs in data["Overall"].items():
        for comb_str, top_n_list in combs.items():
            if not isinstance(top_n_list, list) or len(top_n_list) == 0:
                continue
            top1 = top_n_list[0]
            if top1 > best_top1:
                best_top1 = top1
                best_comb = (comb_num, comb_str)
                best_overall_topn = top_n_list

    if best_comb is None:
        raise RuntimeError("[❌] No valid combination found in Overall section.")

    return best_comb, best_overall_topn  # ((comb_num, comb_str), [top@1, top@3, ...])


def extract_results_for_comb(data, comb_num, comb_str):
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

    (comb_num, comb_str), overall_topn = get_best_overall_comb(data)
    print(f"✅ Best Overall Top-1 Comb = ({comb_num}, {comb_str})")

    selected_results = extract_results_for_comb(data, comb_num, comb_str)

    result_to_save = {
        "best_comb_num": comb_num,
        "best_comb_str": comb_str,
        "overall_topn": overall_topn,
        "project_results": selected_results
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result_to_save, f, indent=2)

    print(f"✅ Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
