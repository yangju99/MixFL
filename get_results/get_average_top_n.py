import os
import pickle
import json
from itertools import combinations

LLM_MAP = {
    1: "codebert_base",
    2: "codegen_350m",
    3: "codet5_base",
    4: "graphcodebert_base",
    5: "incoder_1b",
    6: "unixcoder_base",
}

DATASETS = [
    "Chart",
    "Lang",
    "Math",
    "Time",
    "Closure",
    "Mockito",
]

SEEDS = [str(i) for i in range(1, 6)]
RESULTS_ROOT = "../results"
TOP_N_LEN = 5
OUTPUT_JSON = os.path.join(RESULTS_ROOT, "averaged_top_n.json")

def load_top_n(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[!] Failed to load {path}: {e}")
        return None

def average_lists(list_of_lists):
    avg = [sum(x) / len(x) for x in zip(*list_of_lists)]
    return [round(v) for v in avg]

def sum_lists(a, b):
    return [x + y for x, y in zip(a, b)]

def main():
    llm_ids = sorted(LLM_MAP.keys())
    result_dict = {}
    overall_dict = {}   # {comb_num: {comb_str: [summed top_n]}}

    for project in DATASETS:
        result_dict[project] = {}

        for r in range(1, len(llm_ids) + 1):
            for comb in combinations(llm_ids, r):
                comb_str = "".join(str(i) for i in comb)
                comb_num = str(len(comb))

                top_n_across_seeds = []

                for seed in SEEDS:
                    pkl_path = os.path.join(RESULTS_ROOT, seed, project, comb_num, comb_str, "top_n.pkl")
                    top_n = load_top_n(pkl_path)

                    if top_n is None or not isinstance(top_n, list) or len(top_n) != TOP_N_LEN:
                        print(f"[!] Skipping seed {seed}: Invalid or missing data for {project}/{comb_num}/{comb_str}")
                        continue

                    top_n_across_seeds.append(top_n)

                if len(top_n_across_seeds) == 0:
                    raise RuntimeError(
                        f"[❌ ERROR] No valid seeds found for {project}/{comb_num}/{comb_str}"
                    )

                averaged = average_lists(top_n_across_seeds)

                result_dict[project].setdefault(comb_num, {})[comb_str] = averaged

                overall_dict.setdefault(comb_num, {})
                if comb_str not in overall_dict[comb_num]:
                    overall_dict[comb_num][comb_str] = averaged
                else:
                    overall_dict[comb_num][comb_str] = sum_lists(
                        overall_dict[comb_num][comb_str],
                        averaged
                    )

    result_dict["Overall"] = {}
    for comb_num, comb_data in overall_dict.items():
        result_dict["Overall"][comb_num] = {}
        for comb_str, summed_top_n in comb_data.items():
            result_dict["Overall"][comb_num][comb_str] = summed_top_n  

    with open(OUTPUT_JSON, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✅ All results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()


