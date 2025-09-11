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
OUTPUT_JSON = os.path.join(RESULTS_ROOT, "total_top_1_set.json")


def load_top_1_set(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            if not isinstance(data, set):
                raise ValueError("Not a set")
            return data
    except Exception as e:
        print(f"[!] Failed to load {path}: {e}")
        return None


def main():
    llm_ids = sorted(LLM_MAP.keys())
    result_dict = {}
    overall_dict = {}  # {comb_num: {comb_str: set()}}

    for project in DATASETS:
        result_dict[project] = {}

        for r in range(1, len(llm_ids) + 1):
            for comb in combinations(llm_ids, r):
                comb_str = "".join(str(i) for i in comb)
                comb_num = str(len(comb))

                union_set = set()

                for seed in SEEDS:
                    pkl_path = os.path.join(RESULTS_ROOT, seed, project, comb_num, comb_str, "top_1_set.pkl")
                    top_1_set = load_top_1_set(pkl_path)

                    if top_1_set is None:
                        print(f"[!] Skipping seed {seed}: Invalid or missing data for {project}/{comb_num}/{comb_str}")
                        continue

                    union_set.update(top_1_set)

                if not union_set:
                    raise RuntimeError(f"[❌] No valid top_1 sets for {project}/{comb_num}/{comb_str} (All {SEEDS} seeds failed)")

    
                result_dict[project].setdefault(comb_num, {})[comb_str] = sorted(list(union_set))

            
                overall_dict.setdefault(comb_num, {})
                if comb_str not in overall_dict[comb_num]:
                    overall_dict[comb_num][comb_str] = set(union_set)
                else:
                    overall_dict[comb_num][comb_str].update(union_set)


    result_dict["Overall"] = {}
    for comb_num, comb_data in overall_dict.items():
        result_dict["Overall"][comb_num] = {}
        for comb_str, union_set in comb_data.items():
            result_dict["Overall"][comb_num][comb_str] = sorted(list(union_set))


    with open(OUTPUT_JSON, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✅ All union sets saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
