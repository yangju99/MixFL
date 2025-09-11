#!/usr/bin/env python3
import subprocess
from itertools import combinations
import os

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
    "Time",
    "Mockito",
    "Math",
    "Closure",
]

def main():
    llm_ids = sorted(LLM_MAP.keys())  

    for seed in range(1, 6):  # seed from 1 to 5
        for project_type in DATASETS:
            for r in range(1, len(llm_ids) + 1):
                for comb in combinations(llm_ids, r):
                    sorted_ids = sorted(comb)
                    comb_number = len(sorted_ids)
                    id_str = "".join(str(i) for i in sorted_ids)
                    result_dir = f"./results/{seed}/{project_type}/{comb_number}/{id_str}"
                    result_file = os.path.join(result_dir, "experiments.txt")

                    if os.path.isfile(result_file):
                        print(f"[✔] Skipping existing result: {result_file}")
                        continue

                    args = ["--random_seed", str(seed), "--project_type", project_type]
                    for llm_id in sorted_ids:
                        args += ["--llm_id", str(llm_id)]

                    comb_names = ",".join(LLM_MAP[i] for i in sorted_ids)
                    print(f"Running seed={seed}, project_type={project_type}, "
                          f"LLMs=[{comb_names}] (k={r})")

                    cmd = ["python", "leave_one_out_project.py"] + args
                    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()


