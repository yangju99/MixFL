import os
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
from venn import venn
import pdb 


JSON_PATH = "../results/total_top_1_set.json"
OUTPUT_PATH = "../results/figures/overlap.png"


LLM_MAP = {
    1: "CodeBERT",
    2: "CodeGen",
    3: "CodeT5",
    4: "GraphCodeBERT",
    5: "InCoder",
    6: "UniXcoder",
}

def main():
    if not os.path.isfile(JSON_PATH):
        print(f"[!] JSON file not found: {JSON_PATH}")
        return

    with open(JSON_PATH, "r") as f:
        result_dict = json.load(f)

    overall_data = result_dict.get("Overall", {}).get("1", {})  


    if not overall_data:
        print("[!] No Overall data found for comb_num=1")
        return


    sets = OrderedDict()
    for llm_id, llm_name in LLM_MAP.items():

        #change this if you want to get rid of a CodeLM from the resulting png.
        if llm_id == 4:
            continue

        comb_str = str(llm_id)
        if comb_str not in overall_data:
            print(f"[SKIP] No data for {llm_name} (ID {comb_str})")
            continue
        sets[llm_name] = set(overall_data[comb_str])
        print(f"[OK] {llm_name}: {len(sets[llm_name])} items")

    if len(sets) < 2:
        print("[!] Not enough sets to plot a Venn diagram")
        return


    all_union = set().union(*sets.values())
    all_intersection = set.intersection(*sets.values()) if len(sets) >= 2 else set()
    print(f"\n[SUMMARY] union={len(all_union)}, intersection(all)={len(all_intersection)}")


    plt.figure(figsize=(10, 7), dpi=200)
    venn(sets, fontsize=12, legend_loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"[✔] Figure saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
