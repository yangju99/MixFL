import os
import pickle

ROOT_DIR = "./chunks_defects4j_1.4.0"

MODELS = [
    "codebert_base",
    "codegen_350m",
    "graphcodebert_base",
    "unixcoder_base",
    "codet5_base",
    "incoder_1b"
]

PROJECTS = ["Lang", "Chart", "Closure", "Math", "Time", "Mockito"]

# SEEDS = [1, 2, 3, 4, 5]
SEEDS = [4, 5]  

def process_model_seed_dir(model: str, seed: int):
    model_seed_dir = os.path.join(ROOT_DIR, model, str(seed))
    data_path = os.path.join(model_seed_dir, "all.pkl")
    output_dir = model_seed_dir  

    if not os.path.isfile(data_path):
        print(f"[SKIP] all.pkl not found: {data_path}")
        return

    with open(data_path, "rb") as f:
        chunks = pickle.load(f)


    project_chunks = {proj: [] for proj in PROJECTS}


    for chunk in chunks:
        project_id = chunk.get("project_id", "") 
        for proj in PROJECTS:
            if project_id.startswith(proj):
                project_chunks[proj].append(chunk)
                break  

    print(f"\n=== {model}/{seed} ===")

    for project_name in project_chunks:
        print(project_name + ": " + str(len(project_chunks[project_name])))


    for proj, proj_chunks in project_chunks.items():
        out_path = os.path.join(output_dir, f"{proj}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(proj_chunks, f)
        print(f"[✔] save completed: {out_path} ({len(proj_chunks)})")


if __name__ == "__main__":
    for model in MODELS:
        for seed in SEEDS:
            process_model_seed_dir(model, seed)

