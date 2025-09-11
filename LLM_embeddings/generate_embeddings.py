import os
import json
import re
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm 
import argparse
from typing import Callable, List, Dict, Any
import importlib


def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        return None


def load_get_context_embedding(llm_name: str) -> Callable[[str], Any]:

    try:
        module = importlib.import_module(llm_name)
    except ModuleNotFoundError:
        raise ValueError(
            f"unsupported LLM : {llm_name}\n"
            f"check LLM name (EX: codebert_base, codegen_350m, unixcoder_base ...)."
        )
    if not hasattr(module, "get_context_embedding"):
        raise ValueError(f"{llm_name} doesn't has  get_context_embedding function.")
    return getattr(module, "get_context_embedding")



def generate_chunks(
    data_dir: str,
    save_dir: str,
    embedding_size: int,
    get_context_embedding: Callable[[str], Any],
):
    """
    code + comment embedding.
    """
    chunks: List[Dict[str, Any]] = []

    for entry in tqdm(os.listdir(data_dir), desc="Generating embeddings for buggy versions"):
        bug_id = entry  # e.g., Chart_1
        print(f"[•] Processing: {bug_id}")

        snippet_path = os.path.join(data_dir, entry, "snippet.json")
        covered_methods = read_json(snippet_path)
        if not covered_methods:
            print(f"[!] skip: snippet.json not found or empty: {snippet_path}")
            continue

        method_num = len(covered_methods)
        id_to_code, id_to_comment = {}, {}
        fault_ids = []

        for method_id, covered_method in enumerate(covered_methods):
            signature = covered_method.get("signature", "")
            _method_name = signature.split("(")[0] if signature else f"method_{method_id}"

            id_to_code[method_id] = covered_method.get("snippet", "") or ""
            id_to_comment[method_id] = (covered_method.get("comment", "") or "").strip()

            if covered_method.get("is_bug", False):
                fault_ids.append(method_id)

        chunk = {
            "project_id": entry,
            "code_vector": np.zeros((method_num, embedding_size), dtype=np.float32),
            "comment_vector": np.zeros((method_num, embedding_size), dtype=np.float32),
            "code_comment_vector": np.zeros((method_num, embedding_size), dtype=np.float32),
        }

        sorted_ids = sorted(id_to_code.keys())

        for method_id in sorted_ids:
            code_text = id_to_code[method_id]
            comment_text = id_to_comment[method_id]
            combined_text = f"{code_text}\n{comment_text}" if comment_text else code_text

            # ---- code_vector ----
            try:
                emb_code = get_context_embedding(code_text).detach().cpu().numpy()
                chunk["code_vector"][method_id] = emb_code
            except Exception as e:
                print(f"[!] code embedding fail: {entry} #{method_id}: {e}")

            # ---- comment_vector ----
            if comment_text:
                try:
                    emb_comment = get_context_embedding(comment_text).detach().cpu().numpy()
                    chunk["comment_vector"][method_id] = emb_comment
                except Exception as e:
                    print(f"[!] comment embedding fail: {entry} #{method_id}: {e}")

            # ---- code_comment_vector ----
            try:
                emb_combined = get_context_embedding(combined_text).detach().cpu().numpy()
                chunk["code_comment_vector"][method_id] = emb_combined
            except Exception as e:
                print(f"[!] code+comment embedding fail: {entry} #{method_id}: {e}")

        chunk["fault_label"] = fault_ids
        chunks.append(chunk)

    os.makedirs(save_dir, exist_ok=True)
    pkl_path = os.path.join(save_dir, "all.pkl")
    with open(pkl_path, "wb") as pkl:
        pickle.dump(chunks, pkl)

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with selectable LLM backend.")
    parser.add_argument("--llm", required=True, help="LLM  name. EX: codebert_base, codegen_350m")
    parser.add_argument("--data_dir", default="./defects4j", help="input directory (default: ./defects4j)")
    parser.add_argument("--save_root", default="./chunks_defects4j", help="root directory")
    parser.add_argument("--seed", type=int, default=42, help="random seed number")
    args = parser.parse_args()

    get_context_embedding = load_get_context_embedding(args.llm)


    sample = get_context_embedding("def add(a,b): return a+b")
    embedding_size = int(sample.shape[0])

 
    save_dir = os.path.join(args.save_root, args.llm, str(args.seed))


    generate_chunks(args.data_dir, save_dir, embedding_size, get_context_embedding)



if __name__ == "__main__":

    main() 






