from torch.utils.data import Dataset, DataLoader
import torch
import dgl
from utils import * 
import pickle
import sys
import logging
from base import BaseModel
import time
from utils import *
import pandas as pd 
import pdb 
from tqdm import tqdm 
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class chunkDataset(Dataset):
    """
    chunk structure
      {
        'project_id': str,
        'fault_label': ...,
        'code_vector': { model_name: np.ndarray [N, D_code_m] },
        'comment_vector': { model_name: np.ndarray [N, D_cmt_m] }
      }
    """
    def __init__(self, chunks):
        self.data = []
        self.idx2id = {}
        for idx, chunk in enumerate(chunks):
            self.idx2id[idx] = chunk['project_id']

            graph_dict = {}
            for model_name, code_vector in chunk['code_vector'].items():
                comment_vector = chunk['comment_vector'][model_name]

                g = dgl.graph(([], []), num_nodes=len(code_vector))
                g.ndata["code_vector"] = torch.as_tensor(code_vector, dtype=torch.float32)
                g.ndata["comment_vector"] = torch.as_tensor(comment_vector, dtype=torch.float32)
                graph_dict[model_name] = g

            fault_gts = chunk['fault_label']
            self.data.append((graph_dict, fault_gts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def leave_one_out_project(params):

    model_chunks = {}
    total_bug_num = None 
    for model_name, data_path in params['model_names'].items(): 

        with open(data_path, 'rb') as te:
            dataset = pickle.load(te)
        te.close()

        params["code_dim"][model_name] = dataset[0]['code_vector'].shape[1]
        params["comment_dim"][model_name] = dataset[0]['comment_vector'].shape[1]

        total_bug_num = len(dataset)

        model_chunks[model_name] = dataset 
    

    chunks = [] 

    for bug_idx in range(total_bug_num):
        chunk = {} 
        code_vectors = {}
        comment_vectors = {}

        for model_name, dataset in model_chunks.items():
            chunk['fault_label'] = dataset[bug_idx]['fault_label']
            chunk['project_id'] = dataset[bug_idx]['project_id']

            code_vectors[model_name] = dataset[bug_idx]['code_vector']
            comment_vectors[model_name] = dataset[bug_idx]['comment_vector']
        
        chunk['code_vector'] = code_vectors 
        chunk['comment_vector'] = comment_vectors
        chunks.append(chunk)    

    device = get_device(params["check_device"])

    hash_id = dump_params(params)

    params["hash_id"] = hash_id
    print("hash_id: ", hash_id)
    
    total_top_n = [0 for _ in range(5)]
    top_1_set = set()

    for i in tqdm(range(len(chunks)), desc="Doing Leave-one-out cross validation") :
        test_data = chunkDataset([chunks[i]])
        train_data = chunkDataset(chunks[:i] + chunks[i+1:])

        train_dl = DataLoader(train_data, batch_size = params['batch_size'], shuffle=True, collate_fn=collate, pin_memory=True)
        test_dl = DataLoader(test_data, batch_size = params['batch_size'], shuffle=False, collate_fn=collate, pin_memory=True)

        model = BaseModel(device, lr = params["learning_rate"], **params)

         # Train model
        eval_res, converge = model.fit(train_dl, test_dl, evaluation_epoch= params['evaluation_epoch'])
        
        for idx, value in enumerate(eval_res['top_n']):
            total_top_n[idx] += value 

        if eval_res['top_n'][0] == 1:
            top_1_set.add(chunks[i]['project_id'])

    dump_scores(params["model_save_dir"], hash_id, total_top_n, top_1_set)
        
    logging.info("Current hash_id {}".format(hash_id))


# Instantiate your Dataset and DataLoader
############################################################################
if __name__ == "__main__":

    DEFAULT_TOP_DIR = "./LLM_embeddings/chunks_defects4j_1.4.0"


    LLM_MAP = {
        1: "codebert_base",
        2: "codegen_350m",
        3: "codet5_base",
        4: "graphcodebert_base",
        5: "incoder_1b",
        6: "unixcoder_base",
    }

    parser = argparse.ArgumentParser(description="Run the fault localization model")
    parser.add_argument("--project_type", required=True, help="EX: Chart, Lang, Math, Time, Closure, Mockito")
    parser.add_argument("--llm_id", type=int, action="append", required=True,
                        help="LLM id. EX: --llm_id 1 --llm_id 2 ...")
    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed for reproducibility")
    parser.add_argument("--top_dir", default=DEFAULT_TOP_DIR, help="data root directory")

    args = parser.parse_args()

    project = args.project_type  
    seed_str = str(args.random_seed)
    top_dir = args.top_dir


    sorted_llm_ids = sorted(args.llm_id)
    model_names = {}
    for mid in sorted_llm_ids:
        if mid not in LLM_MAP:
            raise ValueError(f"Unknown llm_id {mid}. Valid ids: {sorted(LLM_MAP.keys())}")
        model_name = LLM_MAP[mid]
        pkl_path = os.path.join(top_dir, model_name, seed_str, f"{project}.pkl")
        if not os.path.isfile(pkl_path):
            print(f"[!] Warning: not found: {pkl_path}")
        model_names[model_name] = pkl_path

    comb_number = len(sorted_llm_ids)
    id_str = "".join(str(x) for x in sorted_llm_ids)

    # save_dir: ./results/<seed>/<project>/<comb_len>/<sorted_ids>
    result_dir = f"./results/{seed_str}/{project}/{comb_number}/{id_str}"

    random_seed = args.random_seed
    batch_size = 1
    epochs = 30
    evaluation_epoch = 1
    learning_rate = 0.001
    model = "all"
    code_dim = {}
    comment_dim = {}

    seed_everything(random_seed)

    params = {
        "batch_size": batch_size,
        "epochs": epochs,
        "evaluation_epoch": evaluation_epoch,
        "learning_rate": learning_rate,
        "model": model,
        "model_save_dir": result_dir,
        "code_dim": code_dim,
        "comment_dim": comment_dim,
        "check_device": "gpu",
        "classification_hiddens": [128, 64],
        "model_names": model_names,   # {model_name: data_path}
        "seed": int(seed_str),
        "project": project,
    }

    leave_one_out_project(params)






