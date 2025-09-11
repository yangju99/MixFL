import os
import logging
import pickle
import torch
import numpy as np
import random
import json
import logging
import dgl
import inspect
import pdb 

def read_json(filepath):
    if os.path.exists(filepath):
        assert filepath.endswith('.json')
        with open(filepath, 'r') as f:
            return json.loads(f.read())
    else: 
        return None

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        logging.info("Using GPU...")
        return torch.device("cuda")
    logging.info("Using CPU...")
    return torch.device("cpu")

def collate(data):
    """
    data: [ (graph_dict, fault_gts), ... ]
    return:
      per_model_graphs: { model_name: batched_dgl_graph }
      fault_gts: list
      node_counts: List[int]  
    """
    graph_dicts, fault_gts = map(list, zip(*data))


    model_names = list(graph_dicts[0].keys())

    per_model_graphs = {}
    for m in model_names:
        graphs = [gd[m] for gd in graph_dicts]
        per_model_graphs[m] = dgl.batch(graphs)


    node_counts = [gd[model_names[0]].num_nodes() for gd in graph_dicts]

    return per_model_graphs, fault_gts, node_counts


def save_logits_as_dict(logits, keys, filename):
    """
    Saves a list of tensors as a dictionary with variable names as keys and tensor values as dictionary values.
    """
    frame = inspect.currentframe().f_back
    tensor_dict = {}
    
    for logit in logits:
        names = [name for name, var in frame.f_locals.items() if torch.is_tensor(var) and var is tensor and not name.startswith('_')]
        
        if names:
            tensor_dict[names[0]] = logit
            
    return tensor_dict

import hashlib
def dump_params(params):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in params.items()])).encode("utf-8")).hexdigest()[0:8]


    result_dir = os.path.join(params["model_save_dir"], hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(result_dir, "params.json"))

    log_file = os.path.join(result_dir, "running.log")


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )
    return hash_id

from datetime import datetime, timedelta
def dump_scores(result_dir, hash_id, total_top_n, top_1_set):

    with open(os.path.join(result_dir, 'experiments.txt'), 'a+') as fw:
        fw.write(hash_id + ': ' + (datetime.now() + timedelta(hours=8)).strftime("%Y/%m/%d-%H:%M:%S") + '\n')
        fw.write("* Test result -- " + str(total_top_n))
        fw.write('{}{}'.format('=' * 40, '\n'))


    top_1_path = os.path.join(result_dir, "top_1_set.pkl")
    total_top_n_path = os.path.join(result_dir, "top_n.pkl")

    with open(top_1_path, "wb") as f:
        pickle.dump(top_1_set, f)

    with open(total_top_n_path, "wb") as f:
        pickle.dump(total_top_n, f)



def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj,fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False)


def graphs_to_device(per_model_graphs, device):
    return {m: g.to(device) for m, g in per_model_graphs.items()}