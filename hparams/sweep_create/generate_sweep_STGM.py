import os
import json
import numpy as np
from copy import deepcopy
from itertools import product

# Base config
base_config = {
    "optimizer": "sgd",
    "datadir": "dataset",
    "device": "cuda",
    "device_id": "0",
    "dataset": "IMAGENET1k",    
    "num_classes": 1000,
    "model": "CNN",
    "batch_size": 64,
    "local_learning_rate": 0.005,
    "learning_rate_decay": False,
    "learning_rate_decay_gamma": 0.99,
    "global_rounds": 100,
    "local_epochs": 1,
    "algorithm": "FedSTGM",
    "join_ratio": 1.0,
    "random_join_ratio": False,
    "num_clients": 10,
    "prev": 0,
    "times": 1,
    "eval_gap": 1,
    "out_folder": "out",
    "note": None,
    "num_tasks": 500,
    "client_drop_rate": 0.0,
    "time_threthold": 10000,
    "stgm_rounds": 100,
    "stgm_learning_rate": 25,
    "stgm_momentum": 0.5,
    "stgm_step_size": 30,
    "stgm_gamma": 0.5,
    "stgm_c": 0.25,
    "stgm_meta_lr": 0.5,
    "grad_balance": False,
    "coreset": False,
    "tgm": False,
    "sgm": True
}

# New sweep parameters
sweep_params = {
    "stgm_rounds": [25, 100],
    "stgm_learning_rate": [5, 50],
    "stgm_momentum": [0.1, 0.5],
    "stgm_step_size": [10, 50],
    "stgm_gamma": [float(np.round(x, 2)) for x in np.linspace(0.1, 1.0, 4)],
    "stgm_c": [float(np.round(x, 2)) for x in np.linspace(0.1, 1.0, 4)],
    "stgm_meta_lr": [0.5, 1.0],
    "grad_balance": [False, True],
    "local_epochs": [3, 10],
    "local_learning_rate": [0.01, 0.05],
}

# Create all combinations
param_names = list(sweep_params.keys())
param_values = list(sweep_params.values())
all_combinations = list(product(*param_values))  # 3072 configs

# Divide into 14 nearly equal parts
num_total = len(all_combinations)
num_parts = 14
part_indices = np.array_split(np.arange(num_total), num_parts)

# Output root folder
root_folder = "../sweep_STGM_grid_split"
os.makedirs(root_folder, exist_ok=True)

for part_idx, indices in enumerate(part_indices):
    if part_idx < 8:
        comp_id = 1 if part_idx < 4 else 2
        sub_id = part_idx % 4
        folder = os.path.join(root_folder, f"computer{comp_id}", f"part{sub_id}")
        os.makedirs(folder, exist_ok=True)
        gpu_id = "0"
        job_folder = folder
    else:
        gpu_num = (part_idx - 8) % 2  # gpu0 or gpu1
        job_id = (part_idx - 8) // 2  # job0 to job2
        gpu_folder = os.path.join(root_folder, "computer3", f"gpu{gpu_num}")
        job_folder = os.path.join(gpu_folder, f"job{job_id}")
        os.makedirs(job_folder, exist_ok=True)
        gpu_id = str(gpu_num)

    for config_idx in indices:
        config = deepcopy(base_config)
        config["device_id"] = gpu_id
        note_parts = []
        for name, value in zip(param_names, all_combinations[config_idx]):
            config[name] = value
            note_parts.append(f"{name}={value}")
        config["note"] = ", ".join(note_parts)

        filename = os.path.join(job_folder, f"config_{config_idx:04d}.json")
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
