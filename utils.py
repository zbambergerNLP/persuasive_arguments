import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir_exists(probing_dir_path):
    if not os.path.exists(probing_dir_path):
        print(f'Creating directory: {probing_dir_path}')
        os.mkdir(probing_dir_path)


def print_metrics(eval_metrics):
    for split, base_model_type_eval_metrics in eval_metrics.items():
        print(f'\tsplit #{split}')
        for base_model_name, metrics_dict in base_model_type_eval_metrics.items():
            print(f'\t\tbase model name: {base_model_name}')
            for metric_name, metric_value in metrics_dict.items():
                print(f'\tsplit: {split}, base model name: {base_model_name}')
                print(metric_name)
                print(metric_value)
