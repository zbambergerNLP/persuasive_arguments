import random
import typing

import numpy as np
import torch
import os

import constants


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


def get_num_labels(task_name):
    return len(constants.PREMISE_MODE_TO_INT) if task_name == constants.MULTICLASS else constants.NUM_LABELS


def print_metrics(eval_metrics: typing.Mapping[str, typing.Sequence[typing.Mapping[str, float]]]):
    for base_model_type, base_model_eval_metrics in eval_metrics.items():
        metrics_aggregates = {}
        print(f'\n\tBase model type: {base_model_type}')
        for split_index, split_metrics in enumerate(base_model_eval_metrics):
            for metric_name, metric_value in split_metrics.items():
                if metric_name not in metrics_aggregates:
                    metrics_aggregates[metric_name] = []
                metrics_aggregates[metric_name].append(metric_value)

        metric_averages = {}
        metric_stds = {}
        for metric_name, metric_values in metrics_aggregates.items():
            metric_averages[metric_name] = sum(metric_values) / len(metric_values)
            metric_stds[metric_name] = np.std(metric_values)

        for metric_name in metric_averages.keys():
            print(f'\t\tmetric name: {metric_name}\n'
                  f'\t\t\tmean metric value: {metric_averages[metric_name]}\n'
                  f'\t\t\tstandard deviation: {metric_stds[metric_name]}')
