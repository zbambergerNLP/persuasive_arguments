import random
import typing

import numpy as np
import torch
import os
import re
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


def aggregate_metrics_across_splits(
        split_metrics: typing.Sequence[typing.Mapping[str, float]]) -> typing.Mapping[str, typing.Sequence[float]]:
    """

    :param split_metrics:
    :return:
    """
    metrics_aggregates = {}
    for split_index, split_metrics_dict in enumerate(split_metrics):
        for metric_name, metric_value in split_metrics_dict.items():
            if metric_name not in metrics_aggregates:
                metrics_aggregates[metric_name] = []
            metrics_aggregates[metric_name].append(metric_value)
    return metrics_aggregates


def get_metrics_avg_and_std_across_splits(
        metric_aggregates: typing.Mapping[str, typing.Sequence[float]],
        print_results: bool = False,
        is_train: bool = False) -> typing.Tuple[typing.Mapping[str, float], typing.Mapping[str, float]]:
    """

    :param metric_aggregates:
    :param print_results:
    :param is_train:
    :return:
    """
    metric_averages = {}
    metric_stds = {}
    for metric_name, metric_values in metric_aggregates.items():
        metric_averages[metric_name] = sum(metric_values) / len(metric_values)
        metric_stds[metric_name] = np.std(metric_values, axis=-1)
    if print_results:
        for metric_name in metric_averages.keys():
            print(f'\t\tmetric name ({"train" if is_train else "eval"}): {metric_name}\n'
                  f'\t\t\tmean metric value ({"train" if is_train else "eval"}): {metric_averages[metric_name]}\n'
                  f'\t\t\tstandard deviation ({"train" if is_train else "eval"}): {metric_stds[metric_name]}')
    return metric_averages, metric_stds


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


def get_dataset_stats(kg_dataset):
    num_of_positive_examples = sum(kg_dataset.labels)
    num_of_negative_examples = len(kg_dataset.labels) - num_of_positive_examples

    num_of_nodes = 0
    words_in_nodes = 0
    sentences_in_node = 0
    for s in kg_dataset.dataset:
        n = s['id_to_text']
        num_of_nodes += len(n)
        for key in n:
            words_in_nodes += len(n[key].split())
            re_len = len(re.split(r'[.!?]+', n[key]))
            if re_len > 1:
                print(re_len)
                re_len -=1
            sentences_in_node+=re_len
    avg_num_of_words_in_nodes = words_in_nodes / num_of_nodes
    avg_num_of_sentences_in_nodes = sentences_in_node / num_of_nodes
    print(f'number of positive examples {num_of_positive_examples}')
    print(f'number of negatibe examples {num_of_negative_examples}')
    print(f'average number of words in each node {avg_num_of_words_in_nodes}')
    print(f'average number of sentences in each node {avg_num_of_sentences_in_nodes}')
