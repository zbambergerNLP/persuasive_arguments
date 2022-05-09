import random
import typing

import datasets
import numpy as np
import torch
import os
import re
import constants


# TODO: Create a function to count the number of examples associated with each label.
import data_loaders


def set_seed(seed: int = 42):
    """
    Set a consistent seed across random generation mechanisms.

    :param seed: The seed that will be used consistently across number generating systems.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir_exists(directory: str):
    """
    If the provided directory does not exist, but its higher level directory does, create the specified directory.

    :param directory: The string path to the directory, which may or may not exist in the filesystem.
    """
    if not os.path.exists(directory):
        print(f'Creating directory: {directory}')
        os.mkdir(directory)


def get_num_labels(task_name: str) -> int:
    """
    Return the number of labels that exist in the specified task.

    :param task_name: The string name of the task, which provides a mapping to the label space of the task.
    :return: An integer representing the number of labels in the disired task.
    """
    return len(constants.PREMISE_MODE_TO_INT) if task_name == constants.MULTICLASS else constants.NUM_LABELS


def aggregate_metrics_across_splits(
        split_metrics: typing.Sequence[typing.Mapping[str, float]]) -> typing.Mapping[str, typing.Sequence[float]]:
    """
    Aggregate metric values across k folds of cross validation.

    :param split_metrics: A sequence of metric dictionaries. Each entry in the sequence corresponds to a unique set of
        metrics obtained when evaluating the model over a fold (i.e., a fold within k-fold cross validation). Metrics
        are stored as a dictionary where metric names map to their corresponding float values.
    :return: A mapping from the metric names to their values across all folds.
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
        split_name: str,
        print_results: bool = False,) -> typing.Tuple[typing.Mapping[str, float], typing.Mapping[str, float]]:
    """
    Calculate the average metric scores and their standard deviations across k-folds of cross validation.

    :param metric_aggregates: A mapping from the metric names to their values across all folds (within k-fold cross
        validation).
    :param split_name: The name of the current split. One of {"train", "validation", "test"}.
    :param print_results: True if we intend to print the resulting metric averages as well as their standard deviations
        across k folds.
    :return: A 2-tuple containing:
        1. A mapping from metric names to their average value across k folds.
        2. A mapping from metric names to their standard deviation across k folds.
    """
    metric_averages = {}
    metric_stds = {}
    for metric_name, metric_values in metric_aggregates.items():
        metric_averages[metric_name] = sum(metric_values) / len(metric_values)
        metric_stds[metric_name] = float(np.std(metric_values, axis=-1))
    if print_results:
        for metric_name in metric_averages.keys():
            print(f'\t\tmetric name ({split_name}): {metric_name}\n'
                  f'\t\t\tmean metric value ({split_name}): {metric_averages[metric_name]:.3f}\n'
                  f'\t\t\tstandard deviation ({split_name}): {metric_stds[metric_name]:.3f}')
    return metric_averages, metric_stds


def print_metrics(eval_metrics: typing.Mapping[str, typing.Sequence[typing.Mapping[str, float]]]):
    """
    Print the aggregated means and standard deviations of various model types across k folds of cross validation.

    :param eval_metrics: A dictionary mapping a model's name to a sequence of k metric dictionaries. These metric
        dictionaries are obtained over k folds of cross validation.
    """
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
                  f'\t\t\tmean metric value: {metric_averages[metric_name]:.3f}\n'
                  f'\t\t\tstandard deviation: {metric_stds[metric_name]:.3f}')


def get_dataset_stats(kg_dataset: data_loaders.CMVKGDataset):
    """
    Print general information about a given knowledge graph based dataset.

    :param kg_dataset: A knowledge graph dataset consisting of argumentative prepositions (nodes) and their relations
        (edges), as well as a measure of persuasiveness (graph label).
    """
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
                re_len -= 1
            sentences_in_node += re_len
    avg_num_of_words_in_nodes = words_in_nodes / num_of_nodes
    avg_num_of_sentences_in_nodes = sentences_in_node / num_of_nodes
    print(f'number of positive examples {num_of_positive_examples}')
    print(f'number of negative examples {num_of_negative_examples}')
    print(f'average number of words in each node {avg_num_of_words_in_nodes}')
    print(f'average number of sentences in each node {avg_num_of_sentences_in_nodes}')


def create_data_loaders(training_set: datasets.Dataset,
                        validation_set: datasets.Dataset,
                        test_set: datasets.Dataset,
                        batch_size: int = 16,
                        shuffle_train: bool = True,
                        shuffle_validation: bool = False,
                        shuffle_test: bool = False) -> (
        typing.Tuple[torch.utils.data.DataLoader,
                     torch.utils.data.DataLoader,
                     torch.utils.data.DataLoader]):
    """
    Create data loaders corresponding to the train, validation, and test dataset splits.

    :param training_set: The training set on over which the training loader is created. The training set maps BERT
        inputs to their corresponding labels.
    :param validation_set: The validation set over which the validation loader is created. The training set maps BERT
        inputs to their corresponding labels.
    :param test_set: The test set over which the test loader is created. The training set maps BERT inputs to their
        corresponding labels.
    :param batch_size: The size of batches sampled from the test loader.
    :param shuffle_train: True if we wish to shuffle the training set while sampling from the data loader. False
        otherwise.
    :param shuffle_validation: True if we wish to shuffle the validation set while sampling from the data loader. False
        otherwise.
    :param shuffle_test: True if we wish to shuffle the test set while sampling from the data loader. False
        otherwise.
    :return: A 3-tuple containing the data loaders for the training, validation, and test set respectively.
    """
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=shuffle_train
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=shuffle_validation
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle_test
    )
    return train_loader, validation_loader, test_loader