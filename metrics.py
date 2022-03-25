from __future__ import annotations

import typing

import numpy as np
import torch
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support

import constants

accuracy = load_metric(constants.ACCURACY)


def compute_metrics(num_labels: int,
                    preds: typing.Union[torch.Tensor | typing.Sequence[int]],
                    targets: typing.Union[torch.Tensor | typing.Sequence[int]],
                    split_name: str = constants.TRAIN) -> typing.Mapping[str, float]:
    """

    :param num_labels: The number of labels for the probing classification problem.
    :param preds: Model predictions that are compared to ground truth labels to compute metrics.
    :param targets: The ground truth labels supplied by the dataset.
    :param split_name: The string name of the dataset split. Typically one of {train, validation, test}.
    :return: A dictionary mapping metric names to their corresponding values (as evaluated on the provided split).
    """
    average = 'binary' if num_labels == 2 else 'micro'
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=targets, y_pred=preds, average=average)
    precision_key = f'{split_name}_{constants.PRECISION}'
    recall_key = f'{split_name}_{constants.RECALL}'
    f1_key = f'{split_name}_{constants.F1}'
    accuracy_key = f'{split_name}_{constants.ACCURACY}'
    metrics = {
        precision_key: precision,
        recall_key: recall,
        f1_key: f1,
        accuracy_key: accuracy.compute(predictions=preds, references=targets)[constants.ACCURACY]
    }
    return metrics


def compute_metrics_for_binary_classification(
        eval_pred: typing.Tuple[torch.Tensor, torch.Tensor]) -> typing.Mapping[str, float]:
    """
    Return a collection of evaluation metrics given a (logits, labels) pair for a binary classification problem.

    :param eval_pred: A 2-tuple of the form [logits, labels]. Labels is a collection of booleans. Logits is a collection
        of tensors corresponding to the model's logits for each input in the batch.
    :return: A dictionary of metrics containing the following keys: precision, recall, f1, accuracy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_metrics(num_labels=constants.NUM_LABELS, preds=predictions, targets=labels)
    return metrics


def compute_metrics_for_multi_class_classification(
        eval_pred: typing.Tuple[torch.Tensor, torch.Tensor]) -> typing.Mapping[str, float]:
    """
    Return a collection of evaluation metrics given a (logits, labels) pair for a multi-class classification problem.
    :param eval_pred: A 2-tuple of the form [logits, labels]. Labels is a collection of integers representing the label
        of the input. Logits is a collection of tensors corresponding to the model's logits for each input in the batch.
    :return: A dictionary of metrics containing the following keys: precision, recall, f1, accuracy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='micro')
    metrics = {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy.compute(predictions=predictions, references=labels)
    }
    return metrics
