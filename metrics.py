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
                    split_name: str = None) -> typing.Mapping[str, float]:
    """

    :param num_labels: The number of labels for the probing classification problem.
    :param preds: Model predictions that are compared to ground truth labels to compute metrics.
    :param targets: The ground truth labels supplied by the dataset.
    :param split_name: The string name of the dataset split. Typically one of {train, validation, test}.
    :param use_huggingface_prefix:
    :return: A dictionary mapping metric names to their corresponding values (as evaluated on the provided split).
    """
    average = 'binary' if num_labels == 2 else 'micro'
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=targets, y_pred=preds, average=average,
                                                               zero_division=1)

    metrics_name_fn = lambda split_name, metric_name: f"{split_name}_{metric_name}" if split_name else metric_name
    precision_key = metrics_name_fn(split_name=split_name, metric_name=constants.PRECISION)
    recall_key = metrics_name_fn(split_name=split_name, metric_name=constants.RECALL)
    f1_key = metrics_name_fn(split_name=split_name, metric_name=constants.F1)
    accuracy_key = metrics_name_fn(split_name=split_name, metric_name=constants.ACCURACY)
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


def perform_evaluation_on_splits(eval_fn: typing.Callable,
                                 device: torch.device,
                                 train_loader: torch.utils.data.DataLoader = None,
                                 validation_loader: torch.utils.data.DataLoader = None,
                                 test_loader: torch.utils.data.DataLoader = None) -> (
        typing.Mapping[str, typing.Mapping[str, float]]):
    """
    Perform evaluation with a trained model on the splits contained within the provided data loaders.
    :param eval_fn: A function which, given a trained pytorch model, evaluates the data from a supplied data loader.
        This callable function must include the following parameters:
            * dataloader: The data loader containing the dataset on which evaluation is performed.
            * split_name: The name of the split corresponding to the data loader. One of {"train", "validation",
                "test"}.
            * device: The device on which the model and data reside during evaluation.
    :param train_loader: A data loader containing the training set on which the model is evaluated.
    :param validation_loader: A data loader containing the validation set on which the model is evaluated.
    :param test_loader: A data loader containing the test set on which the model is evaluated.
    :param device: The device the holds both the data and the model during evaluation.
    :return: A dictionary mapping the split name to the metrics dictionary associated with evaluation metrics on that
        split.
    """
    all_metrics = {}
    if train_loader:
        all_metrics[constants.TRAIN] = eval_fn(
            dataloader=train_loader,
            split_name=constants.TRAIN,
            device=device)
    if validation_loader:
        all_metrics[constants.VALIDATION] = eval_fn(
            dataloader=validation_loader,
            split_name=constants.VALIDATION,
            device=device)
    if test_loader:
        all_metrics[constants.TEST] = eval_fn(
            dataloader=test_loader,
            split_name=constants.TEST,
            device=device)
    return all_metrics

