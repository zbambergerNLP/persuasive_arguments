from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import constants

accuracy = load_metric(constants.ACCURACY)


def compute_metrics(num_labels: int,
                    preds: typing.Union[torch.Tensor | typing.Sequence[int]],
                    targets: typing.Union[torch.Tensor | typing.Sequence[int]],
                    logits: torch.Tensor = None,
                    is_train: bool = False) -> typing.Mapping[str, float]:
    """

    :param num_labels: The number of labels for the probing classification problem.
    :param preds: Model predictions that are compared to ground truth labels to compute metrics.
    :param targets: The ground truth labels supplied by the dataset.
    :param logits: A tensor from which predictions are drawn.
    :param is_train: True if we are computing metrics in train mode. False otherwise.
    :return:
    """
    if logits:
        preds = np.argmax(logits, axis=-1)
    average = 'binary' if num_labels == 2 else 'micro'
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=targets, y_pred=preds, average=average)
    precision_key = f'{constants.TRAIN}_{constants.PRECISION}' if is_train else constants.PRECISION
    recall_key = f'{constants.TRAIN}_{constants.RECALL}' if is_train else constants.RECALL
    f1_key = f'{constants.TRAIN}_{constants.F1}' if is_train else constants.F1
    accuracy_key = f'{constants.TRAIN}_{constants.ACCURACY}' if is_train else constants.ACCURACY
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
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='binary')
    metrics = {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy.compute(predictions=predictions, references=labels)
    }
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


def get_baseline_scores(task_name: str, corpus_df: pd.DataFrame) -> typing.Mapping[str, float]:
    """Evaluate the performance of an n-gram language model on a probing task.

    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}
    :param corpus_df: A pandas DataFrame instance with the columns {OP_COMMENT, REPLY, LABEL}. The OP_COMMENT
        and REPLY columns consist of text entries while LABEL is {0, 1}.
    :return: A dictionary of metrics containing the following keys: precision, recall, f1, accuracy.
    """
    if task_name == constants.INTRA_ARGUMENT_RELATIONS:
        corpus_df['combined'] = corpus_df[constants.SENTENCE_1] + corpus_df[constants.SENTENCE_2]
    elif task_name == constants.BINARY_CMV_DELTA_PREDICTION:
        corpus_df['combined'] = corpus_df[constants.OP_COMMENT] + corpus_df[constants.REPLY]
    else:  # Binary or multi-class premise mode prediction.
        corpus_df['combined'] = corpus_df[constants.CLAIM_TEXT] + corpus_df[constants.PREMISE_TEXT]

    train, test = train_test_split(corpus_df, test_size=0.2)

    cv = CountVectorizer(binary=True, min_df=1, max_df=0.95, ngram_range=(1, 2))
    cv.fit_transform(train['combined'].values.astype('U'))
    train_feature_set = cv.transform(train['combined'].values.astype('U'))
    test_feature_set = cv.transform(test['combined'].values.astype('U'))

    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    if constants.PREMISE_MODE in train:
        y_train = train[constants.PREMISE_MODE]
    else:
        y_train = train[constants.LABEL]

    lr.fit(train_feature_set, y_train)
    y_pred = lr.predict(test_feature_set)

    if constants.PREMISE_MODE in test:
        y_test = test[constants.PREMISE_MODE]
    else:
        y_test = test[constants.LABEL]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = sum(y_test == y_pred) / len(y_test)

    return {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy
    }
