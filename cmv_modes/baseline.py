import typing
import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

import constants
import data_loaders
import models
import utils


def create_combined_column(task_name: str, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param task_name:
    :param corpus_df:
    :return:
    """
    if task_name == constants.INTRA_ARGUMENT_RELATIONS:
        corpus_df['combined'] = corpus_df[constants.SENTENCE_1] + ' ' + corpus_df[constants.SENTENCE_2]
    elif task_name == constants.BINARY_CMV_DELTA_PREDICTION:
        corpus_df['combined'] = corpus_df[constants.OP_COMMENT] + ' ' + corpus_df[constants.REPLY]
    else:  # Binary or multi-class premise mode prediction.
        corpus_df['combined'] = corpus_df[constants.CLAIM_TEXT] + ' ' + corpus_df[constants.PREMISE_TEXT]
    return corpus_df


def extract_bigram_features(corpus_df: pd.DataFrame,
                            premise_mode: str = None) -> typing.Tuple[np.ndarray, np.ndarray]:
    """

    :param corpus_df:
    :param premise_mode:
    :return:
    """
    cv = CountVectorizer(binary=True, min_df=1, max_df=0.95, ngram_range=(1, 2))
    cv.fit_transform(corpus_df['combined'].values.astype('U'))
    X = cv.transform(corpus_df['combined'].values.astype('U')).toarray()
    y = (corpus_df[constants.PREMISE_MODE] if constants.PREMISE_MODE in corpus_df else corpus_df[
        constants.LABEL]).to_list()
    if premise_mode:
        y = np.array([1 if premise_mode in premise_label else 0 for premise_label in y])
    return X, y


def get_baseline_scores(task_name: str,
                        corpus_df: pd.DataFrame,
                        num_cross_validation_splits: int = 5,
                        premise_mode: str = None,
                        num_epochs: int = 50,
                        batch_size: int = 16,
                        learning_rate: float = 1e-2,
                        optimizer_gamma: float = 0.9) -> (
        typing.Tuple[typing.Mapping[str, float],
                     typing.Mapping[str, float]]):
    """

    :param task_name:
    :param corpus_df:
    :param num_cross_validation_splits:
    :param premise_mode:
    :param num_epochs:
    :param batch_size:
    :param learning_rate:
    :param optimizer_gamma:
    :return:
    """
    corpus_df = create_combined_column(task_name=task_name, corpus_df=corpus_df)
    X, y = extract_bigram_features(corpus_df=corpus_df, premise_mode=premise_mode)
    kf = KFold(n_splits=num_cross_validation_splits)
    num_labels = utils.get_num_labels(task_name=task_name)
    split_train_metrics = []
    split_eval_metrics = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logistic_regression = models.BaselineLogisticRegression(num_features=X_train.shape[1], num_labels=num_labels)
        optimizer = torch.optim.Adam(logistic_regression.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=optimizer_gamma)
        logistic_regression.fit(
            train_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_train, y_train),
                batch_size=batch_size,
                shuffle=True),
            num_labels=num_labels,
            loss_function=torch.nn.BCELoss() if num_labels == 2 else torch.nn.CrossEntropyLoss(),
            num_epochs=num_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        train_metrics = logistic_regression.evaluate(
            test_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
            ),
            num_labels=num_labels)
        split_train_metrics.append(train_metrics)

        eval_metrics = logistic_regression.evaluate(
            test_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_test, y_test),
                batch_size=batch_size,
                shuffle=True,
            ),
            num_labels=num_labels)
        split_eval_metrics.append(eval_metrics)

    eval_metric_aggregates = utils.aggregate_metrics_across_splits(split_eval_metrics)
    train_metric_aggregates = utils.aggregate_metrics_across_splits(split_train_metrics)

    eval_metric_averages, eval_metric_stds = utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=eval_metric_aggregates,
        is_train=False,
        print_results=True)
    utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=train_metric_aggregates,
        is_train=True,
        print_results=True)
    return eval_metric_averages, eval_metric_stds