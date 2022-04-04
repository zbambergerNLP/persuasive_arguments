import typing

import sklearn
import torch
import pandas as pd
import numpy as np
import wandb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

import constants
import data_loaders
import models
import utils


def create_combined_column(task_name: str, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}.
    :param corpus_df: A pandas Dataframe instances with the following columns depending on the task:
        binary_cmv_delta_prediction columns: {OP_COMMENT, REPLY, LABEL}
            * OP_COMMENT and REPLY are text entries.
            * LABEL is a binary label that is True if the reply was given a "delta".
        binary_premise_mode_prediction OR multiclass columns: {CLAIM_TEXT, PREMISE_TEXT, PREMISE_MODE}
            * PREMISE_MODE the consists of text entries
            * PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
            * LABEL is a binary label if `multiclass` is False. If `multiclass` is False, then True examples are ones
              whose premises consist of the argumentative mode (ethos/logos/pathos) currently under study. If
              `multiclass` is True, then the label is an integer representing the exact combination of premise modes
              which the premise exhibits.
        intra_argument_relations columns: {SENTENCE_1, SENTENCE_2, LABEL}.
            * SENTENCE_1 and SENTENCE_2 both consist of argument prepositions (either claims or premises).
            * LABEL is a binary label which signifies whether there is a relation starting from SENTENCE_1 and directed
              towards SENTENCE_2. The label of a given example is True if such a relation exists, and False otherwise.
    :return: A corpus_df as above with an additional column -- 'combined'. The values in this column correspond to the
        concatenation of the first textual input with the second textual input.
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

    :param corpus_df: A pandas Dataframe instances with the following columns depending on the task:
        binary_cmv_delta_prediction columns: {OP_COMMENT, REPLY, LABEL}
            * OP_COMMENT and REPLY are text entries.
            * LABEL is a binary label that is True if the reply was given a "delta".
        binary_premise_mode_prediction OR multiclass columns: {CLAIM_TEXT, PREMISE_TEXT, PREMISE_MODE}
            * PREMISE_MODE the consists of text entries
            * PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
            * LABEL is a binary label if `multiclass` is False. If `multiclass` is False, then True examples are ones
              whose premises consist of the argumentative mode (ethos/logos/pathos) currently under study. If
              `multiclass` is True, then the label is an integer representing the exact combination of premise modes
              which the premise exhibits.
        intra_argument_relations columns: {SENTENCE_1, SENTENCE_2, LABEL}.
            * SENTENCE_1 and SENTENCE_2 both consist of argument prepositions (either claims or premises).
            * LABEL is a binary label which signifies whether there is a relation starting from SENTENCE_1 and directed
              towards SENTENCE_2. The label of a given example is True if such a relation exists, and False otherwise.
        This dataframe also consists of an additional column -- 'combined'. The values in this column correspond to the
        concatenation of the first textual input with the second textual input.
    :param premise_mode: A string in the set {'ethos', 'logos', 'pathos'}.
    :return: A 2-tuples of tensors where the first represents the set of example features, and the latter represents the
        corresponding set of example labels.
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
                        metric_for_early_stopping: str,
                        max_num_rounds_no_improvement: int,
                        num_cross_validation_splits: int = 5,
                        premise_mode: str = None,
                        num_epochs: int = 100,
                        batch_size: int = 16,
                        learning_rate: float = 1e-3,
                        optimizer_gamma: float = 0.9,
                        probing_wandb_entity: str = 'zbamberger') -> (
        typing.Mapping[str, typing.Mapping[str, typing.Mapping[str, float]]]):
    """

<<<<<<< HEAD
    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}.
    :param corpus_df: A pandas Dataframe instances with the following columns depending on the task:
        binary_cmv_delta_prediction columns: {OP_COMMENT, REPLY, LABEL}
            * OP_COMMENT and REPLY are text entries.
            * LABEL is a binary label that is True if the reply was given a "delta".
        binary_premise_mode_prediction OR multiclass columns: {CLAIM_TEXT, PREMISE_TEXT, PREMISE_MODE}
            * PREMISE_MODE the consists of text entries
            * PREMISE_MODE is en entry in {LOGOS, ETHOS, PATHOS}.
            * LABEL is a binary label if `multiclass` is False. If `multiclass` is False, then True examples are ones
              whose premises consist of the argumentative mode (ethos/logos/pathos) currently under study. If
              `multiclass` is True, then the label is an integer representing the exact combination of premise modes
              which the premise exhibits.
        intra_argument_relations columns: {SENTENCE_1, SENTENCE_2, LABEL}.
            * SENTENCE_1 and SENTENCE_2 both consist of argument prepositions (either claims or premises).
            * LABEL is a binary label which signifies whether there is a relation starting from SENTENCE_1 and directed
              towards SENTENCE_2. The label of a given example is True if such a relation exists, and False otherwise.
    :param metric_for_early_stopping: The metric used to determine whether or not to stop early. If the metric of
        interest does not improve within `max_num_rounds_no_improvement`, then we stop early.
    :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
        does not increase. If validation accuracy does not increase within this number of loops, we stop training
        early.
    :param num_cross_validation_splits: An integer that represents the number of partitions formed during k-fold cross
        validation. The validation set size consists of `1 / num_cross_validation_splits` examples from the original
        dataset.
    :param premise_mode: If the task_name is 'binary_premise_mode_prediction', then this string parameter specifies
        which argument mode dataset we are fine-tuning the model on.
    :param num_epochs: The number of training epochs used to train the baseline model.
    :param batch_size: The batch size used while training the baseline model. An integer.
    :param learning_rate: A float representing the learning rate used by the optimizer while training the baseline
        model.
    :param optimizer_gamma: Decays the learning rate of each parameter group by gamma every epoch.
    :param probing_wandb_entity: The wandb entity used to track metrics across training, validation, and test splits.
    :return: A 2-tuple of dictionaries where the first represents metric values, and the second represents metric
        standard deviations.
    """
    corpus_df = create_combined_column(task_name=task_name, corpus_df=corpus_df)
    X, y = extract_bigram_features(corpus_df=corpus_df, premise_mode=premise_mode)
    kf = KFold(n_splits=num_cross_validation_splits)
    num_labels = utils.get_num_labels(task_name=task_name)
    split_train_metrics = []
    split_validation_metrics = []
    split_test_metrics = []
    for validation_set_index, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_validation_and_test = np.array(X)[train_index.astype(int)], np.array(X)[test_index.astype(int)]
        y_train, y_validation_and_test = np.array(y)[train_index.astype(int)], np.array(y)[test_index.astype(int)]
        X_val, X_test, y_val, y_test = (
            sklearn.model_selection.train_test_split(X_validation_and_test, y_validation_and_test, test_size=0.5))
        run_name = f'Baseline experiment: {task_name}, ' \
                   f'Logistic regression over bigram features, ' \
                   f'Split #{validation_set_index + 1}'
        if premise_mode:
            run_name += f' ({premise_mode})'
        run = wandb.init(
            project="persuasive_arguments",
            entity=probing_wandb_entity,
            reinit=True,
            name=run_name)
        logistic_regression = models.BaselineLogisticRegression(num_features=X_train.shape[1], num_labels=num_labels)
        optimizer = torch.optim.Adam(logistic_regression.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=optimizer_gamma)
        logistic_regression.fit(
            train_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_train, y_train),
                batch_size=batch_size,
                shuffle=True),
            validation_loader=torch.utils.data.DataLoader(
              data_loaders.BaselineLoader(X_val, y_val),
              batch_size=batch_size,
              shuffle=True),
            num_labels=num_labels,
            loss_function=torch.nn.BCELoss() if num_labels == 2 else torch.nn.CrossEntropyLoss(),
            num_epochs=num_epochs,
            max_num_rounds_no_improvement=max_num_rounds_no_improvement,
            metric_for_early_stopping=metric_for_early_stopping,
            optimizer=optimizer,
            scheduler=scheduler)

        train_metrics = logistic_regression.evaluate(
            test_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_train, y_train),
                batch_size=batch_size,
                shuffle=True,
            ),
            num_labels=num_labels,
            split_name=constants.TRAIN,
        )
        split_train_metrics.append(train_metrics)

        validation_metrics = logistic_regression.evaluate(
            test_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_val, y_val),
                batch_size=batch_size,
                shuffle=True,
            ),
            num_labels=num_labels,
            split_name=constants.VALIDATION,
        )
        split_validation_metrics.append(validation_metrics)

        test_metrics = logistic_regression.evaluate(
            test_loader=torch.utils.data.DataLoader(
                data_loaders.BaselineLoader(X_test, y_test),
                batch_size=batch_size,
                shuffle=True,
            ),
            num_labels=num_labels,
            split_name=constants.TEST,
        )
        split_test_metrics.append(test_metrics)
        run.finish()

    train_metric_aggregates = utils.aggregate_metrics_across_splits(split_train_metrics)
    validation_metric_aggregates = utils.aggregate_metrics_across_splits(split_validation_metrics)
    test_metric_aggregates = utils.aggregate_metrics_across_splits(split_test_metrics)

    print(f'\n*** {task_name} {f"({premise_mode})" if premise_mode else ""} baseline training metrics: ***')
    train_metric_averages, train_metric_stds = utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=train_metric_aggregates,
        split_name=constants.TRAIN,
        print_results=True)

    print(f'\n*** {task_name} {f"({premise_mode})" if premise_mode else ""} baseline validation metrics: ***')
    validation_metric_averages, validation_metric_stds = utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=validation_metric_aggregates,
        split_name=constants.VALIDATION,
        print_results=True)

    print(f'\n*** {task_name} {f"({premise_mode})" if premise_mode else ""} baseline test metrics: ***')
    test_metric_averages, test_metric_stds = utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=test_metric_aggregates,
        split_name=constants.TEST,
        print_results=True)

    split_metrics = {
        constants.TRAIN: {
            "averages": train_metric_averages,
            "stds": train_metric_stds
        },
        constants.VALIDATION: {
            "averages": validation_metric_averages,
            "stds": validation_metric_stds,
        },
        constants.TEST: {
            "averages": test_metric_averages,
            "stds": test_metric_stds,
        }
    }

    return split_metrics
