from __future__ import annotations

import copy
import datasets
import logging
import os
import torch
import transformers
import typing

import utils
import wandb

import constants
import metrics


class TrainingMetricsCallback(transformers.TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        control_copy = copy.deepcopy(control)
        training_metrics = self._trainer.evaluate(
            eval_dataset=self._trainer.train_dataset,
            metric_key_prefix=constants.TRAIN)
        self._trainer.log_metrics(constants.TRAIN, training_metrics)
        self._trainer.save_metrics(constants.TRAIN, training_metrics)
        return control_copy


class ValidationMetricsCallback(transformers.TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_begin(
            self, 
            args: transformers.TrainingArguments, 
            state: transformers.TrainerState, 
            control: transformers.TrainerControl,
            **kwargs) -> transformers.TrainerControl:
        control_copy = copy.deepcopy(control)
        validation_metrics = self._trainer.evaluate(
            eval_dataset=self._trainer.eval_dataset,
            metric_key_prefix=constants.EVAL)
        self._trainer.log_metrics(constants.EVAL, validation_metrics)
        self._trainer.save_metrics(constants.EVAL, validation_metrics)
        return control_copy


def fine_tune_on_task(dataset: datasets.Dataset,
                      model: transformers.PreTrainedModel | torch.torch.nn.Module,
                      configuration: transformers.TrainingArguments,
                      task_name: str,
                      max_num_rounds_no_improvement: int,
                      is_probing: bool = False,
                      premise_mode: str = None,
                      num_cross_validation_splits: int = 5,
                      logger: logging.Logger = None,
                      probing_wandb_entity: str = None) -> typing.Tuple[transformers.Trainer, dict]:
    """Fine tune a transformer language model on the provided dataset.

    :param dataset: The dataset on which we fine-tune the given model.
    :param model: A pretrained transformer language model for sequence classification.
    :param configuration: A 'transformers.TrainingArguments' instance.
    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}.
    :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
        does not increase. If validation accuracy does not increase within this number of loops, we stop training
        early.
    :param is_probing: True if the task on which we fine tune the model is a probing task. False if the task is a
        downstream task.
    :param premise_mode: If the task_name is 'binary_premise_mode_prediction', then this string parameter specifies
        which argument mode dataset we are fine-tuning the model on.
    :param num_cross_validation_splits: An integer that represents the number of partitions formed during k-fold cross
        validation. The validation set size consists of `1 / num_cross_validation_splits` examples from the original
        dataset.
    :param logger: A logging.Logger instance used for logging.
    :param probing_wandb_entity: The wandb entity used to track metrics across training, validation, and test splits.
    :return: A 2-tuple of the form (trainer, eval_metrics). The trainer is a 'transformers.Trainer' instance used to
        fine-tune the model, and the metrics are a dictionary derived from evaluating the model on the verification set.
    """
    if is_probing:
        probing_dir_path = os.path.join(os.getcwd(), constants.PROBING)
        if not os.path.exists(probing_dir_path):
            os.mkdir(probing_dir_path)
        if task_name == constants.BINARY_PREMISE_MODE_PREDICTION:
            target_dir_path = os.path.join(probing_dir_path, constants.PREMISE_DIR_PATH_MAPPING[premise_mode])
        else:
            target_dir_path = os.path.join(probing_dir_path, task_name)
        if not os.path.exists(target_dir_path):
            os.mkdir(target_dir_path)
        configuration.output_dir = os.path.join(target_dir_path, constants.RESULTS)
        configuration.logging_dir = os.path.join(target_dir_path, constants.LOG)

    shard_train_metrics = []
    shard_validation_metrics = []
    shard_test_metrics = []
    shards = [dataset.shard(num_cross_validation_splits, i, contiguous=True)
              for i in range(num_cross_validation_splits)]

    # TODO: Ensure that the model and inputs are loaded to GPU when available.
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for validation_set_index in range(num_cross_validation_splits):
        split_model = copy.deepcopy(model)
        validation_and_test_sets = shards[validation_set_index].train_test_split(test_size=0.5)
        validation_set = validation_and_test_sets[constants.TRAIN].shuffle()
        test_set = validation_and_test_sets[constants.TEST].shuffle()
        training_set = datasets.concatenate_datasets(
            shards[0:validation_set_index] + shards[validation_set_index + 1:]).shuffle()
        run_name = f'Fine-tune BERT on {task_name}, ' \
                   f'Split #{validation_set_index + 1}'
        if premise_mode:
            run_name += f' ({premise_mode})'
        run = wandb.init(
            project="persuasive_arguments",
            entity=probing_wandb_entity,
            reinit=True,
            name=run_name)
        metrics_function = (
            metrics.compute_metrics_for_multi_class_classification if task_name == constants.MULTICLASS else
            metrics.compute_metrics_for_binary_classification)
        trainer = transformers.Trainer(
            model=split_model,
            args=configuration,
            train_dataset=training_set,
            eval_dataset=validation_set,
            compute_metrics=metrics_function,
        )
        trainer.add_callback(TrainingMetricsCallback(trainer))
        trainer.add_callback(ValidationMetricsCallback(trainer))
        trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience=max_num_rounds_no_improvement))

        # Training
        logger.info("*** Train ***")
        trainer.train()
        trainer.save_model()

        # Evaluation
        logger.info("*** Evaluate ***")
        training_metrics = trainer.evaluate(training_set)
        shard_train_metrics.append(training_metrics)
        trainer.log_metrics(split=constants.TRAIN, metrics=training_metrics)

        validation_metrics = trainer.evaluate(validation_set)
        shard_validation_metrics.append(validation_metrics)
        trainer.log_metrics(split=constants.VALIDATION, metrics=validation_metrics)

        test_metrics = trainer.evaluate(test_set)
        shard_test_metrics.append(test_metrics)
        trainer.log_metrics(split=constants.TEST, metrics=test_metrics)

        run.finish()

    validation_metric_aggregates = utils.aggregate_metrics_across_splits(shard_validation_metrics)
    train_metric_aggregates = utils.aggregate_metrics_across_splits(shard_train_metrics)
    test_metric_aggregates = utils.aggregate_metrics_across_splits(shard_test_metrics)
    print(f'\n*** {task_name} {premise_mode if premise_mode else ""} Train Metrics: ***')
    train_metric_averages, train_metric_stds = utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=train_metric_aggregates,
        split_name=constants.TRAIN,
        print_results=True)
    print(f'\n*** {task_name} {premise_mode if premise_mode else ""} Validation Metrics: ***')
    validation_metric_averages, validation_metric_stds = utils.get_metrics_avg_and_std_across_splits(
        metric_aggregates=validation_metric_aggregates,
        split_name=constants.VALIDATION,
        print_results=True)
    print(f'\n*** {task_name} {premise_mode if premise_mode else ""} Test Metrics: ***')
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

    return trainer, split_metrics
