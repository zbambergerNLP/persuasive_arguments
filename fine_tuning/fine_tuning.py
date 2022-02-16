from __future__ import annotations

import copy
import datasets
import logging
import os
import torch
import transformers
import typing

import preprocessing
import constants
import metrics


class TrainingMetricsCallback(transformers.TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_log(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.log_metrics(self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset,
                metric_key_prefix="train"))
            return control_copy


def fine_tune_on_task(dataset: datasets.Dataset,
                      model: transformers.PreTrainedModel | torch.torch.nn.Module,
                      configuration: transformers.TrainingArguments,
                      task_name: str,
                      is_probing: bool = False,
                      premise_mode: str = None,
                      logger: logging.Logger = None) -> typing.Tuple[transformers.Trainer, dict]:
    """Fine tune a transformer language model on the provided dataset.

    :param dataset: The dataset on which we fine-tune the given model.
    :param model: A pretrained transformer language model for sequence classification.
    :param configuration: A 'transformers.TrainingArguments' instance.
    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}.
    :param is_probing: True if the task on which we fine tune the model is a probing task. False if the task is a
        downstream task.
    :param premise_mode: If the task_name is 'binary_premise_mode_prediction', then this string parameter specifies
        which argument mode dataset we are fine-tuning the model on.
    :param logger: A logging.Logger instance used for logging.
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

    dataset = dataset.train_test_split()
    train_dataset = preprocessing.CMVDataset(dataset[constants.TRAIN])
    test_dataset = preprocessing.CMVDataset(dataset[constants.TEST])

    metrics_function = (
        metrics.compute_metrics_for_multi_class_classification if task_name == constants.MULTICLASS else
        metrics.compute_metrics_for_binary_classification)

    trainer = transformers.Trainer(
        model=model,
        args=configuration,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=metrics_function)
    trainer.add_callback(TrainingMetricsCallback(trainer))

    # Training
    logger.info("*** Train ***")
    train_result = trainer.train()
    training_metrics = train_result.metrics

    trainer.save_model()
    trainer.save_metrics(split=constants.TRAIN, metrics=training_metrics)

    # Evaluation
    logger.info("*** Evaluate ***")
    eval_metrics = trainer.evaluate()

    return trainer, eval_metrics
