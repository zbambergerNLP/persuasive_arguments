import preprocessing
import constants
import metrics
import transformers
import os


def fine_tune_on_task(dataset, model, configuration, task_name, is_probing=False, premise_mode=None):
    """

    :param dataset: The dataset on which we fine-tune the given model.
    :param model: A pretrained transformer language model for sequence classification.
    :param configuration: A 'transformers.TrainingArguments' instance.
    :param task_name: One of {'multiclass'}
    :param is_probing:
    :param premise_mode:
    :return:
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
        compute_metrics=metrics_function,
    )
    trainer.train()
    trainer.save_model()
    eval_metrics = trainer.evaluate()
    return trainer, eval_metrics


# def fine_tune_on_downstream_task(dataset_name, model, configuration):
#     """Perform fine-tuning on the downstream binary classification task.
#
#     :param dataset_name: The name of the file in which the downstream dataset is stored.
#     :param model: A pretrained transformer language model for sequence classification.
#     :param configuration: A 'transformers.TrainingArguments' instance.
#     :return: A 2-tuple of the form [trainer, eval_metrics]. The trainer is a 'transformers.Trainer' instance used to
#         fine-tune the model, and the metrics are a dictionary derived from evaluating the model on the verification set.
#     """
#     print(f'Creating downstream dataset...')
#     dataset = preprocessing.get_cmv_downstream_dataset(
#         dataset_name=dataset_name,
#         tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)
#     )
#     dataset = dataset.train_test_split()
#     train_dataset = preprocessing.CMVDataset(dataset[constants.TRAIN])
#     test_dataset = preprocessing.CMVDataset(dataset[constants.TEST])
#     trainer = transformers.Trainer(
#         model=model,
#         args=configuration,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         compute_metrics=metrics.compute_metrics_for_binary_classification,
#     )
#     trainer.train()
#     trainer.save_model()
#     eval_metrics = trainer.evaluate()
#     return trainer, eval_metrics
#
#
# def fine_tune_model_on_premise_mode(current_path, premise_mode, probing_dataset, model, model_configuration):
#     """Perform fine-tuning on the premise mode binary classification task.
#
#     :param current_path: The current working directory. A string.
#     :param premise_mode: A string representing the premise mode towards which the dataset is oriented. For example,
#         if the premise_mode were 'ethos', then positive labels would be premises who's label contains 'ethos'.
#     :param probing_dataset: Either a 'preprocessing.CMVPremiseModes' or a 'preprocessing.CMVDataset' instance. This
#         dataset maps either premises or claims + premises to a binary label corresponding to whether the text's premise
#         is associated with 'premise_mode'.
#     :param model: A pretrained transformer language model for sequence classification.
#     :param model_configuration: A 'transformers.TrainingArguments' instance.
#     :return: A 2-tuple of the form [trainer, eval_metrics]. The trainer is a 'transformers.Trainer' instance used to
#         fine-tune the model, and the metrics are a dictionary derived from evaluating the model on the verification set.
#     """
#     probing_dir_path = os.path.join(current_path, constants.PROBING)
#     if not os.path.exists(probing_dir_path):
#         os.mkdir(probing_dir_path)
#     premise_mode_probing_dir_path = os.path.join(probing_dir_path, constants.PREMISE_DIR_PATH_MAPPING[premise_mode])
#     if not os.path.exists(premise_mode_probing_dir_path):
#         os.mkdir(premise_mode_probing_dir_path)
#     probing_dataset = probing_dataset.train_test_split()
#     train_dataset = preprocessing.CMVDataset(probing_dataset[constants.TRAIN])
#     test_dataset = preprocessing.CMVDataset(probing_dataset[constants.TEST])
#     model_configuration.output_dir = os.path.join(premise_mode_probing_dir_path, "results")
#     model_configuration.logging_dir = os.path.join(premise_mode_probing_dir_path, "log")
#     trainer = transformers.Trainer(
#         model=model,
#         args=model_configuration,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         compute_metrics=metrics.compute_metrics_for_binary_classification,
#     )
#     trainer.train()
#     trainer.save_model()
#     eval_metrics = trainer.evaluate()
#     return trainer, eval_metrics
#
#
# def fine_tune_model_on_multiclass_premise_mode(current_path, probing_dataset, model, model_configuration):
#     """Perform fine-tuning on the premise mode binary classification task.
#
#     :param current_path: The current working directory. A string.
#     :param probing_dataset: Either a 'preprocessing.CMVPremiseModes' or a 'preprocessing.CMVDataset' instance. This
#         dataset maps either premises or claims + premises to a binary label corresponding to whether the text's premise
#         is associated with 'premise_mode'.
#     :param model: A pretrained transformer language model for sequence classification.
#     :param model_configuration: A 'transformers.TrainingArguments' instance.
#     :return: A 2-tuple of the form [trainer, eval_metrics]. The trainer is a 'transformers.Trainer' instance used to
#         fine-tune the model, and the metrics are a dictionary derived from evaluating the model on the verification set.
#     """
#     probing_dir_path = os.path.join(current_path, constants.PROBING)
#     if not os.path.exists(probing_dir_path):
#         os.mkdir(probing_dir_path)
#     multiclass_premise_mode_probing_dir_path = os.path.join(probing_dir_path, constants.MULTICLASS)
#     if not os.path.exists(multiclass_premise_mode_probing_dir_path):
#         os.mkdir(multiclass_premise_mode_probing_dir_path)
#     probing_dataset = probing_dataset.train_test_split()
#     train_dataset = preprocessing.CMVDataset(probing_dataset[constants.TRAIN])
#     test_dataset = preprocessing.CMVDataset(probing_dataset[constants.TEST])
#     model_configuration.output_dir = os.path.join(multiclass_premise_mode_probing_dir_path, "results")
#     model_configuration.logging_dir = os.path.join(multiclass_premise_mode_probing_dir_path, "log")
#     trainer = transformers.Trainer(
#         model=model,
#         args=model_configuration,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         compute_metrics=metrics.compute_metrics_for_multi_class_classification,
#     )
#     trainer.train()
#     trainer.save_model()
#     eval_metrics = trainer.evaluate()
#     return trainer, eval_metrics
#
#
# def fine_tune_model_on_argument_relation_prediction(current_path, probing_dataset, model, model_configuration):
#     """Perform fine-tuning on the premise mode binary classification task.
#
#     :param current_path: The current working directory. A string.
#     :param probing_dataset: Either a 'preprocessing.CMVPremiseModes' or a 'preprocessing.CMVDataset' instance. This
#         dataset maps either premises or claims + premises to a binary label corresponding to whether the text's premise
#         is associated with 'premise_mode'.
#     :param model: A pretrained transformer language model for sequence classification.
#     :param model_configuration: A 'transformers.TrainingArguments' instance.
#     :return: A 2-tuple of the form [trainer, eval_metrics]. The trainer is a 'transformers.Trainer' instance used to
#         fine-tune the model, and the metrics are a dictionary derived from evaluating the model on the verification set.
#     """
#     probing_dir_path = os.path.join(current_path, constants.PROBING)
#     if not os.path.exists(probing_dir_path):
#         os.mkdir(probing_dir_path)
#
#     intra_argument_relation_probing_dir_path = os.path.join(probing_dir_path, constants.INTRA_ARGUMENT_RELATIONS)
#     if not os.path.exists(intra_argument_relation_probing_dir_path):
#         os.mkdir(intra_argument_relation_probing_dir_path)
#
#     probing_dataset = probing_dataset.train_test_split()
#     train_dataset = preprocessing.CMVDataset(probing_dataset[constants.TRAIN])
#     test_dataset = preprocessing.CMVDataset(probing_dataset[constants.TEST])
#     model_configuration.output_dir = os.path.join(intra_argument_relation_probing_dir_path, "results")
#     model_configuration.logging_dir = os.path.join(intra_argument_relation_probing_dir_path, "log")
#     trainer = transformers.Trainer(
#         model=model,
#         args=model_configuration,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         compute_metrics=metrics.compute_metrics_for_binary_classification,
#     )
#     trainer.train()
#     trainer.save_model()
#     eval_metrics = trainer.evaluate()
#     return trainer, eval_metrics
