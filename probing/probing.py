from __future__ import annotations

import wandb

import constants
import preprocessing
import probing.models as probing_models

import datasets
import os
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import torch
import torch.optim.lr_scheduler as lr_scheduler
import transformers
import typing
import utils


def save_model_embeddings_on_batch(transformer_model: transformers.PreTrainedModel,
                                   batch: typing.Mapping[str, torch.tensor],
                                   model_base_file_name: str,
                                   batch_index: int,
                                   probing_dir_path: str) -> str:
    """Run a batch of inputs from the probing dataset through the model, and save their outputted hidden
    representations.

    :param transformer_model: A pretrained transformer language model for sequence classification.
    :param batch: A dictionary containing tensors corresponding to model inputs. Specifically, these inputs are
        'input_ids', 'attention_mask', and 'token_type_ids' as well as their respective values.
    :param model_base_file_name: The name of the model as used to construct a filename to save the hidden representation
        of the inputs.
    :param batch_index: The index of the batch being fed into the model from the trainer.
    :param probing_dir_path: The path to the probing directory in this repository.
    :return: The string path of the file we saved containing hidden states and label for probing examples.
    """
    model_outputs = transformer_model.forward(
        input_ids=batch[constants.INPUT_IDS],
        attention_mask=batch[constants.ATTENTION_MASK],
        token_type_ids=batch[constants.TOKEN_TYPE_IDS],
        output_hidden_states=True,
    )
    model_embedding_hidden_state = model_outputs.hidden_states[-1][:, 0, :]
    model_file_name = f"{model_base_file_name}_batch_{batch_index + 1}.{constants.PARQUET}"
    model_file_path = os.path.join(probing_dir_path, model_file_name)
    table = pa.table(
        data=[
            pa.array(model_embedding_hidden_state.tolist()),
            pa.array(batch[constants.LABEL].tolist())],
        names=[constants.HIDDEN_STATE, constants.LABEL],
    )
    print(f'Saving pretrained model embedding with shape {model_embedding_hidden_state.shape} '
          f'from batch #{batch_index + 1}  to {model_file_path}')
    pq.write_table(table, model_file_path)
    return model_file_path


def save_hidden_state_outputs(fine_tuned_model_path: str,
                              probing_dataset: preprocessing.CMVProbingDataset,
                              probing_dir_path: str,
                              pretrained_checkpoint_name: str,
                              num_labels: int = 2,
                              hidden_states_batch_size: int = 64) -> (
        typing.Tuple[typing.Sequence[str], typing.Sequence[str]]):
    """Save hidden layer representations of either premises or claim+premise pairs.

    :param fine_tuned_model_path: The path to the file containing a saved language model that was fine-tuned on the
        downstream task.
    :param probing_dataset: A 'preprocessing.CMVDataset' instance. This dataset maps either premises or
        claims + premises to a binary label corresponding to whether the text's premise is associated with
        'premise_mode'.
    :param probing_dir_path: The path to the probing directory in this repository.
    :param pretrained_checkpoint_name: The string name of the pretrained model checkpoint to load.
    :param num_labels: The number of labels for the probing classification problem.
    :param hidden_states_batch_size: The number of probing examples (hidden states and labels) stored in a single file.
    :return: A 2-tuple containing the paths of the produced files.
        pretrained_hidden_state_files: The paths of the probing example files (representing a batch) produced by a
            pretrained transformer model.
        fine_tuned_hidden_state_files: The paths of the probing example files (representing a batch) produced by a
            fine-tuned transformer model.
    """
    assert pretrained_checkpoint_name or fine_tuned_model_path, "Hidden layers must be obtained from the model either" \
                                                                " after pretraining or after fine-tuning."
    # Remove any leftover saved files.
    if os.path.exists(probing_dir_path):
        print(f'Re-initializing {probing_dir_path}...')
        shutil.rmtree(probing_dir_path)
        os.mkdir(probing_dir_path)

    multiclass_prefix = 'multiclass'
    fine_tuned_base_file_name = 'finetuned_bert_hidden_states'
    pretrained_base_file_name = 'pretrained_bert_hidden_states'

    if num_labels > constants.NUM_LABELS:
        pretrained_base_file_name = multiclass_prefix + "_" + pretrained_base_file_name

    pretrained_model = transformers.BertForSequenceClassification.from_pretrained(pretrained_checkpoint_name,
                                                                                  num_labels=num_labels)

    # We do not yet support multiclass probing on models that are fine-tuned on binary classification.
    if num_labels == constants.NUM_LABELS:
        fine_tuned_model = transformers.BertForSequenceClassification.from_pretrained(
            os.path.join(os.getcwd(), fine_tuned_model_path), num_labels=num_labels)

    dataloader = torch.utils.data.DataLoader(probing_dataset, batch_size=hidden_states_batch_size)
    pretrained_hidden_state_files = []
    fine_tuned_hidden_state_files = []
    for batch_index, batch in enumerate(dataloader):
        pretrained_hidden_state_files.append(
            save_model_embeddings_on_batch(
                transformer_model=pretrained_model,
                batch=batch,
                model_base_file_name=pretrained_base_file_name,
                batch_index=batch_index,
                probing_dir_path=probing_dir_path))
        if num_labels == constants.NUM_LABELS:
            fine_tuned_hidden_state_files.append(
                save_model_embeddings_on_batch(
                    transformer_model=fine_tuned_model,
                    batch=batch,
                    model_base_file_name=fine_tuned_base_file_name,
                    batch_index=batch_index,
                    probing_dir_path=probing_dir_path))
    return pretrained_hidden_state_files, fine_tuned_hidden_state_files


def create_hidden_state_dataset_from_files(
        key_phrase: str,
        probing_file_paths: typing.Collection[str] = None,
        probing_dir_path: str = None) -> datasets.Dataset:
    """Create training and validation datasets by loading hidden states stored locally.

    :param probing_dir_path: The path to the probing directory in this repository.
    :param probing_file_paths: A list of paths names in which the probing dataset batches are stored.
    :param key_phrase: A string entry in the following set {'pretrained', 'finetuned', 'multiclass'}.
    :return: A datasets.Dataset instance mapping hidden representations of probing inputs to their corresponding labels.
    """
    if not probing_file_paths:
        hidden_state_batch_file_names = list(
            filter(
                lambda file_name: file_name.endswith(constants.PARQUET) and key_phrase in file_name,
                os.listdir(probing_dir_path)))
        probing_file_paths = list(
            map(
                lambda file_name: os.path.join(probing_dir_path, file_name),
                hidden_state_batch_file_names))
    hidden_state_batches = datasets.load_dataset(
        constants.PARQUET, data_files=probing_file_paths)[constants.TRAIN]
    return datasets.Dataset.from_dict(
        {
            constants.HIDDEN_STATE: hidden_state_batches[constants.HIDDEN_STATE],
            constants.LABEL: hidden_state_batches[constants.LABEL]
        })


def load_hidden_state_outputs(probing_dir_path: str = None,
                              probing_file_paths: typing.Collection[str] = None,
                              pretrained: bool = False,
                              fine_tuned: bool = False,
                              multiclass: bool = False) -> (
        typing.Mapping[str, datasets.Dataset]):
    """Load hidden layer representations that were generated by 'save_hidden_state_outputs'.

    :param probing_dir_path: The path to the probing directory in this repository.
    :param probing_file_paths: A list of file paths in which the probing dataset batches are stored.
    :param pretrained: True if we would like to load a probing dataset produced by a pre-trained model. False otherwise.
    :param fine_tuned: True if we would like to load a probing dataset produced by a fine-tuned model. False otherwise.
    :param multiclass: True if we would like to load a probing dataset produced by a multiclass pre-trained model.
        Currently, we do not support multiclass fine-tuned models as the models were fine-tuned on a binary prediction
        task, whereas the multiclass probing task consists of 9 labels.

    :return: A dictionary mapping the probing model type to the corresponding dataset.Dataset instance. Each such
        datasets maps a hidden representation of textual inputs to the appropriate label in the probing task
        (e.g., binary premise mode prediction).

        The keys of the resulting dictionary are a subset of {'pretrained', 'finetuned', 'multiclass'}. Values for each
        key consist of the corresponding dataset.Dataset instance."""

    assert pretrained or fine_tuned or multiclass, "At least one model mode must be selected. Please assigned the " \
                                                   "value 'True' to one of the 'pretrained', 'finetuned', or " \
                                                   "'multiclass' parameters."
    assert probing_dir_path or probing_file_paths, "One of `probing_dir_path` or `probing_file_paths` must be " \
                                                   "supplied as a parameter to this function."
    result = {}
    if pretrained:
        print("Loading pretrained hidden states...")
        pretrained_dataset = (
            create_hidden_state_dataset_from_files(key_phrase=constants.PRETRAINED,
                                                   probing_dir_path=probing_dir_path,
                                                   probing_file_paths=probing_file_paths))
        result[constants.PRETRAINED] = pretrained_dataset
    if fine_tuned:
        print("Loading fine tuned hidden states...")
        fine_tuned_dataset = (
            create_hidden_state_dataset_from_files(key_phrase=constants.FINE_TUNED,
                                                   probing_dir_path=probing_dir_path,
                                                   probing_file_paths=probing_file_paths))
        result[constants.FINE_TUNED] = fine_tuned_dataset
    if multiclass:
        print("Loading multi-class hidden states")
        multiclass_dataset = (
            create_hidden_state_dataset_from_files(key_phrase=constants.MULTICLASS,
                                                   probing_dir_path=probing_dir_path,
                                                   probing_file_paths=probing_file_paths))
        result[constants.MULTICLASS] = multiclass_dataset
    return result


def probe_model_on_task(probing_dataset: preprocessing.CMVProbingDataset,
                        probing_model_name: str,
                        generate_new_hidden_state_dataset: bool,
                        task_name: str,
                        num_cross_validation_splits: int,
                        probe_learning_rate: float,
                        probe_training_batch_size: int,
                        probe_eval_batch_size: int,
                        probe_num_epochs: int,
                        probe_optimizer: str = 'sgd',
                        probing_wandb_entity: str = None,
                        pretrained_checkpoint_name: str = None,
                        fine_tuned_model_path: str = None,
                        probe_optimizer_scheduler_gamma: float = None,
                        premise_mode: str = None) -> (
        typing.Tuple[
            typing.Mapping[str, typing.Sequence[torch.Module]],
            typing.Mapping[str, typing.Sequence[typing.Mapping[str, float]]],
            typing.Mapping[str, typing.Sequence[typing.Mapping[str, float]]]]):
    """

    :param probing_dataset: A 'preprocessing.CMVProbingDataset' instance. This dataset maps either premises or
        claims + premises to a binary label corresponding to whether the text's premise is associated with
        'premise_mode'.
    :param probing_model_name: A string representing the model type used for probing. Either 'MLP' or
        'logistic_regression'.
    :param generate_new_hidden_state_dataset: A boolean. True if the user intends to generate a new dictionary mapping
        hidden representations of premises/claims+premises to premise mode.
    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}.
    :param num_cross_validation_splits: An integer that represents the number of partitions formed during k-fold cross
        validation. The validation set size consists of `1 / num_cross_validation_splits` examples from the original
        dataset.
    :param probing_wandb_entity: The wandb entity used to track training.
    :param pretrained_checkpoint_name: The string name of the pretrained model checkpoint to load.
    :param fine_tuned_model_path: The path to the file containing a saved language model that was fine-tuned on the
        downstream task.
    :param mlp_learning_rate: A float representing the learning rate used by the optimizer while training the probe.
    :param mlp_training_batch_size: The batch size used while training the probe. An integer.
    :param mlp_eval_batch_size: The batch size used for probe evaluation. An integer.
    :param mlp_num_epochs: The number of training epochs used to train the probe if using a MLP.
    :param mlp_optimizer_scheduler_gamma: Decays the learning rate of each parameter group by gamma every epoch.
    :param premise_mode: A string representing the premise mode towards which the dataset is oriented. For example,
        if the mode were 'ethos', then positive labels would be premises who's label contains 'ethos'.
    :return: A tuple consisting of three entries:
        pretrained_probing_model: A probing model trained on embeddings produced by a pre-trained transformer model.
        fine_tuned_probing_model: A probing model trained on embeddings produced by a pre-trained and fine-tuned
            transformer model.
        eval_metrics: A dictionary mapping string model names ("pretrained", "finetuned" or "multiclass") to a
            dictionary of metrics (string keys to mapping to metric values).
    """
    probing_dir_path = os.path.join(os.getcwd(), constants.PROBING)
    utils.ensure_dir_exists(probing_dir_path)

    if task_name == constants.BINARY_PREMISE_MODE_PREDICTION:
        assert premise_mode, 'When probing with the binary premise mode prediction task, you must run this function' \
                             'for each premise mode, and specify which one you are currently running within the ' \
                             '"premise_mode" parameter.'
        task_probing_dir_path = os.path.join(probing_dir_path, constants.PREMISE_DIR_PATH_MAPPING[premise_mode])
    else:
        assert task_name in [constants.BINARY_PREMISE_MODE_PREDICTION,
                             constants.MULTICLASS,
                             constants.INTRA_ARGUMENT_RELATIONS], f"{task_name} is an unsupported probing task."
        task_probing_dir_path = os.path.join(probing_dir_path, task_name)

    utils.ensure_dir_exists(task_probing_dir_path)

    if generate_new_hidden_state_dataset:
        num_labels = (
            len(constants.PREMISE_MODE_TO_INT) if task_name == constants.MULTICLASS else constants.NUM_LABELS)
        # TODO: Use the returned outputs from 'save_hidden_state_outputs' within 'load_hidden_state_outputs'.
        save_hidden_state_outputs(
            fine_tuned_model_path=fine_tuned_model_path,
            pretrained_checkpoint_name=pretrained_checkpoint_name,
            probing_dataset=probing_dataset,
            probing_dir_path=task_probing_dir_path,
            num_labels=num_labels)

    if task_name == constants.MULTICLASS:
        hidden_state_datasets = (
            load_hidden_state_outputs(task_probing_dir_path, pretrained=False, fine_tuned=False, multiclass=True))
    else:
        hidden_state_datasets = (
            load_hidden_state_outputs(task_probing_dir_path, pretrained=True, fine_tuned=True, multiclass=False))

    all_models = {constants.PRETRAINED: [],
              constants.FINE_TUNED: [],
              constants.MULTICLASS: []}
    all_eval_metrics = {constants.PRETRAINED: [],
                    constants.FINE_TUNED: [],
                    constants.MULTICLASS: []}
    all_train_metrics = {constants.PRETRAINED: [],
                     constants.FINE_TUNED: [],
                     constants.MULTICLASS: []}
    for base_model_type, dataset in hidden_state_datasets.items():
        shards = [dataset.shard(num_cross_validation_splits, i, contiguous=True)
                  for i in range(num_cross_validation_splits)]
        for validation_set_index in range(num_cross_validation_splits):
            num_labels = (
                len(constants.PREMISE_MODE_TO_INT) if base_model_type == constants.MULTICLASS else constants.NUM_LABELS)
            validation_set = shards[validation_set_index].shuffle()
            training_set = datasets.concatenate_datasets(
                shards[0:validation_set_index] + shards[validation_set_index+1:]).shuffle()
            run_name = f'Probe {probing_model_name} on {task_name}, ' \
                       f'base model type: {base_model_type}, ' \
                       f'Split #{validation_set_index + 1}'
            if premise_mode:
                run_name += f' ({premise_mode})'
            if probing_wandb_entity:
                run = wandb.init(
                    project="persuasive_arguments",
                    entity=probing_wandb_entity,
                    reinit=True,
                    name=run_name)
            if probing_model_name == constants.MLP:
                probing_model = probing_models.MLP(num_labels=num_labels)
            else:
                probing_model = probing_models.LogisticRegression(num_labels=num_labels)
            loss_function = torch.nn.BCELoss() if num_labels == constants.NUM_LABELS else torch.nn.CrossEntropyLoss()

            # Initialize the optimizer.
            if probe_optimizer == 'sgd':
                optimizer = torch.optim.SGD(probing_model.parameters(), lr=probe_learning_rate)
            elif probe_optimizer == 'adam':
                optimizer = torch.optim.Adam(probing_model.parameters(), lr=probe_learning_rate)

            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=probe_optimizer_scheduler_gamma)
            train_loader = torch.utils.data.DataLoader(
                preprocessing.CMVProbingDataset(training_set),
                batch_size=probe_training_batch_size,
                shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                preprocessing.CMVProbingDataset(validation_set),
                batch_size=probe_eval_batch_size,
                shuffle=True)

            trained_model = probing_models.train_probe(probing_model=probing_model,
                                                       train_loader=train_loader,
                                                       optimizer=optimizer,
                                                       num_labels=num_labels,
                                                       loss_function=loss_function,
                                                       num_epochs=probe_num_epochs,
                                                       scheduler=scheduler)
            train_metrics = probing_models.eval_probe(probing_model=trained_model,
                                                      num_labels=num_labels,
                                                      test_loader=train_loader)
            eval_metrics = probing_models.eval_probe(probing_model=trained_model,
                                                     num_labels=num_labels,
                                                     test_loader=test_loader)
            all_models[base_model_type].append(trained_model)
            all_train_metrics[base_model_type].append(train_metrics)
            all_eval_metrics[base_model_type].append(eval_metrics)
            if probing_wandb_entity:
                run.finish()
    return all_models, all_train_metrics, all_eval_metrics
