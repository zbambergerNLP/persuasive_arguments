import constants
import preprocessing
import probing.models as probing_models

import os
import json
import transformers
import torch
import torch.optim.lr_scheduler as lr_scheduler
import datasets
import numpy as np
import sklearn

SUFFIX = "json"


def save_model_embeddings_on_batch(transformer_model, batch, model_base_file_name, batch_index, probing_dir_path):
    """

    :param transformer_model: A pretrained transformer language model for sequence classification.
    :param batch: A dictionary containing tensors corresponding to model inputs. Specifically, these inputs are
        'input_ids', 'attention_mask', and 'token_type_ids' as well as their respective values.
    :param model_base_file_name: The name of the model as used to construct a filename to save the hidden representation
        of the inputs.
    :param batch_index: The index of the batch being fed into the model from the trainer.
    :param probing_dir_path: The path to the probing directory in this repository.
    """
    model_outputs = transformer_model.forward(
        input_ids=batch[constants.INPUT_IDS],
        attention_mask=batch[constants.ATTENTION_MASK],
        token_type_ids=batch[constants.TOKEN_TYPE_IDS],
        output_hidden_states=True,
    )
    model_embedding_hidden_layer = model_outputs.hidden_states[0][:, 0, :]
    model_file_name = f"{model_base_file_name}_batch_{batch_index + 1}.{SUFFIX}"
    model_file_path = os.path.join(probing_dir_path, model_file_name)
    with open(model_file_path, "w") as f:
        print(f'saving pretrained model embedding with shape {model_embedding_hidden_layer.shape} '
              f'from batch #{batch_index + 1}  to {model_file_path}')
        json.dump(
            {
                'hidden_state': model_embedding_hidden_layer.tolist(),
                'label': batch[constants.LABEL].tolist()
            }, f)


def probe_model_on_premise_mode(mode,
                                dataset,
                                current_path,
                                fine_tuned_model_path,
                                model_checkpoint_name,
                                generate_new_probing_dataset,
                                probing_model,
                                learning_rate,
                                training_batch_size,
                                eval_batch_size):
    """Run the probing objective with a pre-trained LM as well as one fine-tuned on the downstream task.

    :param mode: A string representing the premise mode towards which the dataset is oriented. For example,
        if the mode were 'ethos', then positive labels would be premises who's label contains 'ethos'.
    :param dataset: Either a 'preprocessing.CMVPremiseModes' or a 'preprocessing.CMVDataset' instance. This
        dataset maps either premises or claims + premises to a binary label corresponding to whether the text's premise
        is associated with 'mode'.
    :param current_path: The current working directory. A string.
    :param fine_tuned_model_path: The path to the file containing a saved language model that was fine-tuned on the
        downstream task.
    :param model_checkpoint_name: The name of the model checkpoint to load.
    :param generate_new_probing_dataset: A boolean. True if the user intends to generate a new dictionary mapping
        hidden representations of premises/claims+premises to premise mode.
    :param probing_model: A string representing the model type used for probing. Either 'MLP' or 'logistic_regression'.
    :param learning_rate: A float representing the learning rate used by the optimizer while training the probe.
    :param training_batch_size: The batch size used while training the probe. An integer.
    :param eval_batch_size: The batch size used for probe evaluation. An integer.
    """
    pretrained_probing_model, fine_tuned_probing_mode, probing_eval_metrics = (
        probe_model_with_premise_mode(mode,
                                      dataset,
                                      current_path,
                                      fine_tuned_model_path,
                                      pretrained_checkpoint_name=model_checkpoint_name,
                                      generate_new_probing_dataset=generate_new_probing_dataset,
                                      probing_model=probing_model,
                                      learning_rate=learning_rate,
                                      training_batch_size=training_batch_size,
                                      eval_batch_size=eval_batch_size))
    print(f'{mode} pretrained probe results:')
    print(probing_eval_metrics[constants.PRETRAINED][constants.CONFUSION_MATRIX])
    print(probing_eval_metrics[constants.PRETRAINED][constants.CLASSIFICATION_REPORT])
    print(f'{mode} fine tuned probe results:')
    print(probing_eval_metrics[constants.FINE_TUNED][constants.CONFUSION_MATRIX])
    print(probing_eval_metrics[constants.FINE_TUNED][constants.CLASSIFICATION_REPORT])


def save_hidden_layer_outputs(fine_tuned_model_path,
                              probing_dataset,
                              probing_dir_path,
                              pretrained_checkpoint_name='',
                              num_labels=2):
    """Save hidden layer representations of either premises or claim+premise pairs.

    :param fine_tuned_model_path: The path to the file containing a saved language model that was fine-tuned on the
        downstream task.
    :param probing_dataset: Either a 'preprocessing.CMVPremiseModes' or a 'preprocessing.CMVDataset' instance. This
        dataset maps either premises or claims + premises to a binary label corresponding to whether the text's premise
        is associated with 'premise_mode'.
    :param probing_dir_path: The path to the probing directory in this repository.
    :param pretrained_checkpoint_name: The string name of the pretrained model checkpoint to load.
    :param num_labels: The number of labels for the probing classification problem.
    """
    assert pretrained_checkpoint_name or fine_tuned_model_path, "Hidden layers must be obtained from the model either" \
                                                                " after pretraining or after fine-tuning."
    fine_tuned_base_file_name = 'finetuned_bert_hidden_states'
    pretrained_base_file_name = 'pretrained_bert_hidden_states'

    current_path = os.getcwd()
    if pretrained_checkpoint_name:
        pretrained_model = transformers.BertForSequenceClassification.from_pretrained(pretrained_checkpoint_name,
                                                                                      num_labels=num_labels)
    if fine_tuned_model_path:
        fine_tuned_model = transformers.BertForSequenceClassification.from_pretrained(
            os.path.join(current_path, fine_tuned_model_path),
            num_labels=constants.NUM_LABELS)
    dataloader = torch.utils.data.DataLoader(probing_dataset, batch_size=64)
    for batch_index, batch in enumerate(dataloader):
        save_model_embeddings_on_batch(transformer_model=pretrained_model,
                                       batch=batch,
                                       model_base_file_name=pretrained_base_file_name,
                                       batch_index=batch_index,
                                       probing_dir_path=probing_dir_path)
        save_model_embeddings_on_batch(transformer_model=fine_tuned_model,
                                       batch=batch,
                                       model_base_file_name=fine_tuned_base_file_name,
                                       batch_index=batch_index,
                                       probing_dir_path=probing_dir_path)


def load_hidden_layer_outputs(probing_dir_path):
    """Load hidden layer representations that were generated by 'save_hidden_layer_outputs'.

    :param probing_dir_path: The path to the probing directory in this repository.
    :return: A 4-tuple consisting of ('pretrained_hidden_state_dataset_train', 'pretrained_hidden_state_dataset_test',
                                      'fine_tuned_hidden_state_dataset_train', 'fine_tuned_hidden_state_dataset_test').
            Each of these datasets is an instance of 'preprocessing.CMVPremiseModes'.

    """
    pretrained_hidden_layer_batch_file_names = list(
        filter(
            lambda file_name: file_name.endswith(SUFFIX) and 'pretrained' in file_name,
            os.listdir(probing_dir_path)))
    pretrained_tuned_hidden_layer_batch_file_paths = list(
        map(
            lambda file_name: os.path.join(probing_dir_path, file_name),
            pretrained_hidden_layer_batch_file_names))
    pretrained_hidden_state_batches = datasets.load_dataset(
        SUFFIX, data_files=pretrained_tuned_hidden_layer_batch_file_paths)[constants.TRAIN]
    pretrained_hidden_state_dataset = datasets.Dataset.from_dict({
        constants.HIDDEN_STATE: np.concatenate(pretrained_hidden_state_batches[constants.HIDDEN_STATE]),
        constants.LABEL: np.concatenate(pretrained_hidden_state_batches[constants.LABEL])}).train_test_split()
    pretrained_hidden_state_dataset_train = preprocessing.CMVPremiseModes(
        pretrained_hidden_state_dataset[constants.TRAIN])
    pretrained_hidden_state_dataset_test = preprocessing.CMVPremiseModes(
        pretrained_hidden_state_dataset[constants.TEST])
    fine_tuned_hidden_layer_batch_file_names = list(
        filter(
            lambda file_name: file_name.endswith(SUFFIX) and 'finetuned' in file_name,
            os.listdir(probing_dir_path)))
    fine_tuned_hidden_layer_batch_file_paths = list(
        map(
            lambda file_name: os.path.join(probing_dir_path, file_name),
            fine_tuned_hidden_layer_batch_file_names))
    fine_tuned_hidden_state_batches = datasets.load_dataset(
        SUFFIX, data_files=fine_tuned_hidden_layer_batch_file_paths)[constants.TRAIN]
    fine_tuned_hidden_state_dataset = datasets.Dataset.from_dict({
        constants.HIDDEN_STATE: np.concatenate(fine_tuned_hidden_state_batches[constants.HIDDEN_STATE]),
        constants.LABEL: np.concatenate(fine_tuned_hidden_state_batches[constants.LABEL])
    }).train_test_split()
    fine_tuned_hidden_state_dataset_train = preprocessing.CMVPremiseModes(
        fine_tuned_hidden_state_dataset[constants.TRAIN])
    fine_tuned_hidden_state_dataset_test = preprocessing.CMVPremiseModes(
        fine_tuned_hidden_state_dataset[constants.TEST])
    return (pretrained_hidden_state_dataset_train,
            pretrained_hidden_state_dataset_test,
            fine_tuned_hidden_state_dataset_train,
            fine_tuned_hidden_state_dataset_test)


def probe_model_with_premise_mode(premise_mode,
                                  premise_mode_dataset,
                                  current_path,
                                  fine_tuned_model_path,
                                  pretrained_checkpoint_name,
                                  generate_new_probing_dataset=False,
                                  probing_model=constants.LOGISTIC_REGRESSION,
                                  learning_rate=1e-4,
                                  training_batch_size=16,
                                  eval_batch_size=64,
                                  num_epochs=20, ):
    """

    :param premise_mode: A string representing the premise mode towards which the dataset is oriented. For example,
        if the mode were 'ethos', then positive labels would be premises who's label contains 'ethos'.
    :param premise_mode_dataset: Either a 'preprocessing.CMVPremiseModes' or a 'preprocessing.CMVDataset' instance. This
        dataset maps either premises or claims + premises to a binary label corresponding to whether the text's premise
        is associated with 'premise_mode'.
    :param current_path: The current working directory. A string.
    :param fine_tuned_model_path: The path to the file containing a saved language model that was fine-tuned on the
        downstream task.
    :param pretrained_checkpoint_name: The string name of the pretrained model checkpoint to load.
    :param generate_new_probing_dataset: A boolean. True if the user intends to generate a new dictionary mapping
        hidden representations of premises/claims+premises to premise mode.
    :param probing_model: A string representing the model type used for probing. Either 'MLP' or 'logistic_regression'.
    :param learning_rate: A float representing the learning rate used by the optimizer while training the probe.
    :param training_batch_size: The batch size used while training the probe. An integer.
    :param eval_batch_size: The batch size used for probe evaluation. An integer.
    :param num_epochs: The number of epochs used to train our probing model.
    :return: A 3-tuple consisting of pretrained_probing_model, fine_tuned_probing_model, eval_metrics
    """
    probing_dir_path = os.path.join(current_path, constants.PROBING)
    if not os.path.exists(probing_dir_path):
        print(f'Creating directory: {probing_dir_path}')
        os.mkdir(probing_dir_path)
    premise_mode_probing_dir_path = os.path.join(probing_dir_path, constants.PREMISE_DIR_PATH_MAPPING[premise_mode])
    if not os.path.exists(premise_mode_probing_dir_path):
        print(f'Crating directory: {premise_mode_probing_dir_path}')
        os.mkdir(premise_mode_probing_dir_path)
    if generate_new_probing_dataset:
        save_hidden_layer_outputs(
            fine_tuned_model_path=fine_tuned_model_path,
            pretrained_checkpoint_name=pretrained_checkpoint_name,
            probing_dataset=premise_mode_dataset,
            probing_dir_path=premise_mode_probing_dir_path)
    (pretrained_hidden_state_dataset_train, pretrained_hidden_state_dataset_test, fine_tuned_hidden_state_dataset_train,
     fine_tuned_hidden_state_dataset_test) = (
        load_hidden_layer_outputs(probing_dir_path=premise_mode_probing_dir_path))
    if probing_model == constants.MLP:
        pretrained_probing_model = probing_models.MLP()
        fine_tuned_probing_model = probing_models.MLP()
        loss_function = torch.nn.BCELoss()
        pretrained_optimizer = torch.optim.SGD(pretrained_probing_model.parameters(), lr=learning_rate)
        fine_tuned_optimizer = torch.optim.SGD(fine_tuned_probing_model.parameters(), lr=learning_rate)
        pretrained_scheduler = lr_scheduler.ExponentialLR(pretrained_optimizer, gamma=0.9)
        fine_tuned_scheduler = lr_scheduler.ExponentialLR(fine_tuned_optimizer, gamma=0.9)
        pretrained_train_loader = torch.utils.data.DataLoader(
            pretrained_hidden_state_dataset_train, batch_size=training_batch_size, shuffle=True, num_workers=1)
        pretrained_test_loader = torch.utils.data.DataLoader(
            pretrained_hidden_state_dataset_test, batch_size=training_batch_size, shuffle=True, num_workers=1)
        fine_tuned_train_loader = torch.utils.data.DataLoader(
            fine_tuned_hidden_state_dataset_train, batch_size=training_batch_size, shuffle=True, num_workers=1)
        fine_tuned_test_loader = torch.utils.data.DataLoader(
            fine_tuned_hidden_state_dataset_test, batch_size=eval_batch_size, shuffle=True, num_workers=1)
        fine_tuned_probing_model.train_probe(fine_tuned_train_loader,
                                             fine_tuned_optimizer,
                                             loss_function=loss_function,
                                             num_epochs=num_epochs,
                                             scheduler=fine_tuned_scheduler)
        pretrained_probing_model.train_probe(pretrained_train_loader,
                                             pretrained_optimizer,
                                             loss_function=loss_function,
                                             num_epochs=num_epochs,
                                             scheduler=pretrained_scheduler)
        pretrained_confusion_matrix, pretrained_classification_report = (
            pretrained_probing_model.eval_probe(test_loader=pretrained_test_loader))
        fine_tuned_confusion_matrix, fine_tuned_classification_report = (
            fine_tuned_probing_model.eval_probe(test_loader=fine_tuned_test_loader))
        eval_metrics = {
            constants.PRETRAINED: {
                constants.CONFUSION_MATRIX: pretrained_confusion_matrix,
                constants.CLASSIFICATION_REPORT: pretrained_classification_report
            },
            constants.FINE_TUNED: {
                constants.CONFUSION_MATRIX: fine_tuned_confusion_matrix,
                constants.CLASSIFICATION_REPORT: fine_tuned_classification_report
            }
        }
    elif probing_model == constants.LOGISTIC_REGRESSION:
        pretrained_hidden_states_train = (
            pretrained_hidden_state_dataset_train.cmv_premise_mode_dataset[constants.HIDDEN_STATE]
        )
        fine_tuned_hidden_states_train = (
            fine_tuned_hidden_state_dataset_train.cmv_premise_mode_dataset[constants.HIDDEN_STATE])
        pretrained_targets_train = pretrained_hidden_state_dataset_train.cmv_premise_mode_dataset[constants.LABEL]
        fine_tuned_targets_train = fine_tuned_hidden_state_dataset_train.cmv_premise_mode_dataset[constants.LABEL]
        pretrained_probing_model = (
            sklearn.linear_model.LogisticRegression(random_state=0, multi_class='ovr').fit(
                pretrained_hidden_states_train, pretrained_targets_train))
        fine_tuned_probing_model = (
            sklearn.linear_model.LogisticRegression(random_state=0, multi_class='ovr').fit(
                fine_tuned_hidden_states_train, fine_tuned_targets_train))
        pretrained_hidden_states_eval = (
            pretrained_hidden_state_dataset_test.cmv_premise_mode_dataset[constants.HIDDEN_STATE])
        fine_tuned_hidden_states_eval = (
            fine_tuned_hidden_state_dataset_test.cmv_premise_mode_dataset[constants.HIDDEN_STATE])
        pretrained_targets_eval = pretrained_hidden_state_dataset_test.cmv_premise_mode_dataset[constants.LABEL]
        fine_tuned_targets_eval = fine_tuned_hidden_state_dataset_test.cmv_premise_mode_dataset[constants.LABEL]
        pretrained_preds_eval = pretrained_probing_model.predict(pretrained_hidden_states_eval)
        fine_tuned_preds_eval = fine_tuned_probing_model.predict(fine_tuned_hidden_states_eval)
        pretrained_confusion_matrix = sklearn.metrics.confusion_matrix(pretrained_targets_eval, pretrained_preds_eval)
        fine_tuned_confusion_matrix = sklearn.metrics.confusion_matrix(fine_tuned_targets_eval, fine_tuned_preds_eval)
        pretrained_classification_report = (
            sklearn.metrics.classification_report(pretrained_targets_eval, pretrained_preds_eval))
        fine_tuned_classification_report = (
            sklearn.metrics.classification_report(fine_tuned_targets_eval, fine_tuned_preds_eval))
        eval_metrics = {
            constants.PRETRAINED: {
                constants.CONFUSION_MATRIX: pretrained_confusion_matrix,
                constants.CLASSIFICATION_REPORT: pretrained_classification_report
            },
            constants.FINE_TUNED: {
                constants.CONFUSION_MATRIX: fine_tuned_confusion_matrix,
                constants.CLASSIFICATION_REPORT: fine_tuned_classification_report
            }
        }
    return pretrained_probing_model, fine_tuned_probing_model, eval_metrics
