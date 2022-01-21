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
                                eval_batch_size,
                                num_epochs):
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
    :param num_epochs: The number of training epochs used to train the probe if using a MLP.
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
                                      eval_batch_size=eval_batch_size,
                                      num_epochs=num_epochs))
    print(f'{mode} pretrained probe results:')
    print(probing_eval_metrics[constants.PRETRAINED][constants.CONFUSION_MATRIX])
    print(probing_eval_metrics[constants.PRETRAINED][constants.CLASSIFICATION_REPORT])
    print(f'{mode} fine tuned probe results:')
    print(probing_eval_metrics[constants.FINE_TUNED][constants.CONFUSION_MATRIX])
    print(probing_eval_metrics[constants.FINE_TUNED][constants.CLASSIFICATION_REPORT])


def save_hidden_layer_outputs(fine_tuned_model_path,
                              probing_dataset,
                              probing_dir_path,
                              pretrained_checkpoint_name,
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
    multiclass_prefix = 'multiclass'
    fine_tuned_base_file_name = 'finetuned_bert_hidden_states'
    pretrained_base_file_name = 'pretrained_bert_hidden_states'

    current_path = os.getcwd()
    pretrained_base_file_name = multiclass_prefix + "_" + pretrained_base_file_name
    pretrained_model = transformers.BertForSequenceClassification.from_pretrained(pretrained_checkpoint_name,
                                                                                  num_labels=num_labels)

    # We do not yet support multiclass probing on models that are fine-tuned on binary classification.
    if num_labels != len(constants.PREMISE_MODE_TO_INT):
        fine_tuned_base_file_name = multiclass_prefix + "_" + fine_tuned_base_file_name
        fine_tuned_model = transformers.BertForSequenceClassification.from_pretrained(
            os.path.join(current_path, fine_tuned_model_path), num_labels=num_labels)

    # TODO(zbamberger): Make batch_size a parameter.
    dataloader = torch.utils.data.DataLoader(probing_dataset, batch_size=64)
    for batch_index, batch in enumerate(dataloader):
        save_model_embeddings_on_batch(transformer_model=pretrained_model,
                                       batch=batch,
                                       model_base_file_name=pretrained_base_file_name,
                                       batch_index=batch_index,
                                       probing_dir_path=probing_dir_path)
        if num_labels != len(constants.PREMISE_MODE_TO_INT):
            save_model_embeddings_on_batch(transformer_model=fine_tuned_model,
                                           batch=batch,
                                           model_base_file_name=fine_tuned_base_file_name,
                                           batch_index=batch_index,
                                           probing_dir_path=probing_dir_path)


def get_file_paths(probing_dir_path, key_phrase):
    """

    :param probing_dir_path:
    :param key_phrase:
    :return:
    """
    hidden_layer_batch_file_names = list(
        filter(
            lambda file_name: file_name.endswith(SUFFIX) and key_phrase in file_name,
            os.listdir(probing_dir_path)))
    hidden_layer_batch_file_paths = list(
        map(
            lambda file_name: os.path.join(probing_dir_path, file_name),
            hidden_layer_batch_file_names))
    hidden_state_batches = datasets.load_dataset(
        SUFFIX, data_files=hidden_layer_batch_file_paths)[constants.TRAIN]
    hidden_state_dataset = datasets.Dataset.from_dict({
        constants.HIDDEN_STATE: np.concatenate(hidden_state_batches[constants.HIDDEN_STATE]),
        constants.LABEL: np.concatenate(hidden_state_batches[constants.LABEL])}).train_test_split()
    hidden_state_dataset_train = preprocessing.CMVPremiseModes(hidden_state_dataset[constants.TRAIN])
    hidden_state_dataset_test = preprocessing.CMVPremiseModes(hidden_state_dataset[constants.TEST])
    return hidden_state_dataset_train, hidden_state_dataset_test


def load_hidden_layer_outputs(probing_dir_path,
                              pretrained=False,
                              fine_tuned=False,
                              multiclass=False):
    """Load hidden layer representations that were generated by 'save_hidden_layer_outputs'.

    :param probing_dir_path: The path to the probing directory in this repository.
    :param pretrained: True if we would like to load a probing dataset produced by a pre-trained model. False otherwise.
    :param fine_tuned: True if we would like to load a probing dataset produced by a fine-tuned model. False otherwise.
    :param multiclass: True if we would like to load a probing dataset produced by a multiclass pre-trained model.
        Currently, we do not support multiclass fine-tuned models as the models were fine-tuned on a binary prediction
        task, whereas the multiclass probing task consists of 9 labels.

    :return: A dictionary mapping the probing model type to the train and test datasets it produces. Concretely,
        each such datasets maps a hidden representation of textual inputs to the appropriate label in the probing task
        (binary predection or multi-class prediction of premise mode).

        The keys of the resulting dictionary are a subset of {'pretrained', 'finetuned', 'multiclass'}. Values for each
        key consist of additional dictionaries. These inner dictionaries consist of the keys 'train' and 'test'. The
        values of this inner dictionary are the training set, test set respectively. Each of these  datasets is an
        instance of 'preprocessing.CMVPremiseModes'."""
    assert pretrained or fine_tuned or multiclass, "At least one model mode must be selected. Please assigned the "\
                                                   "value 'True' to one of the 'pretrained', 'finetuned', or "\
                                                   "'multiclass' parameters."
    result = {}
    if pretrained:
        pretrained_hidden_state_dataset_train, pretrained_hidden_state_dataset_test = (
            get_file_paths(probing_dir_path, key_phrase=constants.PRETRAINED))
        result[constants.PRETRAINED] = {constants.TRAIN: pretrained_hidden_state_dataset_train,
                                        constants.TEST: pretrained_hidden_state_dataset_test}
    if fine_tuned:
        fine_tuned_hidden_state_dataset_train, fine_tuned_hidden_state_dataset_test = (
            get_file_paths(probing_dir_path, key_phrase=constants.FINE_TUNED))
        result[constants.FINE_TUNED] = {constants.TRAIN: fine_tuned_hidden_state_dataset_train,
                                        constants.TEST: fine_tuned_hidden_state_dataset_test}
    if multiclass:
        multiclass_pretrained_hidden_state_dataset_train, multiclass_pretrained_hidden_state_dataset_test = (
            get_file_paths(probing_dir_path, key_phrase=constants.MULTICLASS))
        result[constants.MULTICLASS] = {constants.TRAIN: multiclass_pretrained_hidden_state_dataset_train,
                                        constants.TEST: multiclass_pretrained_hidden_state_dataset_test}
    return result


def probe_with_logistic_regression(hidden_state_datasets, base_model_type):
    """

    :param hidden_state_datasets: A dictionary mapping the probing model type to the train and test datasets it
        produces. Concretely, each such datasets maps a hidden representation of textual inputs to the appropriate
        label in the probing task (binary prediction or multi-class prediction of premise mode).

        The keys of the resulting dictionary are a subset of {'pretrained', 'finetuned', 'multiclass'}. Values for each
        key consist of additional dictionaries. These inner dictionaries consist of the keys 'train' and 'test'. The
        values of this inner dictionary are the training set, test set respectively. Each of these  datasets is an
        instance of 'preprocessing.CMVPremiseModes'
    :param base_model_type: One of 'pretrained', 'finetuned' or 'multiclass'.
    :return: A two-tuple consisting of:
        (1) A trained probing model. This is a 'sklearn.linear_model.LogisticRegression' instance.
        (2) Model evaluation metrics on the test set. This is a A dictionary whose keys are base model type strings
         as defined above, and whose values are inner dictionaries. The inner dictionaries map metric names to their
         corresponding values.
    """
    assert base_model_type in hidden_state_datasets, \
        f"{base_model_type} is not included within 'hidden_state_datasets' keys: {hidden_state_datasets.keys()}"

    dataset_train = hidden_state_datasets[base_model_type][constants.TRAIN]
    dataset_test = hidden_state_datasets[base_model_type][constants.TEST]
    hidden_states_train = dataset_train.cmv_premise_mode_dataset[constants.HIDDEN_STATE]
    targets_train = dataset_train.cmv_premise_mode_dataset[constants.LABEL]
    probing_model = (
        sklearn.linear_model.LogisticRegression(
            random_state=0,
            multi_class='auto',
            # Logistic regression performs best with default values for 'weight_class', 'C' and 'max_iter').
            class_weight='balanced',
            C=10,
            max_iter=1000).fit(hidden_states_train, targets_train))
    hidden_states_eval = (
        dataset_test.cmv_premise_mode_dataset[constants.HIDDEN_STATE])
    targets_eval = (
        dataset_test.cmv_premise_mode_dataset[constants.LABEL])
    preds_eval = probing_model.predict(hidden_states_eval)
    confusion_matrix = sklearn.metrics.confusion_matrix(targets_eval, preds_eval)
    classification_report = sklearn.metrics.classification_report(targets_eval, preds_eval)
    print(f'{base_model_type} confusion matrix: {confusion_matrix}')
    print(f'{base_model_type} classification report: {classification_report}')

    eval_metrics = {
        constants.CONFUSION_MATRIX: confusion_matrix,
        constants.CLASSIFICATION_REPORT: classification_report}

    return probing_model, None, eval_metrics


def probe_model_with_mlp(hidden_state_datasets,
                         num_labels,
                         learning_rate,
                         training_batch_size,
                         eval_batch_size,
                         num_epochs,
                         base_model_type,
                         scheduler_gamma=0.9):
    """

    :param hidden_state_datasets:
    :param num_labels:
    :param learning_rate:
    :param training_batch_size:
    :param eval_batch_size:
    :param num_epochs:
    :param base_model_type:
    :param scheduler_gamma:
    :return:
    """
    probing_model = probing_models.MLP(num_labels=num_labels)
    loss_function = torch.nn.BCELoss() if num_labels == constants.NUM_LABELS else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(probing_model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    train_loader = torch.utils.data.DataLoader(
        hidden_state_datasets[base_model_type][constants.TRAIN],
        batch_size=training_batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        hidden_state_datasets[base_model_type][constants.TEST],
        batch_size=eval_batch_size,
        shuffle=True)
    probing_model.train_probe(train_loader,
                              optimizer,
                              num_labels=num_labels,
                              loss_function=loss_function,
                              num_epochs=num_epochs,
                              scheduler=scheduler)
    confusion_matrix, classification_report = (
        probing_model.eval_probe(test_loader=test_loader))
    print(f'{base_model_type} confusion matrix: {confusion_matrix}')
    print(f'{base_model_type} classification report: {classification_report}')
    eval_metrics = {
        constants.CONFUSION_MATRIX: confusion_matrix,
        constants.CLASSIFICATION_REPORT: classification_report
    }
    return probing_model, None, eval_metrics


def probe_model_on_multiclass_premise_modes(dataset,
                                            current_path,
                                            fine_tuned_model_path,
                                            pretrained_checkpoint_name,
                                            generate_new_probing_dataset=False,
                                            probing_model=constants.LOGISTIC_REGRESSION,
                                            learning_rate=1e-4,
                                            training_batch_size=16,
                                            eval_batch_size=64,
                                            num_epochs=20,
                                            scheduler_gamma=0.9):
    """

    :param dataset:
    :param current_path:
    :param fine_tuned_model_path:
    :param pretrained_checkpoint_name:
    :param generate_new_probing_dataset:
    :param probing_model:
    :param learning_rate:
    :param training_batch_size:
    :param eval_batch_size:
    :param num_epochs:
    :return:
    """
    probing_dir_path = os.path.join(current_path, constants.PROBING)
    if not os.path.exists(probing_dir_path):
        print(f'Creating directory: {probing_dir_path}')
        os.mkdir(probing_dir_path)
    premise_mode_probing_dir_path = os.path.join(probing_dir_path, constants.MULTICLASS)
    if not os.path.exists(premise_mode_probing_dir_path):
        print(f'Crating directory: {premise_mode_probing_dir_path}')
        os.mkdir(premise_mode_probing_dir_path)
    if generate_new_probing_dataset:
        save_hidden_layer_outputs(
            fine_tuned_model_path=fine_tuned_model_path,
            pretrained_checkpoint_name=pretrained_checkpoint_name,
            probing_dataset=dataset,
            probing_dir_path=premise_mode_probing_dir_path,
            num_labels=len(constants.PREMISE_MODE_TO_INT))
    hidden_state_datasets = (
        load_hidden_layer_outputs(
            probing_dir_path=os.path.join(probing_dir_path, constants.MULTICLASS),
            pretrained=False,
            fine_tuned=False,
            multiclass=True))
    if probing_model == constants.LOGISTIC_REGRESSION:
        return probe_with_logistic_regression(hidden_state_datasets, base_model_type=constants.MULTICLASS)
    else:
        return probe_model_with_mlp(hidden_state_datasets=hidden_state_datasets,
                                    num_labels=len(constants.PREMISE_MODE_TO_INT),
                                    learning_rate=learning_rate,
                                    training_batch_size=training_batch_size,
                                    eval_batch_size=eval_batch_size,
                                    num_epochs=num_epochs,
                                    base_model_type=constants.MULTICLASS,
                                    scheduler_gamma=scheduler_gamma)


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
                                  num_epochs=20):
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
    hidden_layer_datasets = (
        load_hidden_layer_outputs(premise_mode_probing_dir_path, pretrained=True, fine_tuned=True, multiclass=False))
    if probing_model == constants.MLP:
        pretrained_probing_model, _, pretraining_eval_metrics = probe_model_with_mlp(
            hidden_layer_datasets,
            num_labels=constants.NUM_LABELS,
            learning_rate=learning_rate,
            training_batch_size=training_batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs=num_epochs,
            base_model_type=constants.PRETRAINED,
            scheduler_gamma=0.9)
        fine_tuned_probing_model, _, fine_tuned_eval_metrics = probe_model_with_mlp(
            hidden_layer_datasets,
            num_labels=constants.NUM_LABELS,
            learning_rate=learning_rate,
            training_batch_size=training_batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs=num_epochs,
            base_model_type=constants.FINE_TUNED,
            scheduler_gamma=0.9)
        eval_metrics = {constants.PRETRAINED: pretraining_eval_metrics,
                        constants.FINE_TUNED: fine_tuned_eval_metrics}
        return pretrained_probing_model, fine_tuned_probing_model, eval_metrics
    elif probing_model == constants.LOGISTIC_REGRESSION:
        pretrained_probing_model, _, pretraining_eval_metrics = probe_with_logistic_regression(
            hidden_state_datasets=hidden_layer_datasets,
            base_model_type=constants.PRETRAINED)
        fine_tuned_probing_model, _, fine_tuned_eval_metrics = probe_with_logistic_regression(
            hidden_state_datasets=hidden_layer_datasets,
            base_model_type=constants.FINE_TUNED)
        eval_metrics = {constants.PRETRAINED: pretraining_eval_metrics,
                        constants.FINE_TUNED: fine_tuned_eval_metrics}
        return pretrained_probing_model, fine_tuned_probing_model, eval_metrics
    else:
        raise RuntimeError(f"Probing model must be one of {constants.LOGISTIC_REGRESSION} or {constants.MLP}.")
