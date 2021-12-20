import os
import json
import numpy as np
import datasets
import sklearn
import transformers
import torch
import constants
import preprocessing

SUFFIX = "json"


class MLP(torch.nn.Module):
    def __init__(self):
        """

        """
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.layers(x)

    def train_probe(self, train_loader, optimizer, loss_function, num_epochs=5):
        """

        :param train_loader:
        :param optimizer:
        :param loss_function:
        :param num_epochs:
        :return:
        """
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                targets = data['label']
                outputs = self(data['hidden_state'])
                preds = torch.argmax(outputs, dim=1)
                loss = loss_function(outputs, torch.nn.functional.one_hot(targets, 2).float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_correct_preds = (preds == targets).sum().float()
                accuracy = num_correct_preds / targets.shape[0] * 100
                epoch_acc += accuracy
                num_batches += 1
            print(
                f'Epoch {epoch + 1:03}: | '
                f'Loss: {epoch_loss / num_batches:.5f} |'
                f'Acc: {epoch_acc / num_batches:.3f}'
            )

    def eval_probe(self, test_loader):
        """

        :param test_loader:
        :return:
        """
        preds_list = []
        targets_list = []
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                outputs = self(data['hidden_state'])
                preds = torch.argmax(outputs, dim=1)
                preds_list.append(preds)
                targets = data['label']
                targets_list.append(targets)
        preds_list = np.concatenate(preds_list)
        targets_list = np.concatenate(targets_list)
        confusion_matrix = sklearn.metrics.confusion_matrix(targets_list, preds_list)
        classification_report = sklearn.metrics.classification_report(targets_list, preds_list)
        return confusion_matrix, classification_report


def save_hidden_layer_outputs(fine_tuned_model_path,
                              probing_dataset,
                              probing_dir_path):
    """

    :param fine_tuned_model_path:
    :param probing_dataset:
    :param probing_dir_path:
    :return:
    """
    base_file_name = 'finetuned_bert_hidden_states'
    current_path = os.getcwd()
    model = transformers.BertForSequenceClassification.from_pretrained(
        os.path.join(current_path, fine_tuned_model_path),
        num_labels=constants.NUM_LABELS)
    dataloader = torch.utils.data.DataLoader(probing_dataset, batch_size=64)
    for batch_index, batch in enumerate(dataloader):
        file_name = f"{base_file_name}_batch_{batch_index + 1}.{SUFFIX}"
        file_path = os.path.join(probing_dir_path, file_name)
        model_outputs = model.forward(
            input_ids=batch[constants.INPUT_IDS],
            attention_mask=batch[constants.ATTENTION_MASK],
            token_type_ids=batch[constants.TOKEN_TYPE_IDS],
            output_hidden_states=True,
        )
        # Access the embedding within the [CLS] across all entries in the batch.
        embedding_hidden_layer = model_outputs.hidden_states[0][:, 0, :]
        with open(file_path, "w") as f:
            print(f'saving model embedding with shape {embedding_hidden_layer.shape} from batch #{batch_index + 1}'
                  f' to {file_path}')
            json.dump(
                {
                    'hidden_state': embedding_hidden_layer.tolist(),
                    'label': batch[constants.LABEL].tolist()
                }, f)


def load_hidden_layer_outputs(probing_dir_path):
    """

    :param probing_dir_path:
    :return:
    """
    hidden_layer_batch_file_names = list(
        filter(
            lambda file_name: file_name.endswith(SUFFIX),
            os.listdir(probing_dir_path)))
    hidden_layer_batch_file_paths = list(
        map(
            lambda file_name: os.path.join(probing_dir_path, file_name),
            hidden_layer_batch_file_names))
    hidden_state_batches = datasets.load_dataset(
        SUFFIX, data_files=hidden_layer_batch_file_paths)['train']
    hidden_state_dataset = datasets.Dataset.from_dict({
        'hidden_state': np.concatenate(hidden_state_batches['hidden_state']),
        'label': np.concatenate(hidden_state_batches[constants.LABEL])
    }).train_test_split()
    hidden_state_dataset_train = preprocessing.CMVPremiseModes(
        hidden_state_dataset[constants.TRAIN])
    hidden_state_dataset_test = preprocessing.CMVPremiseModes(
        hidden_state_dataset[constants.TEST])
    return hidden_state_dataset_train, hidden_state_dataset_test


def probe_model_with_premise_mode(premise_mode,
                                  premise_mode_dataset,
                                  current_path,
                                  fine_tuned_model_path,
                                  generate_new_probing_dataset=False,
                                  probing_model=constants.LOGISTIC_REGRESSION,
                                  learning_rate=1e-4,
                                  training_batch_size=16,
                                  eval_batch_size=64,
                                  num_epochs=20,):
    """

    :param premise_mode:
    :param premise_mode_dataset:
    :param current_path:
    :param fine_tuned_model_path:
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
        os.mkdir(probing_dir_path)
    premise_mode_probing_dir_path = os.path.join(probing_dir_path, constants.PREMISE_DIR_PATH_MAPPING[premise_mode])
    if not os.path.exists(premise_mode_probing_dir_path):
        os.mkdir(premise_mode_probing_dir_path)
    if generate_new_probing_dataset:
        save_hidden_layer_outputs(
            fine_tuned_model_path=fine_tuned_model_path,
            probing_dataset=premise_mode_dataset,
            probing_dir_path=premise_mode_probing_dir_path)
    premise_mode_hidden_state_dataset_train, premise_mode_hidden_state_dataset_test = (
        load_hidden_layer_outputs(probing_dir_path=premise_mode_probing_dir_path))

    if probing_model == constants.MLP:
        model = MLP()
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_loader = torch.utils.data.DataLoader(
            premise_mode_hidden_state_dataset_train, batch_size=training_batch_size, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(
            premise_mode_hidden_state_dataset_test, batch_size=eval_batch_size, shuffle=True, num_workers=1)
        probing_model.train_probe(train_loader, optimizer, loss_function, num_epochs=num_epochs)
        confusion_matrix, classification_report = (
            probing_model.eval_probe(test_loader=test_loader))
        eval_metrics = {
            constants.CONFUSION_MATRIX: confusion_matrix,
            constants.CLASSIFICATION_REPORT: classification_report
        }
    elif probing_model == constants.LOGISTIC_REGRESSION:
        hidden_states_train = premise_mode_hidden_state_dataset_train.cmv_premise_mode_dataset[constants.HIDDEN_STATE]
        targets_train = premise_mode_hidden_state_dataset_train.cmv_premise_mode_dataset[constants.LABEL]
        model = (
            sklearn.linear_model.LogisticRegression(random_state=0).fit(hidden_states_train, targets_train))
        hidden_states_eval = premise_mode_hidden_state_dataset_test.cmv_premise_mode_dataset[constants.HIDDEN_STATE]
        targets_eval = premise_mode_hidden_state_dataset_test.cmv_premise_mode_dataset[constants.LABEL]
        preds_eval = model.predict(hidden_states_eval)
        confusion_matrix = sklearn.metrics.confusion_matrix(targets_eval, preds_eval)
        classification_report = sklearn.metrics.classification_report(targets_eval, preds_eval)
        eval_metrics = {
            constants.CONFUSION_MATRIX: confusion_matrix,
            constants.CLASSIFICATION_REPORT: classification_report
        }
    return model, eval_metrics
