from __future__ import annotations

import math
import os
import typing

import datasets
import torch
import torch.nn.functional as F
import torch_geometric.data
import transformers
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import numpy as np

import utils
import wandb

import constants
import metrics
from cmv_modes import preprocessing_knowledge_graph
from metrics import compute_metrics


class LogisticRegressionProbe(torch.nn.Module):
    def __init__(self, num_labels):
        super(LogisticRegressionProbe, self).__init__()
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class MLPProbe(torch.nn.Module):
    def __init__(self, num_labels):
        """
        Initialize an MLP consisting of one linear layer that maintain the hidden dimensionality, and then a projection
        into num_labels dimensions. The first linear layer has a ReLU non-linearity, while the final (logits)
        layer has a softmax activation.

        :param num_labels: The output dimensionality of the MLP that corresponds to the number of possible labels for
            the classification task.
        """
        super(MLPProbe, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, num_labels),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        """

        :param x: A batch of input tensors with dimensionality [batch_size, hidden_dimension]
        :return: The output tensor for the batch of shape [batch_size, num_labels].
        """
        return self.layers(x)


def train_probe(probing_model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                validation_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.optimizer.Optimizer,
                num_labels: int,
                loss_function: torch.nn.BCELoss | torch.nn.CrossEntropyLoss,
                num_epochs: int,
                max_num_rounds_no_improvement: int,
                metric_for_early_stopping: str,
                scheduler=None):
    """Train the probing model on a classification task given the provided training set and parameters.

    :param probing_model: A torch.nn.Module on which we are performing training.
    :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance. This loader
        contains the training set.
    :param validation_loader: A 'torch.utils.data.DataLoader` wrapping a `preprocessing.CMVDataset` instance. This
        loader contains the validation set.
    :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
    :param num_labels: An integer representing the output space (number of labels) for the probing classification
        problem.
    :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
    :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
    :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
        does not increase. If validation accuracy does not increase within this number of loops, we stop training
        early.
    :param metric_for_early_stopping: The metric used to determine whether or not to stop early. If the metric of
        interest does not improve within `max_num_rounds_no_improvement`, then we stop early.
    :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
    """

    # Initialize variables for early stopping.
    best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
    utils.ensure_dir_exists(best_model_dir_path)
    best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_probe.pt')
    lowest_loss = math.inf
    highest_accuracy = 0
    num_rounds_no_improvement = 0
    epoch_with_optimal_performance = 0

    probing_model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_num_batches = 0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            targets = data[constants.LABEL]
            outputs = probing_model(data[constants.HIDDEN_STATE])
            preds = torch.argmax(outputs, dim=1)
            loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_correct_preds = (preds == targets).sum().float()
            accuracy = num_correct_preds / targets.shape[0] * 100
            train_acc += accuracy
            train_num_batches += 1
        scheduler.step()
        wandb.log({f"{constants.TRAIN} {constants.ACCURACY}": train_acc / train_num_batches,
                   f"{constants.TRAIN} {constants.EPOCH}": epoch,
                   f"{constants.TRAIN} {constants.LOSS}": train_loss / train_num_batches})

        probing_model.eval()
        validation_loss = 0.0
        validation_acc = 0.0
        validation_num_batches = 0
        for i, data in enumerate(validation_loader, 0):
            targets = data[constants.LABEL]
            outputs = probing_model(data[constants.HIDDEN_STATE])
            preds = torch.argmax(outputs, dim=1)
            loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
            validation_loss += loss.item()
            num_correct_preds = (preds == targets).sum().float()
            accuracy = num_correct_preds / targets.shape[0] * 100
            validation_acc += accuracy
            validation_num_batches += 1

        validation_loss = validation_loss / validation_num_batches
        validation_acc = validation_acc / validation_num_batches
        wandb.log({f"{constants.VALIDATION} {constants.ACCURACY}": validation_acc,
                   f"{constants.VALIDATION} {constants.EPOCH}": epoch,
                   f"{constants.VALIDATION} {constants.LOSS}": validation_loss})

        if metric_for_early_stopping == constants.LOSS and validation_loss < lowest_loss:
            lowest_loss = validation_loss
            num_rounds_no_improvement = 0
            epoch_with_optimal_performance = epoch
            torch.save(probing_model.state_dict(), best_model_path)
        elif metric_for_early_stopping == constants.ACCURACY and validation_acc > highest_accuracy:
            highest_accuracy = validation_acc
            num_rounds_no_improvement = 0
            epoch_with_optimal_performance = epoch
            torch.save(probing_model.state_dict(), best_model_path)
        else:
            num_rounds_no_improvement += 1

        if num_rounds_no_improvement == max_num_rounds_no_improvement:
            print(f'Performing early stopping after {epoch} epochs.\n'
                  f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
            probing_model.load_state_dict(torch.load(best_model_path))
            return probing_model

        probing_model.train()

    return probing_model


def eval_probe(probing_model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               num_labels: int,
               split_name: str = constants.TEST,
               log_results: bool = True) -> typing.Mapping[str, float]:
    """Evaluate the trained classification probe on a held out test set.

    :param probing_model: A torch.nn.Module on which we are performing evaluation.
    :param test_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance for some
        premise mode.
    :param num_labels: The number of labels for the probing classification problem.
    :param split_name: The string name of the dataset split. Typically one of {train, validation, test}.
    :param log_results: True if we wish to log evaluation results on the provided dataset to wandb. False otherwise.
    :return: A mapping from metric names (keys) to metric values (floats) obtained while evaluating the probing model.
    """
    preds_list = []
    targets_list = []
    probing_model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            outputs = probing_model(data[constants.HIDDEN_STATE])
            preds = torch.argmax(outputs, dim=1)
            preds_list.append(preds)
            targets = data[constants.LABEL]
            targets_list.append(targets)
    preds_list = np.concatenate(preds_list)
    targets_list = np.concatenate(targets_list)
    eval_metrics = metrics.compute_metrics(num_labels=num_labels,
                                           preds=preds_list,
                                           targets=targets_list,
                                           split_name=split_name)
    if log_results:
        for metric_name, metric_value in eval_metrics.items():
            wandb.summary[f"eval_{metric_name}"] = metric_value
    return eval_metrics


class BaselineLogisticRegression(torch.nn.Module):
    """

    """

    def __init__(self, num_features, num_labels):
        """

        :param num_features: An integer representing the number of dimensions that exist in the input tensor.
        :param num_labels: The number of labels for the probing classification problem.
        """
        super(BaselineLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_labels)

    def forward(self, x):
        """Pass input x through the model as part of the forward pass.

        :param x: A tensor with shape [batch_size, num_labels].
        :return: A tensor with shape [batch_size, num_labels].
        """
        outputs = torch.sigmoid(self.linear(x.float()))
        return outputs

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            num_labels: int,
            loss_function: typing.Union[torch.nn.BCELoss, torch.nn.CrossEntropyLoss],
            num_epochs: int,
            optimizer: torch.optim.Optimizer,
            max_num_rounds_no_improvement: int,
            metric_for_early_stopping: str,
            scheduler: typing.Union[torch.optim.lr_scheduler.ExponentialLR,
                                    torch.optim.lr_scheduler.ConstantLR] = None,
            ):
        """

        :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance. This loader
            contains the training set.
        :param validation_loader: A 'torch.utils.data.DataLoader` wrapping a `preprocessing.CMVDataset` instance. This
            loader contains the validation set.
        :param num_labels: The number of labels for the probing classification problem.
        :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
        :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
        :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
        :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
        :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
            does not increase. If validation accuracy does not increase within this number of loops, we stop training
            early.
        :param metric_for_early_stopping: The metric used to determine whether or not to stop early. If the metric of
            interest does not improve within `max_num_rounds_no_improvement`, then we stop early.
        """
        self.train()
        highest_accuracy = 0
        lowest_loss = math.inf
        num_rounds_no_improvement = 0
        epoch_with_optimal_performance = 0
        best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
        utils.ensure_dir_exists(best_model_dir_path)
        best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_probe.pt')
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_acc = 0.0
            train_num_batches = 0
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                targets = data[constants.LABEL]
                outputs = self(data['features'])
                preds = torch.argmax(outputs, dim=1)
                loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
                loss.backward()
                optimizer.step()
                num_correct_preds = (preds == targets).sum().float()
                accuracy = num_correct_preds / targets.shape[0] * 100
                train_num_batches += 1
                train_loss += loss.item()
                train_acc += accuracy
            wandb.log({f"{constants.TRAIN} {constants.ACCURACY}": train_acc / train_num_batches,
                       f"{constants.TRAIN} {constants.EPOCH}": epoch,
                       f"{constants.TRAIN} {constants.LOSS}": train_loss / train_num_batches})
            scheduler.step()

            self.eval()
            validation_loss = 0.0
            validation_acc = 0.0
            validation_num_batches = 0
            for i, data in enumerate(validation_loader, 0):
                targets = data[constants.LABEL]
                outputs = self(data["features"])
                preds = torch.argmax(outputs, dim=1)
                loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
                validation_loss += loss.item()
                num_correct_preds = (preds == targets).sum().float()
                accuracy = num_correct_preds / targets.shape[0] * 100
                validation_acc += accuracy
                validation_num_batches += 1

            validation_loss = validation_loss / validation_num_batches
            validation_acc = validation_acc / validation_num_batches
            wandb.log({f"{constants.VALIDATION} {constants.ACCURACY}": validation_acc,
                       f"{constants.VALIDATION} {constants.EPOCH}": epoch,
                       f"{constants.VALIDATION} {constants.LOSS}": validation_loss})

            if metric_for_early_stopping == constants.LOSS and validation_loss < lowest_loss:
                lowest_loss = validation_loss
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                torch.save(self.state_dict(), best_model_path)
            elif metric_for_early_stopping == constants.ACCURACY and validation_acc > highest_accuracy:
                highest_accuracy = validation_acc
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                torch.save(self.state_dict(), best_model_path)
            else:
                num_rounds_no_improvement += 1

            if num_rounds_no_improvement == max_num_rounds_no_improvement:
                print(f'Performing early stopping after {epoch} epochs.\n'
                      f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
                self.load_state_dict(torch.load(best_model_path))
                break

    def evaluate(self, 
                 test_loader: torch.utils.data.DataLoader,
                 num_labels: int,
                 split_name: str = constants.TEST,
                 log_results: bool = True) -> typing.Mapping[str, float]:
        """

        :param test_loader: A 'torch.utils.data.DataLoader` wrapping a `preprocessing.CMVDataset` instance. This
            loader contains the test set.
        :param num_labels: An integer representing the output space (number of labels) for the probing classification
            problem.
        :param split_name: The string name of the dataset split. Typically one of {train, validation, test}.
        :param log_results: True if we wish to log evaluation results on the provided dataset to wandb. False otherwise.
        :return: A mapping from metric names (keys) to metric values (floats) obtained while evaluating the probing
            model."""
        preds_list = []
        targets_list = []
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                outputs = self(data['features'])
                preds = torch.argmax(outputs, dim=1)
                preds_list.append(preds)
                targets = data[constants.LABEL]
                targets_list.append(targets)
        preds_list = np.concatenate(preds_list)
        targets_list = np.concatenate(targets_list)
        eval_metrics = compute_metrics(num_labels=num_labels,
                                       preds=preds_list,
                                       targets=targets_list,
                                       split_name=split_name)
        if log_results:
            for metric_name, metric_value in eval_metrics.items():
                wandb.summary[f"eval_{metric_name}"] = metric_value
        return eval_metrics


# TODO: Expand this model's GCN architecture.
class GCNWithBertEmbeddings(torch.nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_classes: int,
                 hidden_layer_dim: int,
                 use_frozen_bert: bool = True):
        """

        :param num_node_features: The dimensionality of each node within the batch of graph inputs.
        :param num_classes: The number of possible output classes produced by the final layer of the model.
        :param hidden_layer_dim: The dimensionality of the GNN's intermediate layers.
        :param use_frozen_bert: A boolean denoting whether or not the BERT model which produces node embeddings should
            be updated during training.
        """
        super().__init__()

        self.bert_model = transformers.BertModel.from_pretrained(constants.BERT_BASE_CASED)
        if use_frozen_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_layer_dim = hidden_layer_dim
        self.conv1 = GCNConv(self.num_node_features, self.hidden_layer_dim)
        self.conv2 = GCNConv(self.hidden_layer_dim, self.num_classes)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self,
                data: torch_geometric.data.Data):
        """

        :param data: A collection of node and edge data corresponding to a batch of graphs inputted to the model.
        :return: A probability distribution over the output labels. A torch tensor of shape
            [batch_size, num_output_labels].
        """
        x, edge_index = data.x, data.edge_index
        input_ids, token_type_ids, attention_mask = torch.hsplit(x, sections=3)
        bert_outputs = self.bert_model(
            input_ids=torch.squeeze(input_ids, dim=1).long(),
            token_type_ids=torch.squeeze(token_type_ids, dim=1).long(),
            attention_mask=torch.squeeze(attention_mask, dim=1).long(),
        )
        node_embeddings = bert_outputs['last_hidden_state'][:, 0, :]
        x = self.conv1(node_embeddings, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1)


class BertParagraphWithMLP(torch.nn.Module):
    def __init__(self, bert_model, layers, pooling=None):
        super(BertParagraphWithMLP, self).__init__()
        self.bert_model = bert_model
        self.layers = layers
        self.pooling = pooling

    def forward(self, X):
        bert_outputs = self.bert_model(X)
        mlp_outputs = self.layers(bert_outputs)
        pooling_outputs = self.pooling(mlp_outputs)
        return F.log_softmax(pooling_outputs, dim=1)


class BertBaseline(torch.nn.Module):
    def __init__(self, use_frozen_bert, mlp_layers, device):
        super(BertBaseline, self).__init__()

        self._bert_model = transformers.BertModel.from_pretrained(constants.BERT_BASE_CASED).to(device)
        if use_frozen_bert:
            for param in self._bert_model.parameters():
                param.requires_grad = False
        self._mlp = mlp_layers.to(device)
        self.device = device

    def forward(self, bert_inputs):
        """Pass input x through the model as part of the forward pass.

        :param bert_inputs: A dictionary mapping string keys to tensor values corresponding to BERT inputs. Each
            input tensor has shape [batch_size, sequence_length]
        :return: A tensor with shape [batch_size, num_labels].
        """
        bert_outputs = self._bert_model(**bert_inputs)
        sequence_embeddings = bert_outputs['pooler_output'].to(self.device)
        # sequence_embeddings = bert_outputs['last_hidden_state'].to(self.device)[:, 0, :]
        logits = self._mlp(sequence_embeddings)
        label_probabilities = F.softmax(logits, dim=1)
        return label_probabilities

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            num_labels: int,
            loss_function: typing.Union[torch.nn.BCELoss, torch.nn.CrossEntropyLoss],
            num_epochs: int,
            optimizer: torch.optim.Optimizer,
            max_num_rounds_no_improvement: int,
            metric_for_early_stopping: str,
            scheduler: typing.Union[torch.optim.lr_scheduler.ExponentialLR,
                                    torch.optim.lr_scheduler.ConstantLR] = None,
            grad_accumulation_steps: int = 1):
        """

        :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance. This loader
            contains the training set.
        :param validation_loader: A 'torch.utils.data.DataLoader` wrapping a `preprocessing.CMVDataset` instance. This
            loader contains the validation set.
        :param num_labels: The number of labels for the probing classification problem.
        :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
        :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
        :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
        :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
        :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
            does not increase. If validation accuracy does not increase within this number of loops, we stop training
            early.
        :param metric_for_early_stopping: The metric used to determine whether or not to stop early. If the metric of
            interest does not improve within `max_num_rounds_no_improvement`, then we stop early.
        """
        self.train()
        max_accuracy = 0
        min_loss = math.inf
        num_rounds_no_improvement = 0
        epoch_with_optimal_performance = 0
        best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
        utils.ensure_dir_exists(best_model_dir_path)
        best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_bert_baseline.pt')
        for epoch in range(num_epochs):
            epoch_training_metrics = {}
            loss = 0
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                targets = data[constants.LABEL].to(device)
                # print(f'\ttrain targets: {targets}')
                outputs = self({
                    constants.INPUT_IDS: data[constants.INPUT_IDS].to(self.device),
                    constants.TOKEN_TYPE_IDS: data[constants.TOKEN_TYPE_IDS].to(self.device),
                    constants.ATTENTION_MASK: data[constants.ATTENTION_MASK].to(self.device)
                })
                preds = torch.argmax(outputs, dim=1).to(device)
                # print(f'\ttrain preds: {preds}')
                loss = loss + loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
                training_metrics = metrics.compute_metrics(
                    num_labels=num_labels,
                    preds=preds.cpu().numpy(),
                    targets=targets.cpu().numpy(),
                    split_name=constants.TRAIN)

                # include training loss in batch metrics.
                training_metrics[f'{constants.TRAIN}_{constants.LOSS}'] = loss.detach().cpu().numpy()

                if (i + 1) % grad_accumulation_steps == 0:
                    # every 10 iterations of batches of size 10
                    optimizer.zero_grad()
                    loss.backward()
                    # huge graph is cleared here
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    loss = 0

                for metric_name, metric_value in training_metrics.items():
                    if metric_name not in epoch_training_metrics:
                        epoch_training_metrics[metric_name] = []
                    epoch_training_metrics[metric_name].append(metric_value)

            aggregated_metrics = {}
            for metric_name, metric_values in epoch_training_metrics.items():
                aggregated_metrics[metric_name] = np.mean(metric_values)
            aggregated_metrics[f'{constants.TRAIN}_{constants.EPOCH}'] = epoch
            wandb.log(aggregated_metrics)
            # print(f'Train metrics:\n{aggregated_metrics}')

            # Perform evaluation.
            if epoch % 5 == 0:
                epoch_validation_metrics = {}
                self.eval()
                for i, data in enumerate(validation_loader):
                    targets = data[constants.LABEL].to(device)
                    # print(f'\tvalidation targets: {targets}')
                    outputs = self({
                        constants.INPUT_IDS: data[constants.INPUT_IDS].to(device),
                        constants.TOKEN_TYPE_IDS: data[constants.TOKEN_TYPE_IDS].to(device),
                        constants.ATTENTION_MASK: data[constants.ATTENTION_MASK].to(device)
                    }).to(device)
                    output_distribution = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(output_distribution, dim=1).to(device)
                    # print(f'\tvalidation preds: {preds}')
                    validation_metrics = metrics.compute_metrics(
                        num_labels=num_labels,
                        preds=preds.cpu().numpy(),
                        targets=targets.cpu().numpy(),
                        split_name=constants.VALIDATION)

                    # Include validation loss in batch metrics.
                    validation_metrics[f'{constants.VALIDATION}_{constants.LOSS}'] = (
                        loss_function(outputs,
                                      torch.nn.functional.one_hot(targets, num_labels).float()).detach().cpu().numpy()
                    )

                    for metric_name, metric_value in validation_metrics.items():
                        if metric_name not in epoch_validation_metrics:
                            epoch_validation_metrics[metric_name] = []
                        epoch_validation_metrics[metric_name].append(metric_value)

                # Perform metric aggregation over batches
                aggregated_metrics = {}
                for metric_name, metric_values in epoch_validation_metrics.items():
                    aggregated_metrics[metric_name] = np.mean(metric_values)
                aggregated_metrics[f'{constants.VALIDATION}_{constants.EPOCH}'] = epoch

                # Stop early if accuracy does not increase within `max_num_rounds_no_improvement`evaluation runs.
                epoch_accuracy = aggregated_metrics[f'{constants.VALIDATION}_{constants.ACCURACY}']
                epoch_loss = aggregated_metrics[f'{constants.VALIDATION}_{constants.LOSS}']
                if metric_for_early_stopping == constants.ACCURACY and epoch_accuracy > max_accuracy:
                    max_accuracy = epoch_accuracy
                    num_rounds_no_improvement = 0
                    epoch_with_optimal_performance = epoch
                    torch.save(self.state_dict(), best_model_path)
                elif metric_for_early_stopping == constants.LOSS and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    num_rounds_no_improvement = 0
                    epoch_with_optimal_performance = epoch
                    torch.save(self.state_dict(), best_model_path)
                else:
                    num_rounds_no_improvement += 1

                wandb.log(aggregated_metrics)
                # print(f'Validation metrics:\n{aggregated_metrics}')

                if num_rounds_no_improvement == max_num_rounds_no_improvement:
                    print(f'Performing early stopping after {epoch} epochs.\n'
                          f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
                    self.load_state_dict(torch.load(best_model_path))
                    break


if __name__ == '__main__':
    #model = GCNWithBertEmbeddings(256, 2, 16)
    #print(model)
    examples = preprocessing_knowledge_graph.create_simple_bert_inputs(
        directory_path=os.path.join(os.getcwd(), 'cmv_modes', 'change-my-view-modes-master'),
        version=constants.v2_path)
    features = examples[0]
    labels = examples[1]
    labels = np.array(labels)
    print(f'positive_labels: {sum(labels == 1)}\n'
          f'negative_labels: {sum(labels == 0)}')
    op_text = [pair[0] for pair in features]
    reply_text = [pair[1] for pair in features]
    tokenizer = transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    tokenized_inputs = tokenizer(op_text, reply_text, padding=True, truncation=True)
    transformers.logging.set_verbosity(verbosity)
    dataset_dict = {input_name: input_value for input_name, input_value in tokenized_inputs.items()}
    dataset_dict[constants.LABEL] = labels
    dataset = datasets.Dataset.from_dict(dataset_dict)
    dataset.set_format(type='torch',
                       columns=[
                           constants.INPUT_IDS,
                           constants.TOKEN_TYPE_IDS,
                           constants.ATTENTION_MASK,
                           constants.LABEL])
    dataset = dataset.shuffle()
    print(f'Entire dataset positive_labels: {sum(dataset[constants.LABEL].numpy() == 1)}\n'
          f'Entire dataset negative_labels: {sum(dataset[constants.LABEL].numpy() == 0)}')
    # mlp = torch.nn.Sequential(
    #     torch.nn.Linear(constants.BERT_HIDDEN_DIM, constants.BERT_HIDDEN_DIM),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(constants.BERT_HIDDEN_DIM, 512),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(512, constants.NUM_LABELS),
    # )
    mlp = torch.nn.Sequential(torch.nn.Linear(constants.BERT_HIDDEN_DIM, constants.BERT_HIDDEN_DIM),
                              torch.nn.ReLU(),
                              torch.nn.Linear(constants.BERT_HIDDEN_DIM, constants.NUM_LABELS))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bert_baseline = BertBaseline(use_frozen_bert=True, mlp_layers=mlp, device=device)

    num_cross_validation_splits = 5
    shard_train_metrics = []
    shard_eval_metrics = []
    shards = [dataset.shard(num_cross_validation_splits, i, contiguous=True)
              for i in range(num_cross_validation_splits)]
    for validation_set_index in range(num_cross_validation_splits):
        split_model = copy.deepcopy(bert_baseline).to(device)
        validation_and_test_sets = shards[validation_set_index].train_test_split(test_size=0.5)
        validation_set = validation_and_test_sets[constants.TRAIN].shuffle()
        print(f'Validation set ({validation_set_index}) positive labels: '
              f'{sum(validation_set[constants.LABEL].numpy() == 1)}')
        print(f'Validation set ({validation_set_index}) negative labels: '
              f'{sum(validation_set[constants.LABEL].numpy() == 0)}')
        test_set = validation_and_test_sets[constants.TEST].shuffle()
        training_set = datasets.concatenate_datasets(
            shards[0:validation_set_index] + shards[validation_set_index + 1:]).shuffle()
        run_name = f'Fine-tune BERT+MLP Baseline on {constants.BINARY_CMV_DELTA_PREDICTION}, ' \
                   f'Split #{validation_set_index + 1}'
        run = wandb.init(
            project="persuasive_arguments",
            entity="zbamberger",
            reinit=True,
            name=run_name)
        optimizer = torch.optim.Adam(split_model.parameters(), lr=1e-5)
        bert_baseline.fit(train_loader=torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True),
                          validation_loader=torch.utils.data.DataLoader(validation_set, batch_size=1, shuffle=True),
                          num_labels=2,
                          loss_function=torch.nn.BCELoss(),
                          num_epochs=100,
                          optimizer=optimizer,
                          max_num_rounds_no_improvement=10,
                          metric_for_early_stopping=constants.ACCURACY)
        run.finish()

