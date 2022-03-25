from __future__ import annotations

import math
import os
import typing

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
                max_num_rounds_no_improvement: int = None,
                metric_for_early_stopping: str = None,
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
    probing_model.train()
    max_accuracy = 0
    min_loss = math.inf
    num_rounds_no_improvement = 0
    epoch_with_optimal_performance = 0
    best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
    utils.ensure_dir_exists(best_model_dir_path)
    best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_probe.pt')

    for epoch in range(num_epochs):
        epoch_training_metrics = {}
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            targets = data[constants.LABEL]
            outputs = probing_model(data[constants.HIDDEN_STATE])
            preds = torch.argmax(outputs, dim=1)
            loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            training_metrics = metrics.compute_metrics(
                num_labels=num_labels,
                preds=preds,
                targets=targets,
                split_name=constants.TRAIN)
            # include training loss in batch metrics.
            training_metrics[f'{constants.TRAIN}_{constants.LOSS}'] = loss.detach().numpy()
            for metric_name, metric_value in training_metrics.items():
                if metric_name not in epoch_training_metrics:
                    epoch_training_metrics[metric_name] = []
                epoch_training_metrics[metric_name].append(metric_value)

        aggregated_metrics = {}
        for metric_name, metric_values in epoch_training_metrics.items():
            aggregated_metrics[metric_name] = np.mean(metric_values)
        aggregated_metrics[f'{constants.TRAIN}_{constants.EPOCH}'] = epoch
        wandb.log(aggregated_metrics)
        
        # Perform evaluation.
        if epoch % 5 == 0:
            epoch_validation_metrics = {}
            probing_model.eval()
            for i, data in enumerate(validation_loader):
                targets = data[constants.LABEL]
                outputs = probing_model(data[constants.HIDDEN_STATE])
                preds = torch.argmax(outputs, dim=1)
                validation_metrics = metrics.compute_metrics(
                    num_labels=num_labels,
                    preds=preds,
                    targets=targets,
                    split_name=constants.VALIDATION)

                # include validation loss in batch metrics.
                validation_metrics[f'{constants.VALIDATION}_{constants.LOSS}'] = (
                    loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float()).detach().numpy()
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
                torch.save(probing_model.state_dict(), best_model_path)
            elif metric_for_early_stopping == constants.LOSS and epoch_loss < min_loss:
                min_loss = epoch_loss
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                torch.save(probing_model.state_dict(), best_model_path)
            else:
                num_rounds_no_improvement += 1

            wandb.log(aggregated_metrics)

            if num_rounds_no_improvement == max_num_rounds_no_improvement:
                print(f'Performing early stopping after {epoch} epochs.\n'
                      f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
                probing_model.load_state_dict(torch.load(best_model_path))
                return probing_model

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
        wandb.log(eval_metrics)
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
        max_accuracy = 0
        min_loss = math.inf
        num_rounds_no_improvement = 0
        epoch_with_optimal_performance = 0
        best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
        utils.ensure_dir_exists(best_model_dir_path)
        best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_probe.pt')
        for epoch in range(num_epochs):
            epoch_training_metrics = {}
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                targets = data[constants.LABEL]
                outputs = self(data['features'])
                preds = torch.argmax(outputs, dim=1)
                loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
                loss.backward()
                optimizer.step()
                scheduler.step()
                training_metrics = metrics.compute_metrics(
                    num_labels=num_labels,
                    preds=preds,
                    targets=targets,
                    split_name=constants.TRAIN)

                # include training loss in batch metrics.
                training_metrics[f'{constants.TRAIN}_{constants.LOSS}'] = loss.detach().numpy()

                for metric_name, metric_value in training_metrics.items():
                    if metric_name not in epoch_training_metrics:
                        epoch_training_metrics[metric_name] = []
                    epoch_training_metrics[metric_name].append(metric_value)

            aggregated_metrics = {}
            for metric_name, metric_values in epoch_training_metrics.items():
                aggregated_metrics[metric_name] = np.mean(metric_values)
            aggregated_metrics[f'{constants.TRAIN}_{constants.EPOCH}'] = epoch
            wandb.log(aggregated_metrics)

            # Perform evaluation.
            if epoch % 5 == 0:
                epoch_validation_metrics = {}
                self.eval()
                for i, data in enumerate(validation_loader):
                    targets = data[constants.LABEL]
                    outputs = self(data['features'])
                    preds = torch.argmax(outputs, dim=1)
                    validation_metrics = metrics.compute_metrics(
                        num_labels=num_labels,
                        preds=preds,
                        targets=targets,
                        split_name=constants.VALIDATION)

                    # Include validation loss in batch metrics.
                    validation_metrics[f'{constants.VALIDATION}_{constants.LOSS}'] = (
                        loss_function(outputs,
                                      torch.nn.functional.one_hot(targets, num_labels).float()).detach().numpy()
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
            wandb.log(eval_metrics)
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


class BertUtteranceWithPoolingAndMLP(torch.nn.Module):
    pass


if __name__ == '__main__':
    model = GCNWithBertEmbeddings(256, 2, 16)
    print(model)
    pass

