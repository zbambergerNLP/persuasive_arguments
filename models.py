from __future__ import annotations

import argparse
import copy
import math
import os
import random
import typing

import datasets
import torch
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
from torch_geometric.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import HeteroConv
import torch.nn as nn
import numpy as np
import tqdm

import utils
import wandb

import constants
import metrics
from cmv_modes import preprocessing_knowledge_graph
from data_loaders import create_dataloaders_for_k_fold_cross_validation
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
    lowest_loss = math.inf
    highest_accuracy = 0
    probing_model.train()
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
    A baseline linear regression model meant to be used on bi-gram features from persuasive arguments and their
    contexts.
    """

    def __init__(self, num_features, num_labels):
        """
        Initialize a logistic regression model used to predict argument persuasiveness.

        :param num_features: An integer representing the number of dimensions that exist in the input tensor.
        :param num_labels: The number of labels for the probing classification problem.
        """
        super(BaselineLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_labels)

    def forward(self, x):
        """
        Pass input x through the model as part of the forward pass.

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
        Train the linear regression baseline model to predict argument persuasiveness.

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

            self.train()
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
        Evaluate the ability of a linear regression baseline model to predict argument persuasiveness.

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


class HomophiliousGNN(torch.nn.Module):
    """
    An implementation for a homophilous graph neural network over the knowledge graphs of persuasive arguments.

    Nodes within the argument knowledge graphs correspond to argumentative prepositions. Relations within the knowledge
    graph correspond to either a supporting or an attacking relation from the source preposition towards the target
    preposition.

    The model ultamitely predicts the argument's persuasiveness given the argument's context (e.g., the title and OP
    within the CMV dataset).
    """

    def __init__(self,
                 out_channels: int,
                 hidden_channels: typing.List[int],
                 conv_type: str,
                 use_frozen_bert: bool = True,
                 use_max_pooling: bool = True,
                 encoder_type: str = "sbert",
                 dropout_prob: float = 0.0,
                 super_node: bool = False):
        """
        Initialize a homophilous graph neural network to predict argument persuasiveness given context.

        :param out_channels: The label-space dimensionality of the downstream task. The number of possible labels this
            model is tasked to predict among.
        :param hidden_channels: A list of the hidden dimensions used by this model. The first dimension is used to
            convert from BERT's hidden dimension to one more suitable for graph convolutions. The remaining dimensions
            reflect the outputs of the GNN's convolutional layers. There is also a final linear transformation
            from the final hidden dimension to the label-space dimensionality after pooling/super-node aggregation.
        :param conv_type: The string name of the graph convolutional layers used in this GNN. Must be one of {"GCN",
            "GAT", "SAGE"}.
        :param use_frozen_bert: True if we intend to use a frozen BERT model to produce node embeddings. False if BERT's
            weights should be updated while training the GNN (such that BERT's embeddings would evolve beyond their
            pre-trained values).
        :param use_max_pooling: True if we intend to use max pooling to aggregate node representations.
            Nodes are aggregated as part of graph-classification before projecting to the label-space dimensionality.
        :param encoder_type: A string representing the architecture used for tokenizing texts, and mapping these
            tokenized sequences to dense representations. Must be one of {"bert", "sbert"}
        """
        super().__init__()

        # Choose the encoder type among BERT and a distilled RoBERTa sentence transformer.
        self.encoder_type = encoder_type
        if self.encoder_type == 'sbert':
            self.bert_model = SentenceTransformer('all-distilroberta-v1')
        else:
            self.bert_model = transformers.BertModel.from_pretrained(constants.BERT_BASE_CASED)

        # Since the dataset is quite small, consider freezing the encoder's parameters to avoid over-fitting.
        if use_frozen_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        prev_layer_dimension = hidden_channels[0]
        self.lin1 = Linear(constants.BERT_HIDDEN_DIM, prev_layer_dimension)
        self.convs = torch.nn.ModuleList()
        self.dropout_prob = dropout_prob
        self.dropouts = torch.nn.ModuleList()
        for layer_dim in hidden_channels[1:]:
            if conv_type == constants.GCN:
                conv = GCNConv(prev_layer_dimension, layer_dim, add_self_loops=False)
            elif conv_type == constants.GAT:
                conv = GATConv((-1, -1), layer_dim, add_self_loops=False)
            elif conv_type == constants.SAGE:
                conv = SAGEConv((-1, -1), layer_dim)
            else:
                raise Exception(f'{conv_type} not implemented')
            prev_layer_dimension = layer_dim
            self.convs.append(conv)
            dropout_layer = nn.Dropout(p=dropout_prob)
            self.dropouts.append(dropout_layer)
        self.lin2 = Linear(prev_layer_dimension, out_channels)
        self.loss = nn.BCEWithLogitsLoss()
        self.max_pooling = use_max_pooling
        self.super_node = super_node

    def forward(self,
                x,
                edge_index,
                batch,
                device):
        """
        :param x: A collection of node and edge data corresponding to a batch of graphs inputted to the model.
        :param edge_index:
        :param device:
        :return: A probability distribution over the output labels. A torch tensor of shape
            [batch_size, num_output_labels].
        """
        # The SBERT tokenizer produces input_ids and an attention mask. BERT produces these tensors as well as a token
        # type ID tensor.
        if self.encoder_type == 'sbert':
            input_ids, attention_mask = torch.hsplit(x, sections=2)
            node_embeddings = self.bert_model({
                constants.INPUT_IDS: torch.squeeze(input_ids, dim=1).long(),
                constants.ATTENTION_MASK: torch.squeeze(attention_mask, dim=1).long()})['sentence_embedding']
        else:
            input_ids, token_type_ids, attention_mask = torch.hsplit(x, sections=3)
            bert_outputs = self.bert_model(
                input_ids=torch.squeeze(input_ids, dim=1).long(),
                token_type_ids=torch.squeeze(token_type_ids, dim=1).long(),
                attention_mask=torch.squeeze(attention_mask, dim=1).long(),
            )
            node_embeddings = bert_outputs['last_hidden_state'][:, 0, :]

        node_embeddings = self.lin1(node_embeddings)
        node_embeddings = node_embeddings.relu()

        for i, conv in enumerate(self.convs):
            # TODO: Think of how to create a graph at the word level as opposed to the sentence level. This
            #  might enable greater utilization of BERT encoding. The challenge is how to build a graph from here
            #  (e.g., how do we define edges in our graph? How do we encode edge or node type? etc...).
            # TODO: Consider using attention v2. This will yield definitive approaches relative to existing GAT.
            # TODO: Keep dimensions the same across layers to enable residual connections.
            # TODO: Consider adding residual embeddings by changing 'node_embeddings = conv(...' to
            #  'node_embeddings += conv(...'
            node_embeddings = conv(node_embeddings, edge_index).relu()
            if self.dropout_prob > 0:
                node_embeddings = self.dropouts[i](node_embeddings)

        if self.super_node:
            indexes = []
            for i in range(batch[-1] + 1):
                index = torch.where(batch == i)
                indexes.append(float(index[0][0]))
            node_embeddings = torch.index_select(
                node_embeddings,
                dim=0,
                index=torch.tensor(indexes,
                                   dtype=torch.int32).to(device)
            )
        else:
            if self.max_pooling:
                node_embeddings = global_max_pool(node_embeddings, batch)
            else:
                node_embeddings = global_mean_pool(node_embeddings, batch)
        node_embeddings = self.lin2(node_embeddings)
        return F.log_softmax(node_embeddings, dim=1)


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels,
                 out_channels,
                 hetero_metadata,
                 conv_type: str,
                 use_frozen_bert: bool = True,
                 encoder_type: str = "sbert",
                 dropout_prob: float = 0.0):
        """

        :param hidden_channels:
        :param out_channels:
        :param hetero_metadata:
        :param conv_type:
        :param use_frozen_bert:
        :param encoder_type:
        :param dropout_prob:
        """
        super().__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'sbert':
            self.bert_model = SentenceTransformer('all-distilroberta-v1')
        else:
            self.bert_model = transformers.BertModel.from_pretrained(constants.BERT_BASE_CASED)
        if use_frozen_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.lin_dict = torch.nn.ModuleDict()
        self.node_types = hetero_metadata[0]
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(constants.BERT_HIDDEN_DIM, hidden_channels[0])


        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for i in range(1, len(hidden_channels)):
            if conv_type == constants.SAGE:
                conv = HeteroConv({
                    (constants.CLAIM, constants.RELATION, constants.CLAIM): SAGEConv(-1, hidden_channels[i]),
                    (constants.CLAIM, constants.RELATION, constants.PREMISE): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.PREMISE, constants.RELATION, constants.CLAIM): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.PREMISE, constants.RELATION, constants.PREMISE): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.PREMISE, constants.RELATION, constants.SUPER_NODE): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.CLAIM, constants.RELATION, constants.SUPER_NODE): SAGEConv((-1, -1), hidden_channels[i]),
                }, aggr='sum')
            elif conv_type == constants.GAT:
                conv = HeteroConv({
                    (constants.CLAIM, constants.RELATION, constants.CLAIM): GATConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.CLAIM, constants.RELATION, constants.PREMISE): GATConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.PREMISE, constants.RELATION, constants.CLAIM): GATConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.PREMISE, constants.RELATION, constants.PREMISE): GATConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.PREMISE, constants.RELATION, constants.SUPER_NODE): GATConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.CLAIM, constants.RELATION, constants.SUPER_NODE): GATConv(-1, hidden_channels[i], add_self_loops=False),
                }, aggr='sum')
            else:
                raise Exception(f'{conv_type} not implemented')
            self.convs.append(conv)
            dropout_layer = nn.Dropout(p=dropout_prob)
            self.dropouts.append(dropout_layer)

        self.lin = Linear(hidden_channels[-1], out_channels)

    def forward(self,
                x_dict,
                edge_index_dict,
                device):
        """

        :param x_dict:
        :param edge_index_dict:
        :param device:
        :return:
        """
        for node_type, x in x_dict.items():
            if self.encoder_type == 'sbert':
                input_ids, attention_mask = torch.hsplit(x, sections=2)
                x = self.bert_model({
                    constants.INPUT_IDS: torch.squeeze(input_ids, dim=1).long(),
                    constants.ATTENTION_MASK: torch.squeeze(attention_mask, dim=1).long()})['sentence_embedding']
            else:
                input_ids, token_type_ids, attention_mask = torch.hsplit(x, sections=3)
                bert_outputs = self.bert_model(
                    input_ids=torch.squeeze(input_ids, dim=1).long(),
                    token_type_ids=torch.squeeze(token_type_ids, dim=1).long(),
                    attention_mask=torch.squeeze(attention_mask, dim=1).long(),
                )
                x = bert_outputs['last_hidden_state'][:, 0, :]

            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            x_dict = {key: self.dropouts[i](x) for key, x in x_dict.items()}
        return self.lin(x_dict[constants.SUPER_NODE])


# TODO: Move this to utils.
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of parameters within the provided model.

    :param model: The model in which we are trying to count the number of parameters.
    :return: The number of trainable parameters within our model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
