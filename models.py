from __future__ import annotations

import argparse
import copy
import math
import os
import typing

import datasets
import torch
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv, global_mean_pool,  GATConv, Linear, SAGEConv, global_max_pool, HeteroConv, HGTConv
import torch.nn as nn
import numpy as np
import tqdm

import utils
import wandb

import constants
import metrics
from cmv_modes import preprocessing_knowledge_graph
from metrics import compute_metrics

"""
Example usage: 
srun --gres=gpu:1 -p nlp python models.py \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_gamma 0.9 \
    --test_percent 0.1 \
    --val_percent 0.1 \
    --debug '' \
    --use_k_fold_cross_validation 'True' \
    --num_cross_validation_splits 5 \
    --dropout_probability 0.3 \
    --encoder_type 'sbert' \
    --seed 42 
"""


parser = argparse.ArgumentParser(
    description='Process flags for experiments on processing graphical representations of arguments through GNNs.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=100,
                    help="The number of training rounds over the knowledge graph dataset.")
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during both training and evaluation.")
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-4,
                    help="The learning rate used by the GCN+BERT model during training.")
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-4,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--test_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the test set.')
parser.add_argument('--val_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the validation set.')
parser.add_argument('--debug',
                    type=bool,
                    default=False,
                    help="Work in debug mode")
parser.add_argument('--use_k_fold_cross_validation',
                    type=bool,
                    default=False,
                    help="True if we intend to perform cross validation on the dataset. False otherwise. Using this"
                         "option is advised if the dataset is small.")
parser.add_argument('--max_num_rounds_no_improvement',
                    type=int,
                    default=10,
                    help="The permissible number of validation set evaluations in which the desired metric does not "
                         "improve. If the desired validation metric does not improve within this number of evaluation "
                         "attempts, then early stopping is performed.")
parser.add_argument('--metric_for_early_stopping',
                    type=str,
                    default=constants.ACCURACY,
                    help="The metric to track on the validation set as part of early stopping. If this metric does not "
                         "improve after a few evalution steps on the validation set, we perform early stopping.")
parser.add_argument('--num_cross_validation_splits',
                    type=int,
                    default=5,
                    help="The number of cross validation splits we perform as part of k-fold cross validation.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="The seed used for random number generation and sampling.")
parser.add_argument('--scheduler_gamma',
                    type=float,
                    default=0.9,
                    help="Gamma value used for the learning rate scheduler during training.")
parser.add_argument('--dropout_probability',
                    type=float,
                    default=0.3,
                    help="The dropout probability across each layer of the MLP in the baseline model.")
parser.add_argument('--fold_index',
                    type=int,
                    default=0,
                    help="The partition index of the held out data as part of k-fold cross validation.")
parser.add_argument('--encoder_type',
                    type=str,
                    default='sbert',
                    help="The model used to both tokenize and encode the textual context of argumentative "
                         "prepositions.")


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
                 hidden_channels: list,
                 conv_type: str,
                 use_frozen_bert: bool = True,
                 use_max_pooling: bool = True,
                 encoder_type: str = "sbert",
                 dropout_prob: float = 0.0):
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

    def forward(self, x, edge_index, batch=None):
        """
        :param data: A collection of node and edge data corresponding to a batch of graphs inputted to the model.
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
            node_embeddings = conv(node_embeddings, edge_index).relu()
            node_embeddings = self.dropouts[i](node_embeddings)

        if self.max_pooling:
            node_embeddings = global_max_pool(node_embeddings, batch)
        else:
            node_embeddings = global_mean_pool(node_embeddings, batch)
        node_embeddings = self.lin2(node_embeddings)
        return F.log_softmax(node_embeddings, dim=1)


# TODO: Document this entire class, and all of it's associated methods.
class HGT(torch.nn.Module):
    def __init__(self,
                 hidden_channels: typing.List[int],
                 out_channels: int,
                 hetero_metadata: typing.Tuple[typing.List[str], typing.List[typing.Tuple[str, str, str]]],
                 use_frozen_bert: bool = True,
                 use_max_pooling: bool = True):
        """
        Initialize an HGT model on top of BERT embeddings to predict argument persuasiveness.

        :param hidden_channels: A list of the hidden dimensions used by this model. The first dimension is used to
            convert from BERT's hidden dimension to one more suitable for graph convolutions. The remaining dimensions
            reflect the outputs of the GNN's convolutional layers. There is also a final linear transformation
            from the final hidden dimension to the label-space dimensionality after pooling/super-node aggregation.
        :param out_channels: The label-space dimensionality of the downstream task. The number of possible labels this
            model is tasked to predict among.
        :param hetero_metadata: The input graph's heterogeneous meta-data, i.e. its node and edge types.
        :param use_frozen_bert: True if we intend to use a frozen BERT model to produce node embeddings. False if BERT's
            weights should be updated while training the GNN (such that BERT's embeddings would evolve beyond their
            pre-trained values).
        :param use_max_pooling: True if we intend to use max pooling to aggregate node representations.
            Nodes are aggregated as part of graph-classification before projecting to the label-space dimensionality.
        """
        # TODO: Add comments to document the flow of this initialization.
        super().__init__()
        self.bert_model = transformers.BertModel.from_pretrained(constants.BERT_BASE_CASED)
        if use_frozen_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.node_types = hetero_metadata[0]

        self.lin_dict = torch.nn.ModuleDict()
        prev_layer_dimension = hidden_channels[0]
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(constants.BERT_HIDDEN_DIM, hidden_channels[0])

        self.convs = torch.nn.ModuleList()
        for hidden_layer_dim in hidden_channels[1:]:
            conv = HGTConv(prev_layer_dimension, hidden_layer_dim, hetero_metadata, group='sum')
            self.convs.append(conv)
            prev_layer_dimension = hidden_layer_dim

        self.lin = Linear(prev_layer_dimension, out_channels)
        self.max_pooling = use_max_pooling

    # TODO: Annotate parameter types and add documentation to this function.
    def forward(self,
                x_dict,
                edge_index_dict,
                batch=None):
        """

        :param x_dict:
        :param edge_index_dict:
        :param batch:
        :return:
        """
        for node_type, x in x_dict.items():
            input_ids, token_type_ids, attention_mask = torch.hsplit(x, sections=3)
            bert_outputs = self.bert_model(
                input_ids=torch.squeeze(input_ids, dim=1).long(),
                token_type_ids=torch.squeeze(token_type_ids, dim=1).long(),
                attention_mask=torch.squeeze(attention_mask, dim=1).long(),
            )
            x = bert_outputs['last_hidden_state'][:, 0, :]
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        out = 0
        for node_type, x in x_dict.items():
            if self.max_pooling:
                x_dict[node_type] = global_max_pool(x_dict[node_type], batch)
            else:
                x_dict[node_type] = global_mean_pool(x_dict[node_type], batch)
            # TODO: Introduce functionality for mean/max/sum aggregation of nodes across distinct types.
            out += x_dict[node_type] / len(self.node_types)
        out = self.lin(out)
        return F.log_softmax(out, dim=1)



class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, hetero_metadata,conv_type: str, use_frozen_bert: bool = True ):
        super().__init__()
        self.bert_model = transformers.BertModel.from_pretrained(constants.BERT_BASE_CASED)
        if use_frozen_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.lin_dict = torch.nn.ModuleDict()
        self.node_types = hetero_metadata[0]
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(constants.BERT_HIDDEN_DIM, hidden_channels[0])

        self.convs = torch.nn.ModuleList()
        for i in range(len(hidden_channels)):
            if conv_type == constants.SAGE:
                conv = HeteroConv({
                    (constants.CLAIM, constants.RELATION, constants.CLAIM): SAGEConv(-1, hidden_channels[i]),
                    (constants.CLAIM, constants.RELATION, constants.PREMISE): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.PREMISE, constants.RELATION, constants.CLAIM): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.PREMISE, constants.RELATION, constants.PREMISE): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.PREMISE, constants.RELATION, constants.SUPER_NODE): SAGEConv((-1, -1), hidden_channels[i]),
                    (constants.CLAIM, constants.RELATION, constants.SUPER_NODE): SAGEConv((-1, -1), hidden_channels[i]),
                }, aggr='sum')
            elif conv_type == constants.GCN:
                conv = HeteroConv({
                    (constants.CLAIM, constants.RELATION, constants.CLAIM): GCNConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.CLAIM, constants.RELATION, constants.PREMISE): GCNConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.PREMISE, constants.RELATION, constants.CLAIM): GCNConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.PREMISE, constants.RELATION, constants.PREMISE): GCNConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.PREMISE, constants.RELATION, constants.SUPER_NODE): GCNConv(-1, hidden_channels[i], add_self_loops=False),
                    (constants.CLAIM, constants.RELATION, constants.SUPER_NODE): GCNConv(-1, hidden_channels[i], add_self_loops=False),
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

        self.lin = Linear(hidden_channels[-1], out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            input_ids, token_type_ids, attention_mask = torch.hsplit(x, sections=3)
            bert_outputs = self.bert_model(
                input_ids=torch.squeeze(input_ids, dim=1).long(),
                token_type_ids=torch.squeeze(token_type_ids, dim=1).long(),
                attention_mask=torch.squeeze(attention_mask, dim=1).long(),
            )
            x = bert_outputs['last_hidden_state'][:, 0, :]
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict[constants.SUPER_NODE])

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of parameters within the provided model.

    :param model: The model in which we are trying to count the number of parameters.
    :return: The number of trainable parameters within our model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EncoderBaseline(torch.nn.Module):
    def __init__(self,
                 use_frozen_encoder: bool,
                 mlp_layers: torch.nn.Sequential,
                 device: torch.device,
                 encoder_type: str = "bert"):
        """
        Initialize a persuasiveness prediction baseline model.

        Initialize a transformer encoder model (followed by an MLP) used to predict argument persuasiveness given
        tokenized sequential inputs of the form [argument context, argument]. Within CMV, argument context refers to
        the title and OP text, and the argument corresponds to the persuasive reply (which may or may not have earned a
        Delta).

        :param use_frozen_encoder: True we we intend to freeze the BERT encoder which precedes the MLP. False otherwise.
        :param mlp_layers: The torch.nn layers corresponding to the layers of the baseline's NLP.
        :param device: The device on which the model and data are stored.
        :param encoder_type: The name of the encoder model used to create a representation of inputted arguments.
        """
        super(EncoderBaseline, self).__init__()
        self.encoder_type = encoder_type
        
        # Initialize appropriate encoder.
        if self.encoder_type == "bert":
            self._encoder_model = transformers.BertForSequenceClassification.from_pretrained(
                constants.BERT_BASE_CASED,
                num_labels=constants.NUM_LABELS).to(device)
        elif self.encoder_type == "sbert":
            self._encoder_model = SentenceTransformer("all-distilroberta-v1").to(device)
        else:
            raise RuntimeError(f"Unsupported encoder type: {self.encoder_type}")
        
        if use_frozen_encoder:
            for param in self._encoder_model.parameters():
                param.requires_grad = False
        self._mlp = mlp_layers.to(device)
        self.device = device

    def forward(self, encoder_inputs):
        """Pass tokenized inputs through the model as part of the forward pass.

        :param encoder_inputs: A dictionary mapping string keys to tensor values corresponding to encoder inputs. Each
            input tensor has shape [batch_size, sequence_length]
        :return: A tensor with shape [batch_size, num_labels].
        """
        # Compute hidden representations for context + argument pairs.
        if self.encoder_type == "bert":
            encoder_outputs = self._encoder_model.forward(
                input_ids=encoder_inputs[constants.INPUT_IDS],
                attention_mask=encoder_inputs[constants.ATTENTION_MASK],
                token_type_ids=encoder_inputs[constants.TOKEN_TYPE_IDS],
                output_hidden_states=True,
            )
            sequence_embeddings = encoder_outputs.hidden_states[-1].to(self.device)[:, 0, :]
        else:
            sequence_embeddings = self._encoder_model({
                constants.INPUT_IDS: encoder_inputs[constants.INPUT_IDS],
                constants.ATTENTION_MASK: encoder_inputs[constants.ATTENTION_MASK]})['sentence_embedding']

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
                                    torch.optim.lr_scheduler.ConstantLR] = None):
        """
        Train the baseline model on the persuasiveness prediction task.

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
        self.train().to(device)
        highest_accuracy = 0
        lowest_loss = math.inf
        num_rounds_no_improvement = 0
        epoch_with_optimal_performance = 0
        best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
        utils.ensure_dir_exists(best_model_dir_path)
        best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_probe.pt')
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_acc = 0.0
            train_num_batches = 0
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                targets = data[constants.LABEL].to(device)
                encoder_inputs = {
                    constants.INPUT_IDS: data[constants.INPUT_IDS].to(self.device),
                    constants.ATTENTION_MASK: data[constants.ATTENTION_MASK].to(self.device)
                }
                if self.encoder_type == "bert":
                    encoder_inputs[constants.TOKEN_TYPE_IDS] = data[constants.TOKEN_TYPE_IDS].to(self.device)

                outputs = self(encoder_inputs)
                preds = torch.argmax(outputs, dim=1).to(device)
                loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
                loss.backward()
                optimizer.step()
                num_correct_preds = (preds == targets).sum().float()
                accuracy = num_correct_preds / targets.shape[0] * 100
                train_num_batches += 1
                train_loss += loss.item()
                train_acc += accuracy
            learning_rate = scheduler.get_last_lr()[0]
            scheduler.step()
            wandb.log({
                f"{constants.TRAIN} {constants.ACCURACY}": train_acc / train_num_batches,
                f"{constants.TRAIN} {constants.EPOCH}": epoch,
                f"{constants.TRAIN} {constants.LOSS}": train_loss / train_num_batches,
                "learning rate": learning_rate
            })

            # Perform Evaluation
            self.eval()
            with torch.no_grad():
                validation_loss = 0.0
                validation_acc = 0.0
                validation_num_batches = 0
                for i, data in enumerate(validation_loader, 0):
                    targets = data[constants.LABEL].to(device)
                    encoder_inputs = {
                        constants.INPUT_IDS: data[constants.INPUT_IDS].to(self.device),
                        constants.ATTENTION_MASK: data[constants.ATTENTION_MASK].to(self.device)
                    }
                    if self.encoder_type == "bert":
                        encoder_inputs[constants.TOKEN_TYPE_IDS] = data[constants.TOKEN_TYPE_IDS].to(self.device)
                    outputs = self(encoder_inputs).to(device)
                    preds = torch.argmax(outputs, dim=1).to(device)
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
                    print(f'Performing early stopping after {epoch} epochs.\n')
                    self.load_state_dict(torch.load(best_model_path))
                    break

        print(f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
        self.load_state_dict(torch.load(best_model_path))

    def evaluate(self,
                 dataloader: torch.data.DataLoader,
                 split_name: str,
                 device: torch.device) -> typing.Mapping[str, float]:
        """Evaluate a trained model on a dataset contained within the provided data loader.

        :param dataloader: The dataloader instance containing the dataset on which the model is evaluated.
        :param split_name: One of {"train", "validation", "test"}
        :param device: The device
        :return: A dictionary mapping metric names to their float values.
        """
        self.eval()
        self.to(device)
        with torch.no_grad():
            preds_list = []
            targets_list = []
            for batch in tqdm.tqdm(dataloader):
                targets = batch[constants.LABEL].to(device)
                encoder_inputs = {
                    constants.INPUT_IDS: batch[constants.INPUT_IDS].to(self.device),
                    constants.ATTENTION_MASK: batch[constants.ATTENTION_MASK].to(self.device)
                }
                if self.encoder_type == "bert":
                    encoder_inputs[constants.TOKEN_TYPE_IDS] = batch[constants.TOKEN_TYPE_IDS].to(self.device)
                outputs = self(encoder_inputs).to(device)
                preds = torch.argmax(outputs, dim=1).cpu()
                preds_list.append(preds)
                targets_list.append(targets.cpu())
            preds_list = np.concatenate(preds_list)
            targets_list = np.concatenate(targets_list)
            eval_metrics = metrics.compute_metrics(num_labels=constants.NUM_LABELS,
                                                   preds=preds_list,
                                                   targets=targets_list,
                                                   split_name=split_name)
            for metric_name, metric_value in eval_metrics.items():
                wandb.summary[f"eval_{metric_name}"] = metric_value
            return eval_metrics


if __name__ == '__main__':

    args = parser.parse_args()
    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')
    utils.set_seed(args.seed)

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

    # Tokenize encoder inputs
    columns = [constants.INPUT_IDS, constants.ATTENTION_MASK, constants.LABEL]
    if args.encoder_type == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)
        columns.append(constants.TOKEN_TYPE_IDS)
    elif args.encoder_type == "sbert":
        tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
    else:
        raise RuntimeError(f"invalid encoder type: {args.encoder_type}")
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    tokenized_inputs = tokenizer(op_text, reply_text, padding=True, truncation=True)
    transformers.logging.set_verbosity(verbosity)

    dataset_dict = {input_name: input_value for input_name, input_value in tokenized_inputs.items()}
    dataset_dict[constants.LABEL] = labels
    dataset = datasets.Dataset.from_dict(dataset_dict)
    dataset.set_format(type='torch', columns=columns)
    dataset = dataset.shuffle()
    print(f'Entire dataset positive_labels: {sum(dataset[constants.LABEL].numpy() == 1)}\n'
          f'Entire dataset negative_labels: {sum(dataset[constants.LABEL].numpy() == 0)}')

    # TODO: Create an MLP as a function of a desired model size. This should correspond with the size of a parallel
    #  GNN model, and be passed via flag.
    # TODO: Enable configuration of the MLP architecture through flag parameters.
    mlp = torch.nn.Sequential(
        torch.nn.Linear(constants.BERT_HIDDEN_DIM, constants.BERT_HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout_probability),
        torch.nn.Linear(constants.BERT_HIDDEN_DIM, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout_probability),
        torch.nn.Linear(512, constants.NUM_LABELS),
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder_baseline = EncoderBaseline(
        use_frozen_encoder=True,
        mlp_layers=mlp,
        device=device,
        encoder_type=args.encoder_type,
    )
    model_name = f'{args.encoder_type}+mlp baseline on {constants.BINARY_CMV_DELTA_PREDICTION}'
    experiment_name = f"{model_name} " \
                      f"(seed: #{args.seed}, " \
                      f"lr: {args.learning_rate}, " \
                      f"gamma: {args.scheduler_gamma}, " \
                      f"dropout: {args.dropout_probability}, " \
                      f"w_d: {args.weight_decay})"

    if args.use_k_fold_cross_validation:
        num_cross_validation_splits = args.num_cross_validation_splits
        train_metrics = []
        validation_metrics = []
        test_metrics = []
        shards = [dataset.shard(num_cross_validation_splits, i, contiguous=True)
                  for i in range(num_cross_validation_splits)]
        for validation_set_index in range(num_cross_validation_splits):
            split_model = copy.deepcopy(encoder_baseline).to(device)
            validation_and_test_sets = shards[validation_set_index].train_test_split(
                args.val_percent / (args.val_percent + args.test_percent))
            validation_set = validation_and_test_sets[constants.TRAIN]
            print(f'Validation set ({validation_set_index}) positive labels: '
                  f'{sum(validation_set[constants.LABEL].numpy() == 1)}')
            print(f'Validation set ({validation_set_index}) negative labels: '
                  f'{sum(validation_set[constants.LABEL].numpy() == 0)}')
            test_set = validation_and_test_sets[constants.TEST]
            training_set = datasets.concatenate_datasets(
                shards[0:validation_set_index] + shards[validation_set_index + 1:]).shuffle()
            run = wandb.init(
                project="persuasive_arguments",
                entity="persuasive_arguments",
                group=experiment_name,
                config=args,
                name=f"{experiment_name} [{validation_set_index}]",
                dir='.')
            # TODO: Create a helper utility function which generates the optimizer and scheduler given the necessary
            #  parameters.
            optimizer = torch.optim.Adam(
                split_model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
            train_loader, validation_loader, test_loader = utils.create_data_loaders(
                training_set=training_set,
                validation_set=validation_set,
                test_set=test_set,
                batch_size=args.batch_size)
            split_model.fit(
                train_loader=train_loader,
                validation_loader=validation_loader,
                num_labels=2,
                loss_function=torch.nn.BCELoss(),
                num_epochs=args.num_epochs,
                optimizer=optimizer,
                max_num_rounds_no_improvement=args.max_num_rounds_no_improvement,
                metric_for_early_stopping=constants.ACCURACY,
                scheduler=torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=args.scheduler_gamma
                )
            )
            # TODO: Implement helper function to aggregate metrics across folds during k fold cross validation.
            train_metrics.append(
                split_model.evaluate(
                    dataloader=train_loader,
                    split_name=constants.TRAIN,
                    device=device)
            )
            validation_metrics.append(
                split_model.evaluate(dataloader=validation_loader,
                                     split_name=constants.VALIDATION,
                                     device=device)
            )
            test_metrics.append(
                split_model.evaluate(
                    dataloader=test_loader,
                    split_name=constants.TEST,
                    device=device
                )
            )
            run.finish()

        validation_metric_aggregates = utils.aggregate_metrics_across_splits(validation_metrics)
        train_metric_aggregates = utils.aggregate_metrics_across_splits(train_metrics)
        test_metric_aggregates = utils.aggregate_metrics_across_splits(test_metrics)
        print(f'\n*** Train Metrics: ***')
        train_metric_averages, train_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=train_metric_aggregates,
            split_name=constants.TRAIN,
            print_results=True)
        print(f'\n*** Validation Metrics: ***')
        validation_metric_averages, validation_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=validation_metric_aggregates,
            split_name=constants.VALIDATION,
            print_results=True)
        print(f'\n*** Test Metrics: ***')
        test_metric_averages, test_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=test_metric_aggregates,
            split_name=constants.TEST,
            print_results=True)
    else:
        partitioned_dataset = dataset.train_test_split(args.val_percent + args.test_percent)
        training_set = partitioned_dataset[constants.TRAIN]
        hidden_sets = partitioned_dataset[constants.TEST]
        partitioned_hidden_sets = hidden_sets.train_test_split(
            args.val_percent / (args.val_percent + args.test_percent))
        validation_set = partitioned_hidden_sets[constants.TRAIN]
        test_set = partitioned_hidden_sets[constants.TEST]
        wandb.init(
            project="persuasive_arguments",
            entity="persuasive_arguments",
            group=experiment_name,
            config=args,
            name=f"{experiment_name} [{args.fold_index}]",
            dir='.')
        config = wandb.config
        optimizer = torch.optim.Adam(
            encoder_baseline.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        train_loader, validation_loader, test_loader = utils.create_data_loaders(
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            batch_size=args.batch_size)
        encoder_baseline.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            num_labels=2,
            loss_function=torch.nn.BCELoss(),
            num_epochs=args.num_epochs,
            optimizer=optimizer,
            max_num_rounds_no_improvement=args.max_num_rounds_no_improvement,
            metric_for_early_stopping=args.metric_for_early_stopping,
            scheduler=torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=args.scheduler_gamma
            )
        )
        print(metrics.perform_evaluation_on_splits(
            eval_fn=encoder_baseline.evaluate,
            device=device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader
        ))

    # TODO: Implement an option for k-fold cross validation during wandb sweeps via grouping as is done in
    #  train_and_eval.py

    # TODO: Consolidate code across this entire module.
