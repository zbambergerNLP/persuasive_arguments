from __future__ import annotations

import typing

import torch
import torch.nn.functional as F
import torch_geometric.data
import transformers
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import numpy as np
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
                optimizer: torch.optim.optimizer.Optimizer,
                num_labels: int,
                loss_function: torch.nn.BCELoss | torch.nn.CrossEntropyLoss,
                num_epochs: int,
                scheduler=None):
    """Train the probing model on a classification task given the provided training set and parameters.

    :param probing_model: A torch.nn.Module on which we are performing training.
    :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance for some
        premise mode.
    :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
    :param num_labels: An integer representing the output space (number of labels) for the probing classification
        problem.
    :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
    :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
    :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
    """
    probing_model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            targets = data[constants.LABEL]
            outputs = probing_model(data[constants.HIDDEN_STATE])
            preds = torch.argmax(outputs, dim=1)
            loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_correct_preds = (preds == targets).sum().float()
            accuracy = num_correct_preds / targets.shape[0] * 100
            epoch_acc += accuracy
            num_batches += 1
        scheduler.step()
        wandb.log({constants.ACCURACY: epoch_acc / num_batches,
                   constants.EPOCH: epoch,
                   constants.LOSS: epoch_loss / num_batches})
    return probing_model


def eval_probe(probing_model: torch.nn.Module,
               test_loader: torch.utils.data.DataLoader,
               num_labels: int) -> typing.Mapping[str, float]:
    """Evaluate the trained classification probe on a held out test set.

    :param probing_model: A torch.nn.Module on which we are performing evaluation.
    :param test_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance for some
        premise mode.
    :param num_labels: The number of labels for the probing classification problem.
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
                                           targets=targets_list)
    return eval_metrics


class BaselineLogisticRegression(torch.nn.Module):
    """

    """

    def __init__(self, num_features, num_labels):
        """

        :param num_features:
        :param num_labels:
        """
        super(BaselineLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_labels)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        outputs = torch.sigmoid(self.linear(x.float()))
        return outputs

    def fit(self,
            train_loader,
            num_labels: int,
            loss_function: typing.Union[torch.nn.BCELoss, torch.nn.CrossEntropyLoss],
            num_epochs: int = 100,
            optimizer: torch.optim.Optimizer = torch.optim.SGD,
            scheduler: typing.Union[torch.optim.lr_scheduler.ExponentialLR,
                                    torch.optim.lr_scheduler.ConstantLR] = None,
            validation_loader=None
            ):
        """

        :param train_loader:
        :param num_labels:
        :param loss_function:
        :param num_epochs:
        :param optimizer:
        :param scheduler:
        :param validation_loader:
        :return:
        """
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
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
                num_batches += 1
                epoch_loss += loss.item()
                epoch_acc += accuracy
            wandb.log({f'training_{constants.ACCURACY}': epoch_acc / num_batches,
                       f'training_{constants.EPOCH}': epoch,
                       f'training_{constants.LOSS}': epoch_loss / num_batches})
            scheduler.step()

            # Perform evaluation.
            if epoch % 5 == 0:
                epoch_loss = 0.0
                epoch_acc = 0.0
                num_batches = 0
                self.eval()
                for i, data in enumerate(validation_loader):
                    targets = data[constants.LABEL]
                    outputs = self(data['features'])
                    preds = torch.argmax(outputs, dim=1)
                    loss = loss_function(outputs, torch.nn.functional.one_hot(targets, num_labels).float())
                    num_correct_preds = (preds == targets).sum().float()
                    accuracy = num_correct_preds / targets.shape[0] * 100
                    num_batches += 1
                    epoch_loss += loss.item()
                    epoch_acc += accuracy
                wandb.log({f'validation_{constants.ACCURACY}': epoch_acc / num_batches,
                           f'validation_{constants.EPOCH}': epoch,
                           f'validation_{constants.LOSS}': epoch_loss / num_batches})

    def evaluate(self, test_loader, num_labels):
        """

        :param test_loader:
        :param num_labels:
        :return:
        """
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
                                       targets=targets_list)
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


if __name__ == '__main__':
    model = GCNWithBertEmbeddings(256, 2, 16)
    print(model)
    pass

