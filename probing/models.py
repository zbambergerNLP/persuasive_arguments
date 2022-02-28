from __future__ import annotations

import typing

import numpy as np
import torch
import wandb
import metrics

import constants


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_labels):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class MLP(torch.nn.Module):
    def __init__(self, num_labels):
        """
        Initialize an MLP consisting of one linear layer that maintain the hidden dimensionality, and then a projection
        into num_labels dimensions. The first linear layer has a ReLU non-linearity, while the final (logits)
        layer has a softmax activation.

        :param num_labels: The output dimensionality of the MLP that corresponds to the number of possible labels for
            the classification task.
        """
        super(MLP, self).__init__()
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
