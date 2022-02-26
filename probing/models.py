from __future__ import annotations

import typing

import numpy as np
import sklearn
import torch
import wandb
import metrics

import constants


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

    def train_probe(self,
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.optimizer.Optimizer,
                    num_labels: int,
                    loss_function: torch.nn.BCELoss | torch.nn.CrossEntropyLoss,
                    num_epochs: int,
                    scheduler=None,
                    wandb_run: wandb.sdk.wandb_run.Run = None):
        """Train the probing model on a classification task given the provided training set and parameters.

        :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance for some
            premise mode.
        :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
        :param num_labels: An integer representing the output space (number of labels) for the probing classification
            problem.
        :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
        :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
        :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
        :param wandb_run: A wandb.Run instance used to log metrics through the wandb interface.
        """
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()
                targets = data[constants.LABEL]
                outputs = self(data[constants.HIDDEN_STATE])
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

    def eval_probe(self,
                   test_loader: torch.utils.data.DataLoader,
                   num_labels: int) -> typing.Mapping[str, float]:
        """Evaluate the trained classification probe on a held out test set.

        :param test_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance for some
            premise mode.
        :param num_labels: The number of labels for the probing classification problem.
        :return: A 2-tuple containing ('confusion_matrix', 'classification_report'). 'confusion_matrix' is derived from
        'sklearn.metrics.confusion_matrix' while 'classification_report' is derived from
        'sklearn.metrics.classification_report'.
        """
        preds_list = []
        targets_list = []
        self.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                outputs = self(data[constants.HIDDEN_STATE])
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
