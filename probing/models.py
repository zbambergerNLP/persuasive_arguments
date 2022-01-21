
import numpy as np
import sklearn
import torch

import constants


class MLP(torch.nn.Module):
    def __init__(self, num_labels):
        """
        Initialize an MLP consisting of one linear layer that maintain the hidden dimensionality, and then a projection
        into num_labels dimensions. The first linear layer has a ReLU non-linearity, while the final (logits)
        layer has a softmax activation.
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

    def train_probe(self, train_loader, optimizer, num_labels, loss_function, num_epochs=5, scheduler=None):
        """

        :param train_loader: A 'torch.utils.data.DataLoader' wrapping either a 'preprocessing.CMVPremiseModes' dataset
            or a 'preprocessing.CMVDataset' instance for some premise mode.
        :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
        :param num_labels: An integer representing the output space (number of labels) for the probing classification
            problem.
        :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
        :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
        :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
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
            print(
                f'Epoch {epoch + 1:03}: | '
                f'Loss: {epoch_loss / num_batches:.5f} |'
                f'Acc: {epoch_acc / num_batches:.3f}'
            )

    def eval_probe(self, test_loader):
        """

        :param test_loader: A 'torch.utils.data.DataLoader' wrapping either a 'preprocessing.CMVPremiseModes' dataset
            or a 'preprocessing.CMVDataset' instance for some premise mode.
        :return: A 2-tuple containing ('confusion_matrix', 'classification_report'). 'confusion_matrix' is derived from
        'sklearn.metrics.confusion_matrix' while 'classification_report' is derived from
        'sklearn.metrics.classification_report'.
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