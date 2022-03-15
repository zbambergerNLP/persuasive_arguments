import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
import torch_geometric.loader as geom_data
import wandb
from data_loaders import CMVKGDataset
from models import GCNWithBertEmbeddings
import constants
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Process flags for experiments on processing graphical representations of arguments through GNNs.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=10,
                    help="The number of training rounds over the knowledge graph dataset.")
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during both training and evaluation.")
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-2,
                    help="The learning rate used by the GCN+BERT model during training.")
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-4,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--gcn_hidden_layer_dim',
                    type=int,
                    default=128,
                    help="The dimensionality of the hidden layer within the GCN component of the GCN+BERT model.")
parser.add_argument('--test_percent',
                    type=float,
                    default=0.2,
                    help='The proportion (ratio) of samples dedicated to the test set. The remaining examples are used'
                         'for training.')


def train(model: GCNWithBertEmbeddings,
          dl: geom_data.DataLoader,
          epochs: int,
          optimizer: torch.optim.Optimizer):
    """

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param dl: A torch geometric data loader used to feed batches from the train and test sets to the model.
    :param epochs: The number of iterations over the training set during model training.
    :param optimizer: The torch optimizer used for weight updates during trianing.
    :return: A trained model.
    """
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for sampled_data in tqdm(dl):
            optimizer.zero_grad()
            out = model(sampled_data)
            preds = torch.argmax(out, dim=1)
            y_one_hot = F.one_hot(sampled_data.y, 2)
            loss = model.loss(out.float(),y_one_hot.float() )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_correct_preds = (preds == sampled_data.y).sum().float()
            accuracy = num_correct_preds /sampled_data.y.shape[0] * 100
            epoch_acc += accuracy
            num_batches += 1
        wandb.log({constants.ACCURACY: epoch_acc / num_batches,
                   constants.EPOCH: epoch,
                   constants.LOSS: epoch_loss / num_batches})
    return model


def eval(model: GCNWithBertEmbeddings,
         dataset: CMVKGDataset):
    """

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param dataset: A CMVKGDataLoader instance
    :return:
    """
    model.eval()
    acc = 0.0
    num_batches = 0
    with torch.no_grad():
        for sampled_data in tqdm(dataset):
            out = model(sampled_data)
            preds = torch.argmax(out, dim=1)
            num_correct_preds = (preds == sampled_data.y).sum().float()
            accuracy = num_correct_preds / sampled_data.y.shape[0] * 100
            acc += accuracy
            num_batches += 1
        acc = acc /num_batches
    print(f'Accuracy: {acc:.4f}')


def create_dataloaders(graph_dataset: Dataset,
                       batch_size: int,
                       test_percent: float,
                       num_workers: int = 0):
    """

    :param graph_dataset: A 'CMVKGDataset' instance whose examples correspond to knowledge graphs of persuasive
        arguments.
    :param batch_size: The number of examples processed in a single batch as part of training.
    :param test_percent: The ratio of the original examples dedicated towards a test set. The remaining examples are
        used as part of model training.
    :param num_workers: The number of workers used during training.
    :return:
    """
    num_of_examples = len(graph_dataset.dataset)
    train_len = int((1.0 - test_percent) * num_of_examples)

    indexes = random.sample(range(num_of_examples), num_of_examples)
    train_indexes = indexes[:train_len]
    test_indexes = indexes[train_len:]
    dl_train = geom_data.DataLoader(graph_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=SubsetRandomSampler(train_indexes))
    dl_test = geom_data.DataLoader(graph_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   sampler=SubsetRandomSampler(test_indexes))
    return dl_train, dl_test


if __name__ == '__main__':

    args = parser.parse_args()
    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')

    num_node_features = constants.BERT_HIDDEN_DIM
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_layer_dim=args.gcn_hidden_layer_dim,
    )
    with wandb.init(project="persuasive_arguments", config=config):
        kg_dataset = CMVKGDataset(
            current_path + "/cmv_modes/change-my-view-modes-master",
            version=constants.v2_path,
            debug=False)
        config = wandb.config
        dl_train, dl_test = create_dataloaders(kg_dataset,
                                               batch_size=config.batch_size,
                                               test_percent=args.test_percent)
        model = GCNWithBertEmbeddings(num_node_features,
                                      num_classes=2,
                                      hidden_layer_dim=config.hidden_layer_dim)
        wandb.watch(model, log='all', log_freq=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        model = train(model, dl_train, args.num_epochs, optimizer)
        eval(model, dl_test)
