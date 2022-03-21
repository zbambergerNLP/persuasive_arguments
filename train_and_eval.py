import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
import torch_geometric.loader as geom_data

import utils
import wandb
from data_loaders import CMVKGDataset
from models import GCNWithBertEmbeddings
import constants
import argparse
from tqdm import tqdm


"""
Example command:
srun --gres=gpu:1 -p nlp python3 train_and_eval.py \
    --num_epochs 30 \
    --batch_size 4 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --gcn_hidden_layer_dim 128 \
    --test_percent 0.1 \
    --val_percent 0.1 \
    --rounds_between_evals 5
    
"""


parser = argparse.ArgumentParser(
    description='Process flags for experiments on processing graphical representations of arguments through GNNs.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=30,
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
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the test set.')
parser.add_argument('--val_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the validation set.')
parser.add_argument('--rounds_between_evals',
                    type=int,
                    default=5,
                    help="An integer denoting the number of epcohs that occur between each evaluation run.")


def train(model: GCNWithBertEmbeddings,
          training_loader: geom_data.DataLoader,
          validation_loader: geom_data.DataLoader,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          rounds_between_evals: int) -> GCNWithBertEmbeddings:
    """Train a GCNWithBERTEmbeddings model on examples consisting of persuasive argument knowledge graphs.

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param training_loader: A torch geometric data loader used to feed batches from the training set to the model.
    :param validation_loader: A torch geometric data loader used to feed batches from the validation set to the model.
    :param epochs: The number of iterations over the training set during model training.
    :param optimizer: The torch optimizer used for weight updates during trianing.
    :param rounds_between_evals: An integer denoting the number of epcohs that occur between each evaluation run.
    :return: A trained model.
    """
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for sampled_data in tqdm(training_loader):
            optimizer.zero_grad()
            out = model(sampled_data)
            preds = torch.argmax(out, dim=1)
            y_one_hot = F.one_hot(sampled_data.y, 2)
            loss = model.loss(out.float(), y_one_hot.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_correct_preds = (preds == sampled_data.y).sum().float()
            accuracy = num_correct_preds / sampled_data.y.shape[0] * 100
            epoch_acc += accuracy
            num_batches += 1
        wandb.log({f'training_{constants.ACCURACY}': epoch_acc / num_batches,
                   f'training_{constants.EPOCH}': epoch,
                   f'training_{constants.LOSS}': epoch_loss / num_batches})

        # Perform evaluation on the validation set.
        if epoch % rounds_between_evals == 0:
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            model.eval()
            for sampled_data in tqdm(validation_loader):
                outputs = model(sampled_data)
                preds = torch.argmax(outputs, dim=1)
                y_one_hot = F.one_hot(sampled_data.y, 2)
                loss = model.loss(outputs.float(), y_one_hot.float())
                num_correct_preds = (preds == sampled_data.y).sum().float()
                accuracy = num_correct_preds / sampled_data.y.shape[0] * 100
                num_batches += 1
                epoch_loss += loss.item()
                epoch_acc += accuracy
            wandb.log({f'validation_{constants.ACCURACY}': epoch_acc / num_batches,
                       f'validation_{constants.EPOCH}': epoch,
                       f'validation_{constants.LOSS}': epoch_loss / num_batches})
    return model


def eval(model: GCNWithBertEmbeddings,
         dataset: CMVKGDataset):
    """
    Evaluate the performance of a GCNWithBertEmbeddings model.

    The test set used for this evaluation consists of distinct examples from those used by the model during training.

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param dataset: A CMVKGDataLoader instance
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
                       val_percent: float,
                       test_percent: float,
                       num_workers: int = 0):
    """Create dataloaders over persuasive argument knowledge graphs.

    :param graph_dataset: A 'CMVKGDataset' instance whose examples correspond to knowledge graphs of persuasive
        arguments.
    :param batch_size: The number of examples processed in a single batch as part of training.
    :param test_percent: The ratio of the original examples dedicated towards a test set. The remaining examples are
        used as part of model training.
    :param num_workers: The number of workers used during training.
    :return:
    """
    num_of_examples = len(graph_dataset.dataset)
    test_len = int(test_percent * num_of_examples)
    val_len = int(val_percent * num_of_examples)
    train_len = num_of_examples - test_len -val_len

    indexes = random.sample(range(num_of_examples), num_of_examples)
    test_indexes = indexes[:test_len]
    val_indexes = indexes[test_len:test_len + val_len]
    train_indexes = indexes[test_len + val_len:-1]

    dl_train = geom_data.DataLoader(graph_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=SubsetRandomSampler(train_indexes))
    dl_val = geom_data.DataLoader(graph_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=SubsetRandomSampler(val_indexes))
    dl_test = geom_data.DataLoader(graph_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   sampler=SubsetRandomSampler(test_indexes))
    return dl_train, dl_val, dl_test


if __name__ == '__main__':

    args = parser.parse_args()
    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')

    num_node_features = constants.BERT_HIDDEN_DIM
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with wandb.init(project="persuasive_arguments", config=args, name="GCN with BERT Embeddings"):
        kg_dataset = CMVKGDataset(
            current_path + "/cmv_modes/change-my-view-modes-master",
            version=constants.v2_path,
            debug=False)
        # utils.get_dataset_stats(kg_dataset)
        config = wandb.config
        dl_train,dl_val, dl_test = create_dataloaders(kg_dataset,
                                               batch_size=config.batch_size,
                                               val_percent=config.val_percent,
                                               test_percent=config.test_percent)
        model = GCNWithBertEmbeddings(
            num_node_features,
            num_classes=2,
            hidden_layer_dim=config.gcn_hidden_layer_dim)

        # TODO: Make log_freq a flag.
        wandb.watch(model, log='all', log_freq=5)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
        model = train(model=model,
                      training_loader=dl_train,
                      validation_loader=dl_val,
                      epochs=config.num_epochs,
                      optimizer=optimizer,
                      rounds_between_evals=config.rounds_between_evals)
        eval(model, dl_test)
