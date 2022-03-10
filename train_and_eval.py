import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
import torch_geometric.loader as geom_data
from torch_geometric.loader import DataLoader
import wandb
from cmv_modes.preprocessing_knowledge_graph import CMVKGDataLoader
from models import GCNWithBertEmbeddings
import constants
from tqdm import tqdm



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
            loss =model.loss(out.float(),y_one_hot.float() )
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
         dl: CMVKGDataLoader):
    """

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param dataset: A CMVKGDataLoader instance
    :return:
    :rtype:
    """
    model.eval()
    acc = 0.0
    num_batches = 0
    with torch.no_grad():
        for sampled_data in tqdm(dl):
            out = model(sampled_data)
            preds = torch.argmax(out, dim=1)
            num_correct_preds = (preds == sampled_data.y).sum().float()
            accuracy = num_correct_preds / sampled_data.y.shape[0] * 100
            acc += accuracy
            num_batches += 1
        acc = acc /num_batches
    print(f'Accuracy: {acc:.4f}')

def creat_dataloaders(db: Dataset, batch_size: int, test_percent: float, num_workers: int = 0):
    print('Randomly splitting patients to train and test:')
    num_of_examples = len(db.dataset)
    train_len = int((1.0 - test_percent) * num_of_examples)

    indexes = random.sample(range(num_of_examples), num_of_examples)
    train_indexes = indexes[:train_len]
    test_indexes = indexes[train_len:]
    dl_train = geom_data.DataLoader(db,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=SubsetRandomSampler(train_indexes))
    dl_test = geom_data.DataLoader(db,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   sampler=SubsetRandomSampler(test_indexes))
    return dl_train, dl_test

if __name__ == '__main__':
    epochs = 2
    batch_size = 8
    learning_rate = 0.01
    weight_decay = 5e-4
    test_percent = 0.2
    hidden_layer_dim = 128
    num_node_features = constants.BERT_HIDDEN_DIM
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kg_dataset = CMVKGDataLoader(current_path+ "/cmv_modes/change-my-view-modes-master", version=constants.v2_path, debug=True)

    config = dict(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate = learning_rate,
        weight_decay = weight_decay,
        hidden_layer_dim = hidden_layer_dim
    )
    with wandb.init(project="persuasive_arguments", config=config):
        kg_dataset = CMVKGDataLoader(current_path + "/cmv_modes/change-my-view-modes-master", version=constants.v2_path,
                                     debug=False)

        config = wandb.config
        dl_train, dl_test = creat_dataloaders(kg_dataset, batch_size=config.batch_size, test_percent=test_percent)
        model = GCNWithBertEmbeddings(num_node_features, num_classes=2, hidden_layer_dim=config.hidden_layer_dim)
        wandb.watch(model, log='all', log_freq=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        model = train(model, dl_train, epochs, optimizer)
        eval(model, dl_test)