import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import torch_geometric.data as geom_data

from cmv_modes.preprocessing_knowledge_graph import CMVKGDataLoader, v2_path
from model import GCN
import constants

def train(model, dl, epochs, optimizer):
    model.train()

    for epoch in range(epochs):
        for sampled_data in dl:
            optimizer.zero_grad()
            out = model(sampled_data)
            preds = torch.argmax(out, dim=1)
            y_one_hot = F.one_hot(sampled_data.y, 2)
            loss =model.loss(out.float(),y_one_hot.float() )
            loss.backward()
            optimizer.step()
            print('finished EPOCH')
    return model

def eval(model, dataset):
    model.eval()
    pred = model(dataset).argmax(dim=1)
    correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
    acc = int(correct) / int(dataset.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

def creat_dataloaders(db: Dataset, batch_size: int, test_percent: float, num_workers: int = 0):
    print('Randomly splitting patients to train and test:')
    num_of_examples = len(db.dataset)
    train_len = int((1.0 - test_percent) * num_of_examples)

    indexes = random.sample(range(num_of_examples), num_of_examples)
    train_indexes = indexes[:train_len]
    test_indexes = indexes[train_len:]
    dl_train = geom_data.DataLoader(db, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SubsetRandomSampler(train_indexes))
    dl_test = geom_data.DataLoader(db, batch_size=batch_size, num_workers=num_workers,
                                           sampler=SubsetRandomSampler(test_indexes))
    return dl_train, dl_test

if __name__ == '__main__':
    epochs = 2
    batch_size = 8
    test_percent = 0.2
    hidden_layer_dim = 128
    num_node_features = constants.NODE_DIM
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kg_dataset = CMVKGDataLoader(current_path+ "/cmv_modes/change-my-view-modes-master", version=v2_path, debug=True)

    # kg_dataset = kg_dataset.to(device)
    dl_train, dl_test = creat_dataloaders(kg_dataset,batch_size=batch_size,test_percent=test_percent)
    # batch = next(iter(dl_train))
    model = GCN(num_node_features, num_classes=2, hidden_layer_dim=hidden_layer_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model = train(model, dl_train, epochs, optimizer)