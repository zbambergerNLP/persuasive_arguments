import torch
import torch.nn.functional as F
import os
from cmv_modes.preprocessing_knowledge_graph import CMVKGDataLoader, v2_path
from model import GCN

def train(model, dataset, epochs, optimizer):


    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(dataset)
        loss = F.binary_cross_entropy(out, dataset.labels)
        loss.backward()
        optimizer.step()
    return model

def eval(model, dataset):
    model.eval()
    pred = model(dataset).argmax(dim=1)
    correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
    acc = int(correct) / int(dataset.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kg_dataset = CMVKGDataLoader(current_path, version=v2_path)
    kg_dataset = kg_dataset.to(device)
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)