import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_layer_dim):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_layer_dim = hidden_layer_dim
        self.conv1 = GCNConv(self.num_node_features, self.hidden_layer_dim)
        self.conv2 = GCNConv(self.hidden_layer_dim, self.num_classes)
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch, )
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    model = GCN(256,2,16)
    print(model)

