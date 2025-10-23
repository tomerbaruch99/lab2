import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GNNAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, num_layers=2):
        super(GNNAttention, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers-2):
            self.convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels*heads, out_channels, heads=1))
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = self.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x
