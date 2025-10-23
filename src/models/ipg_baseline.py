import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class IPGBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(IPGBaseline, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = self.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return x
