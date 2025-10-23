import torch
from torch.utils.data import DataLoader
from src.datasets import DIV2KSubset
from src.graph_utils import patch_to_graph
from src.models.ipg_baseline import IPGBaseline
from src.models.gnn_attention import GNNAttention
import torch.nn as nn
import torch.optim as optim

# parameters
patch_size = 16
batch_size = 1
lr = 1e-3
epochs = 5
model_type = 'baseline'  # or 'attention'

dataset = DIV2KSubset(hr_dir='data/DIV2K_sub', lr_dir='data/DIV2K_sub_LR', patch_size=patch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

in_channels = patch_size*patch_size
hidden_channels = 64
out_channels = patch_size*patch_size

if model_type=='baseline':
    model = IPGBaseline(in_channels, hidden_channels, out_channels)
else:
    model = GNNAttention(in_channels, hidden_channels, out_channels)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

for epoch in range(epochs):
    for lr_patches, hr_patches in dataloader:
        data = patch_to_graph(lr_patches[0].numpy())
        optimizer.zero_grad()
        output = model(data)
        hr_tensor = torch.tensor(hr_patches[0].reshape(-1, patch_size*patch_size), dtype=torch.float)
        loss = criterion(output, hr_tensor)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
