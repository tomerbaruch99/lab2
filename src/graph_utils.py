import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import numpy as np

def patch_to_graph(lr_patches, edge_k=4):
    """
    Convert patches to graph: nodes = flattened patch vectors
    edges = KNN based on L2 similarity
    """
    N, h, w = lr_patches.shape
    X = lr_patches.reshape(N, -1)
    nbrs = NearestNeighbors(n_neighbors=edge_k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    edge_index = []
    for i in range(N):
        for j in indices[i][1:]:  # skip self
            edge_index.append([i,j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data
