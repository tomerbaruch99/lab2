import torch
import torch.nn as nn

class PatchDecoder(nn.Module):
    def __init__(self, patch_size):
        super(PatchDecoder, self).__init__()
        self.patch_size = patch_size

    def forward(self, x, H, W):
        # x: [N_nodes, patch_size*patch_size]
        patch_dim = self.patch_size
        N_nodes = x.shape[0]
        patches = x.view(N_nodes, patch_dim, patch_dim).detach().cpu().numpy()
        # simple reassemble assuming grid
        # TODO: smarter reconstruction
        return patches
