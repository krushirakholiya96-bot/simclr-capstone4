import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        # Temperature parameter - controls sharpness of distribution
        self.tau = tau

    def forward(self, z1, z2):
        N = z1.shape[0]
        
        # Concatenate both views
        z = torch.cat([z1, z2], dim=0)  # (2N, 128)
        
        # Cosine similarity matrix
        sim = torch.mm(z, z.T) / self.tau  # (2N, 2N)
        
        # Remove self-similarity from diagonal
        mask = torch.eye(2 * N, dtype=bool).to(z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(N, 2 * N),
            torch.arange(N)
        ]).to(z.device)
        
        # Cross entropy loss
        return F.cross_entropy(sim, labels)