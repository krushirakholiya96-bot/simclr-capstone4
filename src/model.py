import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ResNet50 backbone - remove final FC layer
        resnet = torchvision.models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # 3-layer projection head (SimCLR v2)
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        # Extract features from encoder
        feat = self.encoder(x).flatten(1)  # (B, 2048)
        # Project to embedding space
        proj = self.projector(feat)         # (B, 128)
        # L2 normalize
        return F.normalize(proj, dim=1)