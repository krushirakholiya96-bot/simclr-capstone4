import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from src.model import SimCLRModel


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def extract_features(encoder, loader, device, num_samples=2000):
    encoder.eval()
    features = []
    labels = []
    count = 0

    with torch.no_grad():
        for images, targets in loader:
            if count >= num_samples:
                break
            images = images.to(device)
            feat = encoder.encoder(images).flatten(1)
            features.append(feat.cpu().numpy())
            labels.extend(targets.numpy())
            count += len(images)

    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.array(labels)[:num_samples]
    return features, labels


def plot_tsne(features, labels, title='SimCLR Features — t-SNE'):
    tsne = TSNE(n_components=2, perplexity=30,
                n_iter=1000, random_state=42)
    features_2d = tsne.fit_transform(features)

    class_names = [CLASSES[l] for l in labels]

    fig = px.scatter(
        x=features_2d[:, 0],
        y=features_2d[:, 1],