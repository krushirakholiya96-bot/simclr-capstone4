import torch
import pytest
from src.loss import NTXentLoss


def test_loss_positive():
    # Test that loss is positive for random embeddings
    loss_fn = NTXentLoss(tau=0.07)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    loss = loss_fn(z1, z2)
    assert loss.item() > 0, "Loss should be positive"


def test_loss_identical_embeddings():
    # Test that identical embeddings give near zero loss
    loss_fn = NTXentLoss(tau=0.07)
    z = torch.randn(4, 128)
    z = torch.nn.functional.normalize(z, dim=1)
    loss = loss_fn(z, z)
    assert loss.item() < 1.0, \
        "Loss should be low for identical embeddings"


def test_loss_output_shape():
    # Test that loss is a scalar
    loss_fn = NTXentLoss(tau=0.07)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    loss = loss_fn(z1, z2)
    assert loss.shape == torch.Size([]), \
        "Loss should be a scalar"