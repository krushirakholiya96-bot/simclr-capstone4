import torch
import pytest
from src.model import SimCLRModel


def test_output_shape():
    # Test that output shape is correct
    model = SimCLRModel()
    test_input = torch.randn(4, 3, 32, 32)
    output = model(test_input)
    assert output.shape == (4, 128), \
        f"Expected (4, 128) but got {output.shape}"


def test_l2_norm():
    # Test that output is L2 normalized
    model = SimCLRModel()
    test_input = torch.randn(4, 3, 32, 32)
    output = model(test_input)
    norms = output.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        "Output is not L2 normalized"


def test_parameter_count():
    # Test that model has approximately 25M parameters
    model = SimCLRModel()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 20_000_000, \
        f"Expected >20M params but got {total_params}"