import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from PIL import Image
import io


def test_health_endpoint():
    # Test health endpoint returns 200
    with patch('api.main.encoder', MagicMock()), \
         patch('api.main.classifier', MagicMock()):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'


def test_predict_endpoint():
    # Test predict endpoint with dummy image
    with patch('api.main.encoder') as mock_encoder, \
         patch('api.main.classifier') as mock_classifier:

        # Mock encoder output
        mock_encoder.encoder.return_value = \
            MagicMock(flatten=MagicMock(
                return_value=torch.randn(1, 2048)
            ))
        mock_encoder.parameters.return_value = \
            iter([torch.randn(1)])

        # Mock classifier output
        mock_classifier.classifier.return_value = \
            torch.randn(1, 10)

        from api.main import app
        client = TestClient(app)

        # Create dummy image
        image = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = client.post(
            "/predict",
            files={'file': ('test.png', img_bytes, 'image/png')}
        )
        assert response.status_code == 200