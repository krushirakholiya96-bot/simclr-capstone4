import json
import torch
import torch.nn.functional as F
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from torchvision import transforms
from src.model import SimCLRModel
from api.database.crud import Database

router = APIRouter()

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0)

    # Get model from app state
    from api.main import encoder, classifier
    device = next(encoder.parameters()).device

    tensor = tensor.to(device)

    # Extract features and predict
    with torch.no_grad():
        features = encoder.encoder(tensor).flatten(1)
        logits = classifier.classifier(features)
        probs = F.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = probs.topk(5)
    top5 = [
        {'class': CLASSES[idx.item()],
         'confidence': round(prob.item() * 100, 2)}
        for prob, idx in zip(top5_probs, top5_idx)
    ]

    return {
        'predicted_class': top5[0]['class'],
        'confidence': top5[0]['confidence'],
        'top5': top5,
        'image_name': file.filename
    }