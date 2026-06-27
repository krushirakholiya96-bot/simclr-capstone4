import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.model import SimCLRModel
from api.database.db import create_tables
from api.routes import predict, explain, agent, history


# Global model variables
encoder = None
classifier = None


class SimCLRClassifier(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.encoder = enc
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        feat = self.encoder.encoder(x).flatten(1)
        return self.classifier(feat)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global encoder, classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading SimCLR encoder...")
    encoder = SimCLRModel()
    encoder.load_state_dict(
        torch.load('checkpoints/simclr_encoder_final.pth',
                   map_location=device)
    )
    encoder = encoder.to(device)
    encoder.eval()

    print("Loading classifier...")
    classifier = SimCLRClassifier(encoder)
    classifier.load_state_dict(
        torch.load('checkpoints/classifier.pth',
                   map_location=device)
    )
    classifier = classifier.to(device)
    classifier.eval()

    # Create database tables
    create_tables()

    print("Models loaded! Server ready.")
    yield

    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title='SimCLR Contrastive Learner API',
    description='End-to-end SimCLR image classification with Generative AI',
    version='1.0.0',
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# Include routers
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(agent.router)
app.include_router(history.router)


@app.get("/health")
async def health():
    return {
        'status': 'healthy',
        'model': 'SimCLR ResNet50',
        'version': '1.0.0'
    }