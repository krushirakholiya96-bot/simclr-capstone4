import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import SimCLRModel


class SimCLRClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # Linear classifier — 2048 features to 10 classes
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        # Extract features from encoder
        with torch.no_grad():
            feat = self.encoder.encoder(x).flatten(1)  # (B, 2048)
        return self.classifier(feat)


def get_dataloaders(batch_size=512):
    # Standard transform without augmentation for classifier
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    train_dataset = datasets.CIFAR10(
        root='./data', train=True,
        transform=transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False,
        transform=transform, download=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load trained SimCLR encoder
    encoder = SimCLRModel()
    encoder.load_state_dict(
        torch.load('checkpoints/simclr_encoder_final.pth',
                   map_location=device)
    )
    encoder = encoder.to(device)

    train_loader, test_loader = get_dataloaders()

    model = SimCLRClassifier(encoder).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1 — Linear Probe (encoder frozen)
    print("\nPhase 1: Linear Probe — Encoder Frozen")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-2)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            acc = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch}/20] Loss: {total_loss/len(train_loader):.4f} "
                  f"Test Accuracy: {acc:.2f}%")

    phase1_acc = evaluate(model, test_loader, device)
    print(f"\nPhase 1 Complete! Accuracy: {phase1_acc:.2f}%")

    # Phase 2 — Fine-tuning (encoder unfrozen)
    print("\nPhase 2: Fine-tuning — Encoder Unfrozen")
    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            acc = evaluate(model, test_loader, device)
            print(f"Epoch [{epoch}/30] Loss: {total_loss/len(train_loader):.4f} "
                  f"Test Accuracy: {acc:.2f}%")

    phase2_acc = evaluate(model, test_loader, device)
    print(f"\nPhase 2 Complete! Final Accuracy: {phase2_acc:.2f}%")

    # Save classifier
    torch.save(model.state_dict(), 'checkpoints/classifier.pth')
    print("Classifier saved!")

    return model, phase2_acc


if __name__ == '__main__':
    train_classifier()