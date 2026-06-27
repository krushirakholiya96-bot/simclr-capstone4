import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import SimCLRModel


class SimCLRClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x):
        feat = self.encoder.encoder(x).flatten(1)
        return self.classifier(feat)


def get_test_loader(batch_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    test_dataset = datasets.CIFAR10(
        root='./data', train=False,
        transform=transform, download=True
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += predicted[i].eq(labels[i]).item()
                class_total[label] += 1

    overall_acc = 100.0 * correct / total
    print(f"\nOverall Accuracy: {overall_acc:.2f}%")
    print("\nPer-class Accuracy:")
    for i in range(10):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {classes[i]}: {acc:.2f}%")

    return overall_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = SimCLRModel()
    encoder.load_state_dict(
        torch.load('checkpoints/simclr_encoder_final.pth',
                   map_location=device)
    )

    model = SimCLRClassifier(encoder)
    model.load_state_dict(
        torch.load('checkpoints/classifier.pth',
                   map_location=device)
    )
    model = model.to(device)

    loader = get_test_loader()
    evaluate_model(model, loader, device)