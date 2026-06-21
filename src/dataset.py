import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
           
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

    def __call__(self, x):
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


class SimCLRDataset(Dataset):
    def __init__(self, root='./data', train=True):
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True
        )
        self.transform = SimCLRTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        view1, view2 = self.transform(image)
        return view1, view2, label


def get_dataloader(batch_size=512):
    dataset = SimCLRDataset(root='./data', train=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return loader


# Test karo
if __name__ == '__main__':
    loader = get_dataloader(batch_size=512)
    view1, view2, labels = next(iter(loader))
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"Labels shape: {labels.shape}")
    print("Dataset ready!")