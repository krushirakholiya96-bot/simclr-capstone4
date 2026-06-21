import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model import SimCLRModel
from src.loss import NTXentLoss
from src.dataset import get_dataloader


def train_simclr(epochs=200, batch_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model = SimCLRModel().to(device)
    loss_fn = NTXentLoss(tau=0.07)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    warmup_epochs = 10
    loader = get_dataloader(batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        # LR Warmup
        if epoch <= warmup_epochs:
            lr = 1e-3 * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for batch_idx, (view1, view2, _) in enumerate(loader):
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            z1 = model(view1)
            z2 = model(view2)

            # Compute loss
            loss = loss_fn(z1, z2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] "
                      f"Batch [{batch_idx}/{len(loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

        # Update scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save(model.state_dict(),
                      f'checkpoints/simclr_encoder_epoch{epoch}.pth')
            print(f"Checkpoint saved at epoch {epoch}")

    # Save final model
    torch.save(model.state_dict(), 'checkpoints/simclr_encoder_final.pth')
    print("Training complete! Final model saved.")

    return model


if __name__ == '__main__':
    model = train_simclr(epochs=200, batch_size=512)