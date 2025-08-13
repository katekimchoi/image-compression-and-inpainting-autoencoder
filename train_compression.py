# train_compression.py
from models import NN_Autoencoder, CNN_Autoencoder
from utils import show_compression_reconstructions, imshow, save_checkpoint, load_checkpoint
import argparse
import random
import numpy as np
import os

import torch
from torch import nn, optim
import torchvision
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Train autoencoder for CIFAR-10 dog image compression")
    p.add_argument("--model-name", "-m",
                   type=lambda s: s.upper(), choices=["CNN", "NN"], default="CNN",
                   help="Model architecture to use: 'CNN' or 'NN'")
    p.add_argument("--epochs", "-e", type=int, default=101, help="Number of training epochs")
    p.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    p.add_argument("--dog-class-index", type=int, default=5, help="CIFAR-10 class index for dogs")
    p.add_argument("--device", type=str, default=None,
                   help="Device: 'cuda', 'cpu', or 'mps'. Auto-detect if omitted")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save model checkpoints")
    p.add_argument("--epochs-per-summary", type=int, default=10, help="Number of epochs during training between summaries")
    p.add_argument("--restore-path", type=str, default=None, help="Path to resume training from a model checkpoint")
    return p.parse_args()

def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda but CUDA is not available.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device=mps but MPS is not available.")
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Settings
    model_name = args.model_name
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    dog_class_index = args.dog_class_index
    checkpoint_dir = args.checkpoint_dir
    epochs_per_summary = args.epochs_per_summary
    restore_path = args.restore_path

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset (using CIFAR-10 dog images only)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dog_indices = [i for i, (_, label) in enumerate(train_dataset) if label == dog_class_index]
    train_dog_dataset = Subset(train_dataset, train_dog_indices)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_dog_indices = [i for i, (_, label) in enumerate(test_dataset) if label == dog_class_index]
    test_dog_dataset = Subset(test_dataset, test_dog_indices)

    train_loader = DataLoader(train_dog_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dog_dataset,  batch_size=batch_size*2, shuffle=False)

    # Instantiate model
    if model_name == "NN":
        model = NN_Autoencoder().to(device)
    elif model_name == "CNN":
        model = CNN_Autoencoder().to(device)
    else:
        raise ValueError("model_name must be 'CNN' or 'NN'")

    # Optimizer, Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # (Optional) Resume training from saved checkpoint
    if restore_path is not None:
        load_checkpoint(model, optimizer, restore_path, device)
        print(f"Training resumed from saved checkpoint: {restore_path}")

    # Training
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            recon = model(images)

            if recon.shape != images.shape:
                recon = recon.view_as(images)
            loss = criterion(recon, images.view(recon.shape))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        if epoch % epochs_per_summary == 0:
            show_compression_reconstructions(model, test_loader, device, n_show=8, fname=f"epoch_{epoch:03d}.png")
            print(f"Epoch {epoch:>4d} | MSE {epoch_loss/len(train_loader.dataset):.4f} | "
                  f"Test MSE {model.validate_compression(test_loader, device):.4f}")

    # Save model
    if checkpoint_dir is not None:
        save_path = os.path.join(checkpoint_dir, f"model_{epoch:03d}.pth")
        save_checkpoint(model, optimizer, epoch, path=save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
