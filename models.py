# models.py
import torch 
from torch import nn
import torch.nn.functional as F
from utils import remove_patch

class NN_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),         
            nn.Linear(3072, 1500),
            nn.ReLU(True),
            nn.Linear(1500, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 800),
        )
        self.decoder = nn.Sequential(
            nn.Linear(800, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1500),
            nn.ReLU(True),
            nn.Linear(1500, 3072),
            nn.Tanh()              
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def validate_compression(self, test_loader, device):
        self.eval()
        with torch.no_grad():
            total_test_loss = 0
            criterion = nn.MSELoss()
            for images, _ in test_loader:
                images = images.to(device)
                recon = self(images)
                loss = criterion(recon, images.view(recon.shape))
                total_test_loss += loss.item() * images.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        self.train()
        return avg_test_loss
    
    def validate_inpainting(self, test_loader, device):
        self.eval()
        with torch.no_grad():
            total_test_loss = 0
            criterion = nn.MSELoss()
            for images, _ in test_loader:
                images = images.to(device)
                patch_images = remove_patch(images, device)
                recon = self(patch_images)
                if recon.shape != images.shape:
                    recon = recon.view_as(images)
                loss = criterion(recon, images)
                total_test_loss += loss.item() * patch_images.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        self.train()
        return avg_test_loss

class CNN_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5),
            nn.ReLU()
        )
    
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, 5),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 5),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat
    
    def validate_compression(self, test_loader, device):
        self.eval()
        with torch.no_grad():
            total_test_loss = 0
            criterion = nn.MSELoss()
            for images, _ in test_loader:
                images = images.to(device)
                recon = self(images)
                if recon.shape != images.shape:
                    recon = recon.view_as(images)
                loss = criterion(recon, images)
                total_test_loss += loss.item() * images.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        self.train()
        return avg_test_loss
    
    def validate_inpainting(self, test_loader, device):
        self.eval()
        with torch.no_grad():
            total_test_loss = 0
            criterion = nn.MSELoss()
            for images, _ in test_loader:
                images = images.to(device)
                patch_images = remove_patch(images, device)
                recon = self(patch_images)
                if recon.shape != images.shape:
                    recon = recon.view_as(images)
                loss = criterion(recon, images)
                total_test_loss += loss.item() * patch_images.size(0)
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        self.train()
        return avg_test_loss
