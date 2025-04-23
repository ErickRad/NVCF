import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),  # [batch, 3, 16, 224, 224] -> [batch, 16, 8, 112, 112]
            nn.ReLU(),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 16, 8, 112, 112] -> [batch, 32, 4, 56, 56]
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 32, 4, 56, 56] -> [batch, 64, 2, 28, 28]
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1), # [batch, 64, 2, 28, 28] -> [batch, 128, 1, 14, 14]
            nn.ReLU(),

            nn.Flatten(),  # Flat to a vector tensor

            nn.Linear(128 * 1 * 14 * 14, 512), 
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(512, 128 * 1 * 14 * 14),
            nn.ReLU(),

            nn.Unflatten(1, (128, 1, 14, 14)),  # Rebuild to 128x1x14x14

            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1),  # [batch, 128, 1, 14, 14] -> [batch, 64, 2, 28, 28]
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1),  # [batch, 64, 2, 28, 28] -> [batch, 32, 4, 56, 56]
            nn.ReLU(),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1),  # [batch, 32, 4, 56, 56] -> [batch, 16, 8, 112, 112]
            nn.ReLU(),

            nn.ConvTranspose3d(16, 3, kernel_size=3, stride=2, padding=1),  # [batch, 16, 8, 112, 112] -> [batch, 3, 16, 224, 224]
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        return self.decoder
    
class AffineMotionEstimator(nn.Module):
    def __init__(self):
        super(AffineMotionEstimator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.regressor = nn.Linear(64, 6)

    def estimate_all_affines(model, frame_seq):  # frame_seq: [B, 5, 1, H, W]
        affines = []

        for i in range(4):
            a = frame_seq[:, i]
            b = frame_seq[:, i + 1]
            theta = model(a, b)
            
            affines.append(theta)
            
        return torch.stack(affines, dim=1)  # Shape: [B, 4, 2, 3]

    def forward(self, frame_a, frame_b):
        x = torch.cat([frame_a, frame_b], dim=1)     # Shape: [B, 2, H, W]

        features = self.encoder(x)                   # Shape: [B, 64, 1, 1]
        flat = features.view(features.size(0), -1)   # Shape: [B, 64]

        theta = self.regressor(flat)                 # Shape: [B, 6]
        affine_matrices = theta.view(-1, 2, 3)       # Shape: [B, 2, 3]

        return affine_matrices