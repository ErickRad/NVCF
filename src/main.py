import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from NN import Encoder, Decoder, AffineMotionEstimator

# Initialize models
encoder = Encoder()
decoder = Decoder()
affine_estimator = AffineMotionEstimator()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)
affine_estimator.to(device)

# Dummy dataset: 10 video clips [B, C, T, H, W]
videos = torch.randn(10, 3, 16, 224, 224)
grayscale = torch.mean(videos, dim=1, keepdim=True)  # [B, 1, T, H, W] for affine

# Dataloader
dataset = TensorDataset(videos, grayscale)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loss and optimizers
reconstruction_loss = nn.MSELoss()
optimizer_autoencoder = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
optimizer_affine = optim.Adam(affine_estimator.parameters(), lr=1e-4)

# Training loop placeholder
for epoch in range(2):
    for video_batch, gray_batch in loader:
        video_batch = video_batch.to(device)
        gray_batch = gray_batch.to(device)

        # === Autoencoder Forward Pass ===
        encoded = encoder(video_batch)           # [B, 512]
        reconstructed = decoder(encoded)         # [B, 3, 16, 224, 224]

        loss_rec = reconstruction_loss(reconstructed, video_batch)

        optimizer_autoencoder.zero_grad()
        loss_rec.backward()
        optimizer_autoencoder.step()

        frames = gray_batch[:, :, 6:11]          # Pick 5 middle grayscale frames: [B, 1, 5, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)   # [B, 5, 1, H, W]

        affine_thetas = []
        for i in range(4):
            frame_a = frames[:, i].to(device)
            frame_b = frames[:, i + 1].to(device)
            theta = affine_estimator(frame_a, frame_b)  # [B, 2, 3]
            affine_thetas.append(theta)

        thetas = torch.stack(affine_thetas, dim=1)  # [B, 4, 2, 3]

        print(f"[Epoch {epoch}] Rec Loss: {loss_rec.item():.4f}, Affine shape: {thetas.shape}")
