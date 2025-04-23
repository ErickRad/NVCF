import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from NN import MotionAwareGenerator, Discriminator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = MotionAwareGenerator().to(device)
discriminator = Discriminator().to(device)

# Dummy dataset: 10 video clips [B, C, T, H, W]
videos = torch.randn(10, 3, 16, 224, 224)
grayscale = torch.mean(videos, dim=1, keepdim=True)  # [B, 1, T, H, W]

# Dataloader
dataset = TensorDataset(videos, grayscale)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loss and optimizers
reconstruction_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)

# Training loop placeholder
for epoch in range(2):
    for video_batch, gray_batch in loader:
        video_batch = video_batch.to(device)
        gray_batch = gray_batch.to(device)

        valid = torch.ones(video_batch.size(0), 1).to(device)
        fake = torch.zeros(video_batch.size(0), 1).to(device)

        # === Train Generator ===
        optimizer_G.zero_grad()
        gen_videos = generator(video_batch)
        pred_fake = discriminator(gen_videos)
        loss_G = reconstruction_loss(gen_videos, video_batch) + bce_loss(pred_fake, valid)
        loss_G.backward()
        optimizer_G.step()

        # === Train Discriminator ===
        optimizer_D.zero_grad()
        pred_real = discriminator(video_batch)
        pred_fake = discriminator(gen_videos.detach())
        loss_D_real = bce_loss(pred_real, valid)
        loss_D_fake = bce_loss(pred_fake, fake)
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}] Loss_G: {loss_G.item():.4f} | Loss_D: {loss_D.item():.4f}")
