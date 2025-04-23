import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import torch.nn as nn

BATCH_SIZE = 4
EPOCHS = 100_000
LR = 1e-4

loss_rec = nn.MSELoss()

def get_dummy_dataloader():
    dummy_vids = torch.randn(20, 3, 16, 224, 224)
    dummy_gray = torch.mean(dummy_vids, dim=1, keepdim=True)
    dataset = TensorDataset(dummy_vids, dummy_gray)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train(encoder, decoder, affine_estimator, device):
    dataloader = get_dummy_dataloader()

    optim_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    optim_affine = optim.Adam(affine_estimator.parameters(), lr=LR)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        affine_estimator.train()

        total_loss = 0

        for i, (video, gray) in enumerate(dataloader):
            video = video.to(device)                  # [B, 3, 16, 224, 224]
            gray = gray.to(device)                    # [B, 1, 16, 224, 224]

            # --- Autoencoder ---
            encoded = encoder(video)
            recon = decoder(encoded)
            rec_loss = loss_rec(recon, video)

            optim_ae.zero_grad()
            rec_loss.backward()
            optim_ae.step()

            gray_seq = gray[:, :, 6:11]                  # 5 middle frames
            gray_seq = gray_seq.permute(0, 2, 1, 3, 4)   # [B, 5, 1, H, W]

            affine_preds = []
            for j in range(4):
                f1 = gray_seq[:, j]
                f2 = gray_seq[:, j + 1]
                theta = affine_estimator(f1, f2)         # [B, 2, 3]
                affine_preds.append(theta)

            affines = torch.stack(affine_preds, dim=1)   # [B, 4, 2, 3]


            total_loss += rec_loss.item()

            if i % 5 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] Step {i} | Rec Loss: {rec_loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"==== Epoch {epoch+1} Finished | Avg Rec Loss: {avg_loss:.4f} ====")

        # Save checkpoint
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'affine': affine_estimator.state_dict()
        }, f"checkpoints/nvcf_epoch{epoch+1}.pth")

