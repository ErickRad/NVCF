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

            nn.Unflatten(1, (128, 1, 14, 14)),  # Reconstruindo para a forma 128x1x14x14

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