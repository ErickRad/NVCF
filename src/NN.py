import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            ResidualBlock(16, 32),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            ResidualBlock(64, 128),

            nn.AdaptiveAvgPool3d((1, 14, 14)),
            nn.Flatten(),

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

            nn.Unflatten(1, (128, 1, 14, 14)),

            ResidualBlock(128, 128),

            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            ResidualBlock(64, 64),

            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.ConvTranspose3d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)


class AffineMotionEstimator(nn.Module):
    def __init__(self):
        super(AffineMotionEstimator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.regressor = nn.Linear(64, 6)

    def forward(self, frame_a, frame_b):
        x = torch.cat([frame_a, frame_b], dim=1)

        features = self.encoder(x)
        flat = features.view(features.size(0), -1)
        theta = self.regressor(flat)

        return theta.view(-1, 2, 3)

    def estimate_all_affines(self, frame_seq):
        affines = []

        for i in range(4):
            a = frame_seq[:, i]
            b = frame_seq[:, i + 1]
            theta = self.forward(a, b)

            affines.append(theta)

        return torch.stack(affines, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.bn2 = nn.BatchNorm3d(out_channels)


    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity

        return self.relu(out)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, z):
        encoded = self.encoder(z)
        return self.decoder(encoded)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class MotionAwareGenerator(Generator):
    def __init__(self):
        super(MotionAwareGenerator, self).__init__()

        self.motion_estimator = AffineMotionEstimator()

    def apply_affine(self, frame, affine_matrix):
        grid = F.affine_grid(
            affine_matrix, 
            frame.size(), 
            align_corners=False
        )

        transformed = F.grid_sample(frame, grid, align_corners=False)

        return transformed

    def forward(self, z):
        sequence = super().forward(z)
        B, C, T, H, W = sequence.shape
        reshaped = sequence.permute(0, 2, 1, 3, 4).reshape(B, T, C, H, W)
        affines = self.motion_estimator.estimate_all_affines(reshaped[:, :, 0:1].squeeze(2))

        for i in range(4):
            frame = sequence[:, :, i + 1]
            matrix = affines[:, i]
            sequence[:, :, i + 1] = self.apply_affine(frame, matrix)
            
        return sequence
