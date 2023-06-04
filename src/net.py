import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv1 = nn.Conv2d(3, 32, 2, 1)
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_conv2 = nn.Conv2d(32, 16, 2, 1)
        self.encoder_bn2 = nn.BatchNorm2d(16)
        self.encoder_conv3 = nn.Conv2d(16, 3, 2, 2)
        self.encoder_bn3 = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = F.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x = F.relu(self.encoder_bn3(self.encoder_conv3(x)))
        x = self.sigmoid(x)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_deconv1 = nn.ConvTranspose2d(3, 16, 2, 2)
        self.decoder_bn1 = nn.BatchNorm2d(16)
        self.decoder_deconv2 = nn.ConvTranspose2d(16, 32, 2, 1)
        self.decoder_bn2 = nn.BatchNorm2d(32)
        self.decoder_deconv3 = nn.ConvTranspose2d(32, 3, 2, 1)
        self.decoder_bn3 = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.decoder_bn1(self.decoder_deconv1(x)))
        x = F.relu(self.decoder_bn2(self.decoder_deconv2(x)))
        x = F.relu(self.decoder_bn3(self.decoder_deconv3(x)))
        x = self.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, quantize_level: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantize_level = quantize_level

    def forward(self, x):
        # encode
        x = self.encoder(x)

        # add noise
        # mean = std = torch.tensor(0.5)
        # noise = torch.normal(mean, std) / (2**self.quantize_level)
        # x += noise
        # x = torch.log((x / (1 - x + EPS) + EPS))

        # decode
        x = self.decoder(x)

        return x
