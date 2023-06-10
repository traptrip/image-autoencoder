import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_deconv1 = nn.ConvTranspose2d(3, 16, 2, 2)
        self.decoder_bn1 = nn.BatchNorm2d(16)
        self.decoder_deconv2 = nn.ConvTranspose2d(16, 32, 2, 1)
        self.decoder_bn2 = nn.BatchNorm2d(32)
        self.decoder_deconv3 = nn.ConvTranspose2d(32, 3, 2, 1)
        self.decoder_bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.decoder_bn1(self.decoder_deconv1(x)))
        x = F.relu(self.decoder_bn2(self.decoder_deconv2(x)))
        x = F.relu(self.decoder_bn3(self.decoder_deconv3(x)))
        return x
