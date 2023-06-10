import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv1 = nn.Conv2d(3, 32, 2, 1)
        self.encoder_bn1 = nn.BatchNorm2d(32)
        self.encoder_conv2 = nn.Conv2d(32, 16, 2, 1)
        self.encoder_bn2 = nn.BatchNorm2d(16)
        self.encoder_conv3 = nn.Conv2d(16, 3, 2, 2)
        self.encoder_bn3 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = F.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x = F.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x = F.relu(self.encoder_bn3(self.encoder_conv3(x)))
        return x
