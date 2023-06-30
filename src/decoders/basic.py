import torch.nn as nn
import torch.nn.functional as F


class BasicDecoder(nn.Module):
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


# class BasicDecoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(16)
#         self.conv5 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
#         self.bn5 = nn.BatchNorm2d(3)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = F.relu(self.bn5(self.conv5(x)))
#         return x


class ViTDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x
