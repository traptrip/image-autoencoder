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


# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = nn.Conv2d(3, 32, 2, 2)  # 32, 256, 256
#         self.bn1 = nn.BatchNorm2d(32)
#         self.c2 = nn.Conv2d(32, 128, 2, 2)  # 128, 128, 128
#         self.bn2 = nn.BatchNorm2d(128)
#         self.p1 = nn.MaxPool2d(2, 2)  # 128, 64, 64
#         self.c3 = nn.Conv2d(128, 256, 2, 2)  # 256, 32, 32
#         self.bn3 = nn.BatchNorm2d(256)
#         self.p2 = nn.MaxPool2d(2, 2)  # 256, 16, 16
#         self.c4 = nn.Conv2d(256, 512, 2, 2)  # 512, 8, 8
#         self.bn4 = nn.BatchNorm2d(512)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.c1(x)))
#         x = F.relu(self.bn2(self.c2(x)))
#         x = F.relu(self.p1(x))
#         x = F.relu(self.bn3(self.c3(x)))
#         x = F.relu(self.p2(x))
#         x = F.relu(self.bn4(self.c4(x)))
#         return x
