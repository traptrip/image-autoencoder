import torch.nn as nn
import timm
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, grad_checkpointing=False):
        super().__init__()
        backbone = timm.create_model(
            "maxvit_tiny_tf_512", num_classes=512, pretrained=True
        )
        backbone.grad_checkpointing = grad_checkpointing

        # 3x512x512 -> 512x16x16
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # # 512x16x16 -> 768x2x2
        # self.conv1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(64, 768, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.backbone(x)
        return x
