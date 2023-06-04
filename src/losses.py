import torch
import torch.nn as nn
from torchvision import models


class VGGLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg = models.vgg19("DEFAULT")

    def forward(self, decoded_tensor: torch.Tensor, real_tensor: torch.Tensor):
        x1 = self.vgg(decoded_tensor)
        x2 = self.vgg(real_tensor)
        return torch.mean(torch.abs(x1 - x2))
