import torch
import torch.nn as nn
from src import utils


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        args = cfg["args"] if cfg["args"] else {}
        self.encoder = utils.load_obj(cfg["name"])(**args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        args = cfg["args"] if cfg["args"] else {}
        self.decoder = utils.load_obj(cfg["name"])(**args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.decoder(x)
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
        noise = (torch.rand_like(x) * 0.5 + 0.5) / (2**self.quantize_level)
        x = x + noise

        # decode
        x = self.decoder(x)

        return x
