import torch
import numpy as np
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import SimpleAdaptiveModel


class Compressor:
    def __init__(self, q_level: int) -> None:
        self.q_level = q_level
        self.compressor = self._init_compressor()

    def _init_compressor(self) -> AECompressor:
        n = 2**self.q_level + 1
        probability = dict(zip(range(n), [1 / n] * n))
        model = SimpleAdaptiveModel(probability)
        compressor = AECompressor(model)
        return compressor

    def compress(self, embedding: torch.Tensor) -> tuple[list[int], list[int]]:
        shape = list(embedding.shape)
        embedding = embedding.flatten()
        embedding = (embedding * (2**self.q_level) + 0.5).to(torch.int64)
        embedding = embedding.tolist()
        encoded = self.compressor.compress(embedding)
        return encoded, shape

    def decompress(self, emb: list[int], shape: list[int]) -> torch.Tensor:
        length = np.prod(shape)
        decoded = self.compressor.decompress(emb, length)
        # decoded = np.fromiter(map(int, decoded), dtype=np.int64)
        decoded = torch.tensor(decoded).float()
        decoded = decoded / (2**self.q_level)
        decoded = decoded.view(*shape)
        return decoded
