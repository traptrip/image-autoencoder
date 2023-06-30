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
        embedding = (embedding * (2**self.q_level) + 0.5).to(torch.int32)
        embedding = embedding.tolist()
        embedding = self.compressor.compress(embedding)
        return embedding, shape

    def decompress(self, emb: list[int], shape: list[int]) -> torch.Tensor:
        length = np.prod(shape)
        emb = self.compressor.decompress(emb, length)
        embedding = torch.tensor(emb).float()
        embedding = embedding / (2**self.q_level)
        embedding = embedding.view(*shape)
        return embedding
