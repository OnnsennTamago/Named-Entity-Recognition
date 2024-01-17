import numpy as np
import torch
import torch.nn as nn
from gensim.models import FastText, KeyedVectors

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
