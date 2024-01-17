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

train_set = NERDataset(
        token_seq=train_token_seq,
        label_seq=train_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    train_token_seq, train_label_seq = prepare_conll_data_format(
        path=config["data"]["train_data"]["path"],
        sep=config["data"]["train_data"]["sep"],
        lower=config["data"]["train_data"]["lower"],
        verbose=config["data"]["train_data"]["verbose"],
    )