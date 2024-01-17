import torch
import torch.nn as nn

class BiLSTM(nn.Module):

    def __init__(
        self,
        embedding_layer: nn.Module,
        rnn_layer: nn.Module,
        linear_head: nn.Module,
    ):
        super(BiLSTM, self).__init__()
        self.embedding = embedding_layer  
        self.rnn = rnn_layer 
        self.linear_head = linear_head  

    def forward(self, x: torch.Tensor, x_length: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x) 
        rnn_out = self.rnn(embed, x_length) 
        logits = self.linear_head(rnn_out)  
        return logits