import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DynamicRNN(nn.Module):
    """
    RNN layer wrapper to handle variable-size input.
    """

    def __init__(
        self,
        rnn_unit: nn.Module,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super(DynamicRNN, self).__init__()
        self.rnn = rnn_unit(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x, x_length):
        packed_x = pack_padded_sequence(
            x, x_length.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_rnn_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
        return rnn_out
    
class RNN(torch.nn.Module):
    def __init__(self, word_count,embedding_size, hidden_size, n_class):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # 为什么要embedding
        self.embedding = torch.nn.Embedding(word_count, embedding_size)
        self.n_class = n_class
        
        # input to hidden state
        self.i2s = torch.nn.Linear(embedding_size + hidden_size, hidden_size)
        # input to output
        self.i2o = torch.nn.Linear(embedding_size + hidden_size, n_class)
        self.tanh = torch.nn.functional.tanh()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, pre_hidden):
        if pre_hidden is None:
            pre_hidden = torch.zeros(1, self.hidden_size)
        # 为什么要embedding一下，输入的input_tensor到底是什么
        word_vetor = self.embedding(input_tensor)
        combined = torch.cat((word_vetor, pre_hidden), 1)
        hidden = self.i2s(combined)
        hidden = self.tanh(hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        
        # if self.n_class == 1:
        #     output = torch.nn.functional.sigmoid(output)
        # else:
        #     output = self.softmax(output)
        
        return output, hidden
    