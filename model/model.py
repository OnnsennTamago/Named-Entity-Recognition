import torch

from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    
class dynamicRNN(torch.nn.Module):

    def __init__(self, rnn_unit:torch.nn.Module, 
                 input_size, hidden_size, num_layers, dropout, bidirectional):
        super(dynamicRNN, self).__init__()
        
        self.rnn = rnn_unit(input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        
    def forward(self, x, x_length):
        packed_x = pack_padded_sequence(x, x_length.cpu(), batch_first=True, enforce_sorted=True)
        packed_rnn_out, (self.h, self.c) = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
        return rnn_out
    
class LinearHead(torch.nn.Module):
    def __init__(self, linear_head):
        super(LinearHead, self).__init__()
        self.linear_head = linear_head
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear_head(x))
    
class BiLSTM(torch.nn.Module):
    def __init__(self, embedding_layer, rnn_layer, linear_head):
        super(BiLSTM, self).__init__()
        self.embedding = embedding_layer  
        self.rnn = rnn_layer 
        self.linear_head = linear_head  

    def forward(self, x, x_length):
        embed = self.embedding(x) 
        rnn_out = self.rnn(embed, x_length) 
        logits = self.linear_head(rnn_out)  
        return logits