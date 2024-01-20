import torch

import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def split_dataset(data, train_ratio, valid_ratio):

    pool = np.random.rand(len(data)) 
    mask1 = pool < train_ratio
    offset = train_ratio + valid_ratio
    mask2 = (pool >= train_ratio) * (pool < offset)
    train = data[mask1].reset_index(drop=True)
    valid = data[mask2].reset_index(drop=True)
    test = data[~(mask1 + mask2)].reset_index(drop=True)
    
    return train, valid, test

def process_tokens(tokens, token2id, unk: str = "<UNK>"):
    return [token2id.get(token, token2id[unk]) for token in tokens]

def process_labels(labels,label2id):
    return [label2id[label] for label in labels]

class nerDataset(Dataset):

    def __init__(self, token_seq, label_seq, token2id, label2id, preprocess:bool = True):
        self.token2id = token2id
        self.label2id = label2id
        self.preprocess = preprocess
        
        if preprocess:
            self.token_seq = [process_tokens(tokens, token2id) for tokens in token_seq]
            self.label_seq = [process_labels(labels, label2id) for labels in label_seq]
        else:
            self.token_seq = token_seq 
            self.label_seq = label_seq  

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(self, id):
        if self.preprocess:
            tokens = self.token_seq[id]
            labels = self.label_seq[id]
        else:
            tokens = process_tokens(self.token_seq[id], self.token2id) 
            labels = process_labels(self.label_seq[id], self.label2id) 

        lengths = [len(tokens)]

        return np.array(tokens), np.array(labels), np.array(lengths)
    
class nerCollator:

    def __init__(self, token_padding_value, label_padding_value, percentile = 100):
        self.token_padding_value = token_padding_value
        self.label_padding_value = label_padding_value
        self.percentile = percentile

    def __call__(self, batch):

        tokens, labels, lengths = zip(*batch)

        tokens = [list(i) for i in tokens]
        labels = [list(i) for i in labels]
        # 避免句子过长, 应该给个固定长度，暂时不给
        max_len = int(np.percentile(lengths, self.percentile))

        lengths = torch.tensor(np.clip(lengths, a_min=0, a_max=max_len), dtype=torch.long).squeeze(-1)

        for i in range(len(batch)):
            tokens[i] = torch.tensor(tokens[i][:max_len], dtype=torch.long)
            labels[i] = torch.tensor(labels[i][:max_len], dtype=torch.long)

        sorted_idx = torch.argsort(lengths, descending=True)
        # 打补丁
        tokens = pad_sequence(tokens, padding_value=self.token_padding_value, batch_first=True)[sorted_idx]
        labels = pad_sequence(labels, padding_value=self.label_padding_value, batch_first=True)[sorted_idx]
        lengths = lengths[sorted_idx]

        return tokens, labels, lengths