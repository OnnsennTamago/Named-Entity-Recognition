
import pandas as pd

from utlis import *
from model import *
from dataset import *
from metrics import *

from collections import Counter
from collections import defaultdict
from torch.utils.data import DataLoader
from poprogress import simple_progress as simp

# load data
all_data = pd.read_csv("./Named Entity Recognition/data_preprocess/all-data.csv")
all_len = len(all_data)
print("all_len: ",all_len)

# split data
train_data, valid_data, test_data = split_dataset(all_data, 0.7, 0.15)
print("train_data_size: ",len(train_data))
print("valid_data_size: ",len(valid_data))
print("test_data_size: ",len(test_data))
print("Spliting data done")
print("-"*30)

# get unique labels
label_unique = sorted(get_label_unique(train_data))

# get dicts
label_to_id = {k: v for v,k in enumerate(label_unique)}
id_to_label = {k: v for k,v in enumerate(label_unique)}
print(label_to_id)
print(id_to_label)

# get seq
train_token_seq, train_label_seq = get_data_seq(train_data)
valid_token_seq, valid_label_seq = get_data_seq(valid_data)
test_token_seq, test_label_seq = get_data_seq(test_data)
print("Get sequences done")
print("-"*30)

# get token -> id and label -> id
token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
label_set = sorted(set(label for sentence in train_label_seq for label in sentence))
token_to_id = get_token2id(token2cnt)
print("Encoding data done")
print("-"*30)

# dataset
train_set = nerDataset(train_token_seq, train_label_seq, token_to_id, label_to_id, preprocess=True)
valid_set = nerDataset(valid_token_seq, valid_label_seq, token_to_id, label_to_id, preprocess=True)
test_set = nerDataset(test_token_seq, test_label_seq, token_to_id, label_to_id, preprocess=True)
print("Making datasets done")
print("-"*30)

# dataloader
train_coll_fn = nerCollator(token_to_id["<UNK>"], label_to_id["O"], 100)
valid_coll_fn = nerCollator(token_to_id["<UNK>"], label_to_id["O"], 100)
test_coll_fn = nerCollator(token_to_id["<UNK>"], label_to_id["O"], 100)

train_loader = DataLoader(dataset=train_set, batch_size=256, shuffle=False, collate_fn=train_coll_fn)
valid_loader = DataLoader(dataset=valid_set, batch_size=256, shuffle=False, collate_fn=valid_coll_fn)
test_loader = DataLoader(dataset=test_set, batch_size=256, shuffle=False, collate_fn=test_coll_fn)
print("Making Dataloaders done")
print("-"*30)

embedding_layer = Embedding(num_embeddings=len(token_to_id), embedding_dim=128)

rnn_layer = RNN(rnn_unit=torch.nn.LSTM, input_size=128, hidden_size=256, 
                num_layers=1, dropout=0, bidirectional=True)

linear_head = LinearHead(linear_head=torch.nn.Linear(in_features=(2*256), 
                                                     out_features=len(label_to_id)))

model = BiLSTM(embedding_layer=embedding_layer, rnn_layer=rnn_layer, linear_head=linear_head)#.to(device)
print("Setting models done")
print("-"*30)

criterion = torch.nn.CrossEntropyLoss(reduction="none")
optimizer_type = torch.optim.Adam
optimizer = optimizer_type(params=model.parameters(), lr=0.001, amsgrad=False)
print("Setting metrics done")
print("-"*30)

verbose = True
n_epoch = 1
clip_grad_norm = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range(n_epoch):

    metrics = defaultdict(list)
    model.train()

    for tokens, labels, lengths in simp(train_loader):
        tokens, labels, lengths = (tokens.to(device), labels.to(device), lengths.to(device))

        mask = masking(lengths)

        # forward pass
        logits = model(tokens, lengths)
        loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
        loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

        # backward pass
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm, norm_type=2)

        optimizer.step()
        optimizer.zero_grad()

        # make predictions
        y_true = to_numpy(labels[mask])
        y_pred = to_numpy(logits.argmax(dim=-1)[mask])

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            y_true=y_true,
            y_pred=y_pred,
            idx2label=id_to_label,
        )
        
