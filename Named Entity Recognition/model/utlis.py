
from poprogress import simple_progress as simp

def get_label_unique(data):
    unique_label_list = []
    for label in simp(data["labels"]):
        labels =  label.replace('[','').replace(']','').split(',')
        for x in labels:
            tag = x.replace("'",'').replace(' ','')
            if tag not in unique_label_list:
                unique_label_list.append(tag)
    return unique_label_list

def get_tokens_labels(data, id):
    
    def get_sent_labels_list(data, id):
        labels_list = []
        label = data.loc[id, "labels"]
        labels =  label.replace('[','').replace(']','').split(',')
        for x in labels:
            tag = x.replace("'",'').replace(' ','')
            labels_list.append(tag)
        return labels_list
    
    def get_sent_tokens_list(data, id):
        tokens_list = []
        tokens = data.loc[id, "raw_sentence"].split()
        for token in tokens:
            tokens_list.append(token.lower())
        return tokens_list

    tokens_list = get_sent_tokens_list(data, id)
    labels_list = get_sent_labels_list(data, id)
    return tokens_list, labels_list

def get_data_seq(data):
    data_token_seq, data_label_seq = [], []
    for i in range(len(data)):
        a, b = get_tokens_labels(data, i)
        data_token_seq.append(a)
        data_label_seq.append(b)
    return data_token_seq, data_label_seq
def get_token2id(token2cnt, min_count = 1,add_pad = True, add_unk = True):
    '''
    Get mapping from tokens to indices to use with Embedding layer.
    
    param:
        - min_count : Do not mark number if number of words less then this value.
    '''
    token_to_id = {}

    if add_pad:
        token_to_id["<PAD>"] = len(token_to_id)
    if add_unk:
        token_to_id["<UNK>"] = len(token_to_id)

    for token, cnt in token2cnt.items():
        if cnt >= min_count:
            token_to_id[token] = len(token_to_id)

    return token_to_id
