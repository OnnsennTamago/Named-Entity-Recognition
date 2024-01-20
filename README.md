# NER
NER - Named Entity Recognition
- Rule-based: The code will be tedious when we want to deal with complex text.

# 1 Download data
Data sources: https://github.com/orestxherija/conll-balanced

# 2 Preprocess
Use preprocess.ipynb to get train.csv and test.csv.

# 3 Build model
BiLSTM
BiLSTM + CRF

# 4 Problem
1. The corpus extracted from train_dataset is not big enough. Therefore, some words can not be recognised and classified.

2. CRF learning effect is not better as BiLSTM. The reason may be the learning rate of CRF is not big enough. The model uses the same lr and the loss combined with BiLSTM loss and CRF loss to update parameters.

    This problem is also mentioned at https://kexue.fm/archives/7196.
