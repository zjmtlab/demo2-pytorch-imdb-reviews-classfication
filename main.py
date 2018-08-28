#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import time
import data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import model

torch.manual_seed(1)

trainids, trainseqs, trainlabels = data.load_train_data()
testids, testseqs = data.load_test_data()
word_to_ix = data.load_ix_dics()

train_loader = data.create_dataset(trainids, trainseqs, trainlabels, word_to_ix, True, model.BATCH_SIZE)
test_loader = data.create_dataset(testids, testseqs, [], word_to_ix, False, model.BATCH_SIZE)

lstmmodel = model.LSTMClassifier(model.EMBEDDING_DIM, model.HIDDEN_DIM, len(word_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.Adam(lstmmodel.parameters(), lr=0.001)

for epoch in range(model.EPOCH_NUM):
    total_loss = 0
    for ids, seqs, labels, lengths in train_loader:

        lstmmodel.zero_grad()

        lstmmodel.hidden = lstmmodel.init_hidden()

        tag_score = lstmmodel(seqs,lengths)

        loss = loss_function(tag_score, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print ('Epoch [{}], Loss: {:.4f}' 
               .format(epoch+1, total_loss))

with torch.no_grad():
    with codecs.open("test.result",'w','utf-8') as resultfile:
        resultfile.write("id,sentiment\n")
        for ids, seqs, lengths in test_loader:
            lstmmodel.hidden = lstmmodel.init_hidden()
            tag_scores = lstmmodel(seqs, lengths)
            for id, score in zip(ids, tag_scores):
                resultfile.write('{0},{1}\n'.format(id, score.argmax().item()))
