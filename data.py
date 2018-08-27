#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import codecs
from torch.utils.data import DataLoader, Dataset
import re
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stoplist = stopwords.words('english')

def review_to_words(raw_review):

    review_text = BeautifulSoup(raw_review, "lxml").get_text()

    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    words = letters_only.lower().split()

    words = [w for w in words if not w in stoplist]

    return words[:500]

def load_train_data():
    
    train = pd.read_csv("./data/labeledTrainData.tsv", header= 0, \
                    delimiter="\t", quoting=3)

    clean_review_list = []

    for i in train["review"]:
        clean_review_list.append(review_to_words(i))

    return train["id"], clean_review_list, train["sentiment"]

def load_test_data():
    
    test = pd.read_csv("./data/testData.tsv", header= 0, \
                    delimiter="\t", quoting=3)

    clean_review_list = []

    for i in test["review"]:
        clean_review_list.append(review_to_words(i))

    return test["id"], clean_review_list

def load_ix_dics():

    train = pd.read_csv("./data/unigram_freq.csv", header= 0, \
                    delimiter=",", quoting=3)
    
    word_to_ix = {}
    word_to_ix['UNK'] = 0
    for word in train["word"][:50000]:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    return word_to_ix

def vectorize_data(seqs, to_ix):
    return [[to_ix[tok] if tok in to_ix else to_ix['UNK'] for tok in seq] for seq in seqs]

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    if len(batch[0]) == 4:
        ids, seqs, labels, lens = zip(*batch)

        seqs = pad_sequence([torch.LongTensor(seq) for seq in seqs], True)
        labels_tensor = torch.LongTensor(labels)
        lens_tensor = torch.LongTensor(lens)

        return ids, seqs, labels_tensor, lens_tensor
    else:
        ids, seqs, lens = zip(*batch)

        seqs = pad_sequence([torch.LongTensor(seq) for seq in seqs], True)
        lens_tensor = torch.LongTensor(lens)

        return ids, seqs, lens_tensor


def create_dataset(ids, seqs, labels, word_to_ix, train, bs=4):
    vectorized_seqs = vectorize_data(seqs, word_to_ix)
    seq_lengths = [len(s) for s in vectorized_seqs]
    return DataLoader(IMDBDataset(ids, vectorized_seqs, labels, seq_lengths, train),
                      batch_size=bs,
                      shuffle=False,
                      collate_fn=collate_fn,
                      drop_last=False,
                      num_workers=0)


class IMDBDataset(Dataset):
    def __init__(self, ids, sequences, labels, lens, train):
        self.seqs = sequences
        self.labels = labels
        self.lens = lens
        self.ids = ids
        self.train = train

    def __getitem__(self, index):
        if self.train:
            return self.ids[index], self.seqs[index], self.labels[index], self.lens[index]
        else:
            return self.ids[index], self.seqs[index], self.lens[index]

    def __len__(self):
        return len(self.seqs)

