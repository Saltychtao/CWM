import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import gensim
import logging

PAD_IDX = 0
PAD = '<PAD>'


class DataProvider(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)
        self.iteration = 0
        self.epoch = 0

    def next(self):
        try:
            batch = next(self.dataiter)
            self.iteration += 1
            return batch
        except StopIteration:
            self.epoch += 1
            self.iteration = 0
            self.dataiter = iter(self.dataloader)

            batch = next(self.dataiter)
            return batch


class Dictionary(object):

    def __init__(self, char_file, word_file, cemb_file, wemb_file):
        self.char2idx = {PAD: PAD_IDX}
        self.idx2char = {PAD_IDX: PAD}
        self.word2idx = {}
        self.idx2word = {}
        self.char_idx = 1
        self.word_idx = 0

        self.read_char_file(char_file, cemb_file)
        print('Building Char vocab... Done. Total Chars: %d' %
              len(self.char2idx))

        self.read_word_file(word_file, wemb_file)
        print('Building Word vocab... Done. Total Words: %d' %
              len(self.word2idx))

    def add_character(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.char_idx
            self.idx2char[self.char_idx] = char
            self.char_idx += 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.word_idx
            self.idx2word[self.word_idx] = word
            self.word_idx += 1

    def encode_char(self, char):
        if char in self.char2idx:
            return self.char2idx[char]

    def encode_word(self, word):
        if word in self.word2idx:
            return self.word2idx[word]

    def encode_char_seq(self, char_seq):
        ret = []
        for c in char_seq:
            ret.append(self.encode_char(c))
        return ret

    def read_char_file(self, filepath, cemb_file):

        with open(filepath, 'r') as f:
            for line in f.readlines():
                for c in line.split():
                    self.add_character(c)

    def read_word_file(self, filepath, wemb_file):

        with open(filepath, 'r') as f:
            for line in f.readlines():
                for w in line.split():
                    self.add_word(w)

    def prepare_embedding(self, filepath, embedding_dim, type):
        w2v = gensim.models.Word2Vec.load(filepath)
        print('Word2Vec of %s Loaded' % type)

        dic = self.word2idx if type == 'word' else self.char2idx
        emb_pretrained = np.zeros((len(dic), embedding_dim), dtype='float64')

        for i, t in enumerate(dic):
            if t in w2v:
                emb_pretrained[i] = w2v[t]

        return torch.FloatTensor(emb_pretrained)

    def nchars(self):
        return len(self.char2idx)

    def nwords(self):
        return len(self.word2idx)


class C2WMDataset(Dataset):

    @staticmethod
    def ishan(string):
        return all('\u4e00' <= char <= '\u9fff' for char in string)

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.data = []

    def read_data(self, path):
        with open(path, 'r') as f:
            for line in f.readlines():
                if len(line.split()) == 0:
                    continue
                word = line.split('\n')[0]
                if not C2WMDataset.ishan(word):
                    continue
                if not all([c in self.dictionary.char2idx for c in list(word)]):
                    continue
                if not word in self.dictionary.word2idx:
                    continue
                self.data.append({'chars': self.dictionary.encode_char_seq(
                    list(word)), 'word': self.dictionary.encode_word(word)})

            print('Total training data: {}'.format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad(seq, size, value):
    if len(seq) < size:
        seq.extend([value] * (size - len(seq)))
    return seq


def C2WMCollate(batch):
    batch_x = []
    batch_y = []
    max_len = -1
    for b in batch:
        batch_x.append(b["chars"])
        batch_y.append(b["word"])
        max_len = max(len(b['chars']), max_len)

    for i in range(len(batch_x)):
        batch_x[i] = pad(batch_x[i], max_len, PAD_IDX)

    ret_batch = {
        "chars": np.array(batch_x),
        "word": np.array(batch_y)
    }
    return ret_batch


class StructureClassificationDataset(Dataset):
    def __init__():
        pass
