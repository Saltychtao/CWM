import torch
import torch.nn as nn
import torch.functional as F
from corpus import PAD_IDX


class MLPLayer(nn.Module):
    def __init__(self, ninput_char, ninput_word, nhidden, dropout_rate=0.2,):
        super(MLPLayer, self).__init__()

        self.ninput_char = ninput_char
        self.ninput_word = ninput_word

        self.layer = nn.Sequential(
            nn.Linear(2*ninput_char, nhidden, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(nhidden, ninput_word, bias=False)
        )

    def forward(self, cembs, length):
        # bow = torch.div(torch.t(torch.sum(cembs, 1).squeeze()), length).t()
        return self.layer(cembs)
        # return bow


def init_linear(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)


class C2WMModel(nn.Module):

    def __init__(self, nchars, nwords, ninput_char, ninput_word, nhidden, wemb_pretrained, cemb_pretrained, device, dropout_rate=0.2):
        super(C2WMModel, self).__init__()
        self.nchars = nchars
        self.nwords = nwords
        self.ninput_char = ninput_char
        self.ninput_word = ninput_word
        self.nhidden = nhidden
        self.dropout_rate = dropout_rate
        self.device = device

        self.char_embs = nn.Embedding.from_pretrained(
            cemb_pretrained, freeze=True)
        self.word_embs = nn.Embedding.from_pretrained(
            wemb_pretrained, freeze=True)

        self.mapping = MLPLayer(
            ninput_char, ninput_word, nhidden, dropout_rate)

    def forward(self, x, y=None,inference=False,test=False):
        bsize = x.size()[0]
        cembs = self.char_embs(x).view(bsize, -1)

        length = (x != PAD_IDX).sum(dim=1).float()
        wemb = None
        if y is not None:
            wemb = self.word_embs(y)
        # bow = torch.div(torch.t(torch.sum(cembs, 1).squeeze()), length).t()
        return self.mapping(cembs, length), wemb
