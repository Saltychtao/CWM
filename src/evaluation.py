import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import scipy.stats
from corpus import PAD_IDX, pad


class SemEval(object):

    def __init__(self, filepath, dictionary, device):

        self.word1 = []
        self.word2 = []
        self.chars1 = []
        self.chars2 = []
        self.sim = []
        self.device = device

        self.read_file(filepath, dictionary)

    def read_file(self, filepath, dictionary):
        """
        Each line of the gold file should be like: word1 word2 similarity
        """
        with open(filepath, 'r') as f:
            for line in f.readlines():
                word_list = line.split()

                word1 = word_list[0]
                word2 = word_list[1]
                if len(list(word1)) != 2 or len(list(word2)) != 2:
                    continue
                sim = word_list[2]
                self.word1.append(dictionary.encode_word(word1))
                self.word2.append(dictionary.encode_word(word2))
                self.chars1.append(dictionary.encode_char_seq(list(word1)))
                self.chars2.append(dictionary.encode_char_seq(list(word2)))
                self.sim.append(float(sim))

        self.word1 = np.array(self.word1, )
        self.word2 = np.array(self.word2, )
        self.sim = np.array(self.sim)

        ngold = len(self.sim)
        for i in range(ngold):
            self.chars1[i] = pad(self.chars1[i], 2, PAD_IDX)
            self.chars2[i] = pad(self.chars2[i], 2, PAD_IDX)

        self.chars1 = np.matrix(self.chars1)
        self.chars2 = np.matrix(self.chars2)

    @staticmethod
    def spearman_correlation(x, y):
        n = x.shape[0]
        return 1-6*(x-y).sum()/(n*(np.square(n)-1))

    def __call__(self, emb1, emb2):
        # pred_sim = sklearn.metrics.pairwise.paired_cosine_distances(emb1, emb2)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pred_sim = cos(emb1, emb2).detach().cpu().numpy()
        return scipy.stats.spearmanr(pred_sim, self.sim)

    def calculate_emb(self, model):
        x1 = torch.from_numpy(self.chars1).to(self.device)
        y1 = torch.from_numpy(self.word1).to(self.device)
        pred_emb1, gold_emb1 = model(x1, y1, inference=True, test=True)

        x2 = torch.from_numpy(self.chars2).to(self.device)
        y2 = torch.from_numpy(self.word2).to(self.device)
        pred_emb2, gold_emb1 = model(x2, y2, inference=True, test=True)

        return pred_emb1, pred_emb2
