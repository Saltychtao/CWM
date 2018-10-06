import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.distributions import Categorical
from collections import defaultdict


class CMB(nn.Module):
    class ContextualBandit(nn.Module):
        def __init__(self, ninput, nhidden, noutput, narms,  device):
            super(CMB.ContextualBandit, self).__init__()

            self.device = device
            self.narmsdim = 50

            self.narms = narms
            self.aembs = nn.Embedding(narms, self.narmsdim)
            self.arms = np.empty(narms, dtype=object)
            self.distance = nn.PairwiseDistance()

            # self.arms[i]['input2hidden'] = nn.Linear(
            #     ninput, nhidden).to(device)
            # self.arms[i]['relu'] = nn.ReLU().to(device)
            # self.arms[i]['hidden2output'] = nn.Linear(
            #     nhidden, noutput).to(device)
            self.input2hidden_list = nn.ModuleList(
                [nn.Linear(2*ninput, nhidden).to(device) for i in range(self.narms)])
            self.relu_list = nn.ModuleList(
                [nn.ReLU().to(device) for i in range(self.narms)])
            self.hidden2wemb_list = nn.ModuleList(
                [nn.Linear(nhidden, noutput).to(device) for i in range(self.narms)])

            self.input2reward = nn.Sequential(nn.Linear(
                3*ninput, nhidden), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(nhidden, 1))

            self.history = defaultdict(lambda: 0)

        def forward(self, i, input, dropout=None):
            self.history[i] += 1
            hidden = self.input2hidden_list[i](input)
            if dropout is not None:
                output = self.hidden2wemb_list[i](
                    dropout(hidden))

            else:
                output = self.hidden2wemb_list[i](hidden)
            reward = self.input2reward(
                torch.cat([input, output], dim=1))
            return output, reward

    class Agent(nn.Module):
        def __init__(self, narms, ninput, nhidden, device):
            super(CMB.Agent, self).__init__()
            self.q_arms = torch.FloatTensor(narms).to(device)
            self.narms = narms
            self.device = device

            self.action_network = nn.Sequential(
                nn.Linear(2*ninput, nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, narms),
            )

        def forward(self, context, bandit, epsilon):
            output = self.action_network(context)
            choosen_action = torch.argmax(output)
            prob = (torch.LongTensor(
                [i for i in range(self.narms)]).to(self.device) == choosen_action).float() + torch.FloatTensor([epsilon]).to(self.device) / self.narms

            c = Categorical(prob)
            action = c.sample()
            predict = bandit(action, context)
            return action, predict

    def __init__(self, narms, ninput, nhidden, noutput, device, epsilon, pretrained_cemb, pretrained_wemb):
        super(CMB, self).__init__()
        self.narms = narms

        self.noutput = noutput

        self.bandit = CMB.ContextualBandit(
            ninput, nhidden, noutput, narms, device).to(device)

        self.init_weight()

        self.cembs = nn.Embedding.from_pretrained(pretrained_cemb, freeze=True)
        self.wembs = nn.Embedding.from_pretrained(pretrained_wemb, freeze=True)

        self.device = device

        self.epsilon = epsilon

        self.distance = nn.PairwiseDistance()

        self.loss_fn = nn.MSELoss()
        self.tanh = nn.Tanh()

        # self.agent = CMB.Agent(narms, ninput, nhidden, device).to(device)

        # self.input2hidden = nn.Sequential(
        #     nn.Linear(2*ninput, nhidden),
        #     nn.ReLU(),
        # )
        # self.hidden2output = nn.Linear(nhidden, noutput)

        self.pool = []

    def forward(self, x, y=None, inference=False, test=False):

        if test:
            self.eval()
        else:
            self.train()

        bsize = x.size()[0]
        cembs = self.cembs(x.squeeze()).view(bsize, -1)

        if inference:

            d = np.random.random()
            dropout = nn.Dropout(p=0.2)

            wemb = self.wembs(y)

            rewards = []
            preds = []
            for i in range(self.narms):
                pred, reward = self.bandit(i, cembs, dropout)
                preds.append(pred)
                rewards.append(reward)

            if not test:
                a = np.argmin(rewards)
                self.pool.append((cembs, a, wemb))
            else:
                a = torch.argmin(
                    torch.cat(rewards, dim=1), dim=1)
                preds = torch.stack(preds, dim=1,)
                r = torch.zeros(bsize, self.noutput).to(self.device)
                for i in range(bsize):
                    for j in range(self.noutput):
                        r[i][j] = preds[i][a[i]][j]

                return r, wemb

        else:
            reward_loss = []
            map_loss = []
            total_len = len(self.pool)
            for (x, a, gold) in self.pool:
                pred, expect_reward = self.bandit(a, x)
                true_reward = self.loss_fn(pred, gold)
                map_loss.append(true_reward)
                reward_loss.append(torch.sqrt(self.loss_fn(
                    true_reward, expect_reward)))
            self.pool = []
            reward_loss = torch.sum(torch.stack(reward_loss)).div(total_len)
            map_loss = torch.sum(torch.stack(map_loss)).div(total_len)
            return reward_loss, map_loss

    def init_weight(self):
        for module in self.modules():
            if hasattr(module, 'weight'):
                if not ('BatchNorm' in module.__class__.__name__):
                    init.xavier_uniform_(module.weight, gain=1)
                else:
                    init.constant_(module.weight, 1)
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    init.constant_(module.bias, 0)
