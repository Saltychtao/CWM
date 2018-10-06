import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical
import numpy as np


class Policy(nn.Module):
    def __init__(self, nstate_dim, nhidden, naction, device):
        super(Policy, self).__init__()

        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(nstate_dim, nhidden),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(nhidden, naction),
            nn.Softmax(dim=-1)

        )

        self.policy_history = []
        self.rewards = []
        self.reward_mask = []

        self.reward_history = []
        self.loss_history = []

        self.gamma = 0.99

    def forward(self, cembs):

        return self.proj(cembs)

    def reset(self):
        self.policy_history = []
        self.rewards = []
        self.reward_mask = []


class CoordinateMapping(nn.Module):
    def __init__(self, ninput, nhidden, noutput):
        super(CoordinateMapping, self).__init__()

        self.proj1 = nn.Sequential(
            nn.Linear(2*ninput, nhidden),
            nn.ReLU())

        self.proj2 = nn.Linear(nhidden, noutput)

        self.distance = nn.PairwiseDistance()

    def forward(self, cembs):
        bsize = cembs.size()[0]
        lembs = cembs[:, 0, :]
        rembs = cembs[:, 1, :]

        hidden = self.proj1(cembs.view(bsize, -1))
        hidden_left = self.proj1(torch.cat([lembs, lembs], dim=1))
        hidden_right = self.proj1(torch.cat([rembs, rembs], dim=1))

        return self.proj2(hidden), self.distance(hidden_left, hidden)+self.distance(hidden_right, hidden)+self.distance(hidden_right, hidden_left)


class SubordinateMapping(nn.Module):
    def __init__(self, ninput, nhidden, noutput):
        super(SubordinateMapping, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(2*ninput, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, noutput)
        )

    def forward(self, cembs):
        return self.proj(cembs)


class MLPMapping(nn.Module):
    def __init__(self, ninput, nhidden, noutput):
        super(MLPMapping, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(2*ninput, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, noutput)
        )

    def forward(self, cembs):
        return self.proj(cembs)


class CWMRL(nn.Module):

    def __init__(self, nchars, nwords, ninput_char, ninput_word, nhidden, wemb_pretrained, cemb_pretrained, device, dropout_rate=0.2, nmapping_fn=5):
        super(CWMRL, self).__init__()
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

        self.naction = nmapping_fn

        self.policy = Policy(2*ninput_char, nhidden, self.naction, self.device)

        self.mapping_fns = dict()
        for i in range(nmapping_fn):
            self.mapping_fns[i] = MLPMapping(
                ninput_char, nhidden, ninput_word).to(self.device)

        self.distance = nn.PairwiseDistance()

    def forward(self, x, y=None):
        """
        Return: tensor(bsize x nword_input)
        """

        bsize = x.size()[0]
        cembs = self.char_embs(x).view(bsize, -1)
        action = self.select_action(cembs).view(-1, 1)

        action_mask = torch.FloatTensor(
            bsize, self.naction).to(self.device).zero_().scatter_(1, action, 1)

        mapped = torch.stack([self.mapping_fns[i](cembs)
                              for i in range(self.naction)], dim=1)

        wemb = None

        if y is not None:
            wemb = self.word_embs(y)
            reward = self.calculate_reward(
                torch.unsqueeze(wemb, dim=1), mapped)

            # rewards : [tensor(bsize x naction)]
            self.policy.rewards.append(reward)

            # reward_mask : [tensor(bsize x naction)]
            self.policy.reward_mask.append(action_mask)

            # self.policy.rewards = reward

        return torch.sum(torch.unsqueeze(action_mask, dim=-1) * mapped, dim=1), wemb

    def calculate_reward(self, wemb, mapped, ):

        final_reward = self.distance(
            wemb.transpose(1, 2), mapped.transpose(1, 2))

        return final_reward

    def select_action(self, state):

        state = self.policy(state)

        c = Categorical(state)

        action = c.sample()

        self.policy.policy_history.append(c.log_prob(action))

        # self.policy.policy_history = c.log_prob(action)

        return action

    def reset(self):
        self.constraint_loss = []

    def update_policy(self, optimizer):
        R = 0
        rewards = self.policy.rewards
        # rewards = self.policy.rewards

        # for r in self.policy.rewards[::-1]:
        #     R = r + self.policy.gamma * R
        #     rewards.insert(0, R)

        rewards = torch.stack(rewards, dim=-1)

        masks = torch.stack(self.policy.reward_mask, dim=-1)

        policy_history = torch.stack(self.policy.policy_history, dim=-1)

        masked_rewards = torch.sum(masks * rewards, dim=1)

        # rewards = torch.FloatTensor(rewards).to(self.device)
        # rewards = (rewards - rewards.mean()) / \
        #     (torch.std(rewards))

        reward_loss = (torch.mean(
            (policy_history*masked_rewards).mul_(-1), dim=1))

        # self.policy.loss_history.append(reward_loss.item())

        # self.policy.reward_history.append(np.sum(self.policy.rewards.numpy()))
        self.policy.reset()
        self.reset()

        loss = reward_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
