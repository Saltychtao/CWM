import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tensorboardX import SummaryWriter

from corpus import Dictionary, C2WMDataset, C2WMCollate, DataProvider
from ContextualBandit import CMB
from evaluation import SemEval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(trainFile, validFile, model_save_file):

    ninput_char = 300
    ninput_word = 300
    nhidden = 150
    batch_size = 1
    dev_batch_size = 512
    epochs = 5000
    lr = 1
    epsilon = 0.1
    nextRetrain = 512
    coefficient = 1
    narms = 4
    wemb_filepth = 'data/word2vec_word/wiki.zh.text.simplified.word.vec'
    cemb_filepth = 'data/word2vec_character/wiki.zh.text.simplified.character.vec'
    word_file = 'data/vocab.set'
    char_file = 'data/cleanchar.set'
    seen_wordsim_filepath = 'data/wordsim240seen.clean'
    unseen_wordsim_filepath = 'data/wordsim240unseen.clean'

    dictionary = Dictionary(char_file, word_file, cemb_filepth, wemb_filepth)

    traindataset = C2WMDataset(dictionary)
    traindataset.read_data(trainFile)
    traindataloader = DataLoader(
        traindataset, batch_size=batch_size, shuffle=True, collate_fn=C2WMCollate
    )

    validdataset = C2WMDataset(dictionary)
    validdataset.read_data(validFile)
    validdataloader = DataLoader(
        validdataset, batch_size=dev_batch_size, shuffle=True, collate_fn=C2WMCollate
    )
    nchars = dictionary.nchars()
    nwords = dictionary.nwords()

    evaluation1 = SemEval(seen_wordsim_filepath, dictionary, device)
    evaluation2 = SemEval(unseen_wordsim_filepath, dictionary, device)

    # prepare embedding
    wemb_pretrained = dictionary.prepare_embedding(
        wemb_filepth, ninput_word, type='word')
    cemb_pretrained = dictionary.prepare_embedding(
        cemb_filepth, ninput_char, type='char')

    # model
    model = CMB(narms=narms, ninput=ninput_char, nhidden=nhidden, noutput=ninput_word, device=device,
                epsilon=epsilon, pretrained_cemb=cemb_pretrained, pretrained_wemb=wemb_pretrained).to(device)
    # optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    #
    writer = SummaryWriter('logs/bandit-4emb')
    global_step = 0

    best_spearson = -1e8
    best_mse = 1e8

    dataprovider = DataProvider(traindataloader)
    while global_step < 3000:
        for i in range(nextRetrain):
            batch = dataprovider.next()
            x = torch.from_numpy(batch["chars"]).to(device)
            y = torch.from_numpy(batch["word"]).to(device).view(-1)
            model(x, y, inference=True)

        reward_loss, map_loss = model(x, y, inference=False)
        writer.add_scalar('reward_loss', reward_loss.item(), global_step)
        writer.add_scalar('map_loss', map_loss.item(), global_step)

        loss = reward_loss + map_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        nextRetrain = coefficient * nextRetrain

        seen = test_spearman(evaluation1, model, device,
                             writer=writer, global_step=global_step)
        unseen = test_spearman(evaluation2, model, device,
                               writer=writer, global_step=global_step)
        writer.add_scalar('spearson/seen', seen, global_step)
        writer.add_scalar('spearson/unseen', unseen, global_step)
        test(validdataloader, model, device,
             writer=writer, global_step=global_step)
        global_step += 1

    writer.close()
    print('Training finished ! Best Spearman Correlation: {:.2f}'.format(
        100*best_spearson))


# def train_epoch(dataloader, model, optimizer, device, writer, global_step, epoch, total_epoch):

#     model.train()

#     mean_batch_loss = 0
#     total_batches = len(dataloader)
#     loss_func = nn.MSELoss()
#     for i, batch in enumerate(dataloader):

#         x = torch.from_numpy(batch["chars"]).to(device)
#         y = torch.from_numpy(batch["word"]).to(device).view(-1)
#         pred_emb, true_emb = model(x, y)
#         batch_loss = loss_func(pred_emb, true_emb)

#         mean_batch_loss += batch_loss.item()

#         global_step += 1
#         if global_step % 100 == 0:
#             writer.add_scalar('bandit_training_loss', mean_batch_loss, global_step)

#     print(
#         '\rEpoch[{}/{}], loss: {:.4f}'.format(total_epoch, epoch, mean_batch_loss/total_batches), end='')
#     print('Distribution : {}'.format(model.bandit.history))

#     return global_step


def test(dataloader, model, device, writer=None, global_step=None):

    loss_func = nn.MSELoss(reduction='sum')
    mean_batch_loss = 0
    total_batch = len(dataloader)
    total_instance = 0
    for i, batch in enumerate(dataloader):

        x = torch.from_numpy(batch['chars']).to(device)
        y = torch.from_numpy(batch['word']).to(device).view(-1)
        pred_emb, true_emb = model(x, y, inference=True, test=True)
        batch_loss = loss_func(pred_emb, true_emb)

        mean_batch_loss += batch_loss.item()
        total_instance += x.size()[0]

    print('[Dev loss: {:.4f}]'.format(mean_batch_loss/total_instance))
    if writer is not None:
        writer.add_scalar('dev_mse', mean_batch_loss /
                          total_instance, global_step)
    return mean_batch_loss/total_batch


def test_spearman(evaluation, model, device, writer=None, global_step=None):
    model.eval()

    emb1, emb2 = evaluation.calculate_emb(model)
    spearman_correlation = evaluation(emb1, emb2)[0]

    # if writer is not None and global_step is not None:
    #     writer.add_scalar('spearman_correlation',
    #                       spearman_correlation, global_step)
    print('Dev Spearman correlation: {:.2f}'.format(
        100*spearman_correlation))

    return spearman_correlation


if __name__ == '__main__':

    train('data/train240.set', 'data/dev240.set', 'models/model.pth')
