import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tensorboardX import SummaryWriter

from corpus import Dictionary, C2WMDataset, C2WMCollate
from model import C2WMModel
from evaluation import SemEval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(1)


def train(trainFile, validFile, model_save_file):

    ninput_char = 300
    ninput_word = 300
    nhidden = 150
    dropout = 0.5
    batch_size = 128
    epochs = 10000
    lr = 1
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
        validdataset, batch_size=batch_size, shuffle=True, collate_fn=C2WMCollate
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
    model = C2WMModel(nchars,
                      nwords,
                      ninput_char,
                      ninput_word,
                      nhidden,
                      wemb_pretrained,
                      cemb_pretrained,
                      device=device,
                      dropout_rate=dropout
                      ).to(device)
    # optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    #
    writer = SummaryWriter('logs/nonlinear-300hidden')
    global_step = 0

    best_mse = 1e8
    best_spearson = 0
    for epoch in range(epochs):
        global_step = train_epoch(
            traindataloader,
            model,
            optimizer,
            device,
            devdataloader=validdataloader,
            writer=writer,
            global_step=global_step,
            epoch=epoch,
            total_epoch=epochs
        )
        eval_mse = test(validdataloader, model, device, writer, epoch)
        spearson_seen = test_spearman(evaluation1, model, device)
        spearson_unseen = test_spearman(evaluation2, model, device)
        writer.add_scalar('spearson/seen', spearson_seen, epoch)
        writer.add_scalar('spearson/unseen', spearson_unseen, epoch)
        if eval_mse < best_mse:
            best_mse = eval_mse

            print(' Saving model ...')
            torch.save(
                {"state_dict": model.state_dict()},
                model_save_file
            )
    writer.close()
    print('Training finished ! Test Spearman Correlation: {:.2f}, Best Dev MSE :{:.4f}'.format(
        100*spearson_unseen, best_mse))


def train_epoch(dataloader, model, optimizer, device, writer, global_step, epoch, total_epoch):

    model.train()

    # loss_func = nn.CosineEmbeddingLoss(margin=0.5)
    loss_func = nn.MSELoss()
    mean_batch_loss = 0
    total_batches = len(dataloader)
    for i, batch in enumerate(dataloader):

        x = torch.from_numpy(batch["chars"]).to(device)
        y = torch.from_numpy(batch["word"]).to(device).view(-1)
        
        pred_emb, true_emb = model(x, y)
        batch_loss = loss_func(pred_emb, true_emb)

        # l1_reg = 0
        # for name, w in model.named_parameters():
        #     if w.requires_grad and 'mapping' in name:
        #         l1_reg += torch.sum(torch.abs(w))

        loss = batch_loss  # + 0.0001 * l1_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_batch_loss += batch_loss.item()
        

        global_step += 1
        # if global_step % 100 == 0:
        #     writer.add_scalar('training_loss', batch_loss.item(), global_step)

    print(
        '\rEpoch[{}/{}], loss: {:.4f}'.format(total_epoch, epoch, mean_batch_loss/total_batches), end='')

    return global_step


def test(dataloader, model, device, writer=None, global_step=None):

    model.eval()

    # loss_func = nn.CosineEmbeddingLoss(margin=0.5)
    loss_func = nn.MSELoss(reduction='sum')
    mean_batch_loss = 0
    total_batch = len(dataloader)
    total_instance = 0
    for i, batch in enumerate(dataloader):
        x = torch.from_numpy(batch['chars']).to(device)
        y = torch.from_numpy(batch['word']).to(device).view(-1)

        pred_emb, true_emb = model(x, y)
        batch_loss = loss_func(pred_emb, true_emb)

        mean_batch_loss += batch_loss.item()
        total_instance += x.size()[0]

    print('\n[Dev loss: {:.4f}]'.format(mean_batch_loss/total_instance))
    if writer is not None:
        writer.add_scalar('dev_mse', mean_batch_loss /
                          total_instance, global_step)
    return mean_batch_loss/total_batch


def test_spearman(evaluation, model, device, writer=None, global_step=None):
    model.eval()

    emb1, emb2 = evaluation.calculate_emb(model)
    spearman_correlation = evaluation(emb1, emb2)[0]

    if writer is not None and global_step is not None:
        writer.add_scalar('data/spearman_correlation',
                          spearman_correlation, global_step)
    print('Dev Spearman correlation: {:.2f}'.format(
        100*spearman_correlation))
    return spearman_correlation


if __name__ == '__main__':

    train('data/train240.set', 'data/dev240.set', 'models/model.pth')
