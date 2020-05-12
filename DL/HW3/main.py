# coding: utf-8
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import data
import model


#writer = SummaryWriter()

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')

parser.add_argument('--n_input', type=int, default=200, help='embedding dimension')
parser.add_argument('--n_hidden', type=int, default=200, help='hidden state dimension')

parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N', help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35, help='sequence length')

parser.add_argument('--seed', type=int, default=1234, help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')

parser.add_argument('--attFlag', type=bool, default=True, help='whether use Attention mechanism')
parser.add_argument('--layerNormFlag', type=bool, default=False, help='whether use Layer Normalization')
parser.add_argument('--lr_Flag', type=bool, default=True, help='whether use learning rate strategy')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = False

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)


# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)
nvoc = len(data_loader.vocabulary)
max_sql = data_loader.max_sql

if args.attFlag:
    LMmodel = model.AttLMModel(nvoc, args.n_input, args.n_hidden, layerNorm=args.layerNormFlag)
else:
    LMmodel = model.LMModel(nvoc, args.n_input, args.n_hidden, layerNorm=args.layerNormFlag)

LMmodel = LMmodel.to(device)
########################################

if args.lr_Flag:
    optimizer = optim.SGD(LMmodel.parameters(), lr=0.1, momentum=0.9)
    lr_strategy = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
else:
    optimizer = optim.SGD(LMmodel.parameters(), lr=0.01, momentum=0.9)
    lr_strategy = None

criterion = nn.CrossEntropyLoss()


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
def evaluate(model, data_loader, criterion, epoch):
    model.train(False)
    data_loader.set_valid()
    total_loss = 0.0
    total_len = 0
    while True:
        h_ini, c_ini = model.init_hidden(args.eval_batch_size)
        state_list = [(None, h_ini, c_ini)]
        data, target, end_flag = data_loader.get_batch()
        if end_flag:
            break
        else:
            data = data.to(device)  # data: (seq_len, batch)
            target = target.to(device)
            seq_len = data.size(0)
            batch_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            for i in range(0, seq_len):
                word = data[i].unsqueeze(0)
                word_next = target[i]
                h_prev = state_list[-1][1]
                c_prev = state_list[-1][2]
                yt, ht, ct = model(word, h_prev, c_prev)
                state_list.append((yt, ht, ct))
                batch_loss = batch_loss + criterion(yt, word_next)

            total_loss += batch_loss.item()
            total_len += seq_len

    return total_loss/total_len


def evaluateAtt(model, data_loader, criterion, epoch):
    model.train(False)
    data_loader.set_valid()
    total_loss = 0.0
    total_len = 0
    while True:
        h_ini, c_ini = model.init_hidden(args.eval_batch_size)
        state_list = [(None, h_ini, c_ini)]
        data, target, end_flag = data_loader.get_batch()
        if end_flag:
            break
        else:
            data = data.to(device)
            target = target.to(device)
            seq_len = data.size(0)
            memory_pool = h_ini
            batch_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            for i in range(0, seq_len):
                word = data[i].unsqueeze(0)
                word_next = target[i]
                h_prev = state_list[-1][1]
                c_prev = state_list[-1][2]
                yt, ht, ct = model(word, memory_pool, h_prev, c_prev)
                memory_pool = torch.cat((memory_pool, ht), dim=0)
                state_list.append((yt, ht, ct))
                batch_loss = batch_loss + criterion(yt, word_next)

            total_loss += batch_loss.item()
            total_len += seq_len

    return total_loss / total_len

########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train(model, data_loader, criterion, optimizer, epoch):
    model.train(True)
    data_loader.set_train()
    total_loss = 0.0
    total_len = 0
    while True:
        h_ini, c_ini = model.init_hidden(args.train_batch_size)
        state_list = [(None, h_ini, c_ini)]
        data, target, end_flag = data_loader.get_batch()
        if end_flag:
            break
        else:
            data = data.to(device)  # data: (seq_len, batch)
            target = target.to(device)
            seq_len = data.size(0)
            batch_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            optimizer.zero_grad()
            for i in range(0, seq_len):
                word = data[i].unsqueeze(0)
                word_next = target[i]
                h_prev = state_list[-1][1]
                c_prev = state_list[-1][2]
                yt, ht, ct = model(word, h_prev, c_prev)
                state_list.append((yt, ht, ct))
                batch_loss = batch_loss + criterion(yt, word_next)

            total_loss += batch_loss.item()
            total_len += seq_len

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

    return total_loss / total_len


def trainAtt(model, data_loader, criterion, optimizer, epoch):
    model.train(True)
    data_loader.set_train()
    total_loss = 0.0
    total_len = 0
    while True:
        h_ini, c_ini = model.init_hidden(args.train_batch_size)
        state_list = [(None, h_ini, c_ini)]
        data, target, end_flag = data_loader.get_batch()
        if end_flag:
            break
        else:
            data = data.to(device)
            target = target.to(device)
            seq_len = data.size(0)
            memory_pool = h_ini
            batch_loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            optimizer.zero_grad()
            for i in range(0, seq_len):
                word = data[i].unsqueeze(0)
                word_next = target[i]
                h_prev = state_list[-1][1]
                c_prev = state_list[-1][2]
                yt, ht, ct = model(word, memory_pool, h_prev, c_prev)
                memory_pool = torch.cat((memory_pool, ht), dim=0)
                state_list.append((yt, ht, ct))
                batch_loss = batch_loss + criterion(yt, word_next)

            total_loss += batch_loss.item()
            total_len += seq_len

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

    return total_loss / total_len
########################################


# Loop over epochs.
if args.attFlag:
    # LSTM model with attention
    for epoch in range(1, args.epochs+1):
        train_mean_loss = trainAtt(LMmodel, data_loader, criterion, optimizer, epoch)
        print("Training, Perplexity at Epoch", epoch, ":\t", np.exp(train_mean_loss))
        eval_mean_loss = evaluateAtt(LMmodel, data_loader, criterion, epoch)
        print("Evaluation, Perplexity at Epoch", epoch, ":\t", np.exp(eval_mean_loss))
        if lr_strategy is not None:
            lr_strategy.step()
else:
    # LSTM model
    for epoch in range(1, args.epochs+1):
        train_mean_loss = train(LMmodel, data_loader, criterion, optimizer, epoch)
        print("Training, Perplexity at Epoch", epoch, ":\t", np.exp(train_mean_loss))
        eval_mean_loss = evaluate(LMmodel, data_loader, criterion, epoch)
        print("Evaluation, Perplexity at Epoch", epoch, ":\t", np.exp(eval_mean_loss))
        if lr_strategy is not None:
            lr_strategy.step()
