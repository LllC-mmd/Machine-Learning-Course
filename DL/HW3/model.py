import torch
import torch.nn as nn
from basicModule import *


class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput, nhid, nlayers=1, layerNorm=False):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(num_embeddings=nvoc, embedding_dim=ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you RNN model here. You can add additional parameters to the function.
        self.rnn = LSTMBlock(ninput, nhid)
        if layerNorm:
            self.layerNorm_h = nn.LayerNorm(normalized_shape=nhid, elementwise_affine=True)
            self.layerNorm_c = nn.LayerNorm(normalized_shape=nhid, elementwise_affine=True)
        else:
            self.layerNorm_h = None
            self.layerNorm_c = None
        ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        if self.layerNorm_h is not None:
            self.layerNorm_h.weight.data.uniform_(-init_uniform, init_uniform)
            self.layerNorm_h.bias.data.zero_()
            self.layerNorm_c.weight.data.uniform_(-init_uniform, init_uniform)
            self.layerNorm_c.bias.data.zero_()

    def forward(self, inputs, h_prev=None, c_prev=None):
        embeddings = self.drop(self.encoder(inputs))  # embeddings: [seq_len=1, batch_size, embedding_size]
        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        ht, ct = self.rnn(embeddings, h_prev, c_prev)
        if self.layerNorm_h is not None:
            ht = self.layerNorm_h(ht)
            ct = self.layerNorm_c(ct)
        ########################################
        ht = self.drop(ht)    # [seq_len=1, batch_size, hidden_size]
        decoded = self.decoder(ht.view(ht.size(0)*ht.size(1), ht.size(2)))
        decoded = decoded.view(ht.size(0), ht.size(1), decoded.size(1))
        return decoded.squeeze(0), ht, ct

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(1, batch_size, self.nhid), weight.new_zeros(1, batch_size, self.nhid))


class AttLMModel(nn.Module):

    def __init__(self, nvoc, ninput, nhid, nlayers=1, layerNorm=False):
        super(AttLMModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(num_embeddings=nvoc, embedding_dim=ninput)
        ########################################
        # Construct you RNN model here. You can add additional parameters to the function.
        self.attWa = nn.Parameter(torch.Tensor(nhid, nhid))
        self.attUa = nn.Parameter(torch.Tensor(nhid, nhid))
        self.attV = nn.Parameter(torch.Tensor(nhid))
        self.rnn = AttLSTMBlock(ninput, nhid)
        if layerNorm:
            self.layerNorm_h = nn.LayerNorm(normalized_shape=nhid, elementwise_affine=True)
            self.layerNorm_c = nn.LayerNorm(normalized_shape=nhid, elementwise_affine=True)
        else:
            self.layerNorm_h = None
            self.layerNorm_c = None
        #####################################s###
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nvoc = nvoc
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.attWa.data.uniform_(-init_uniform, init_uniform)
        self.attUa.data.uniform_(-init_uniform, init_uniform)
        self.attV.data.zero_()
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)
        if self.layerNorm_h is not None:
            self.layerNorm_h.weight.data.uniform_(-init_uniform, init_uniform)
            self.layerNorm_h.bias.data.zero_()
            self.layerNorm_c.weight.data.uniform_(-init_uniform, init_uniform)
            self.layerNorm_c.bias.data.zero_()

    def forward(self, inputs, memory_pool, h_prev=None, c_prev=None):
        embeddings = self.drop(self.encoder(inputs))      # embeddings: [seq_len=1, batch_size, embedding_size]
        ########################################
        # For torch.matmul(ht, self.attWa): [batch_size, nhid] * [nhid, nhid] => [batch_size, nhid]
        # For torch.matmul(memory_pool, self.attUa): [mem_len, batch_size, nhid] * [nhid, nhid] => [mem_len, batch_size, nhid]
        attWeight = torch.matmul(torch.tanh(torch.matmul(h_prev, self.attWa) + torch.matmul(memory_pool, self.attUa)), self.attV)
        attWeight = torch.softmax(attWeight, dim=0)    # attWeight: [mem_len, batch_size]
        contextVec = torch.sum(attWeight.unsqueeze(2)*memory_pool, dim=0)   # contextVec: [batch_size, nhid]
        att_h = torch.cat((h_prev, contextVec.unsqueeze(0)), dim=2)   # att_c: [1, batch_size, 2*hid]

        ht, ct = self.rnn(embeddings, att_h, c_prev)
        ht = ht[:, :, 0:self.nhid]
        if self.layerNorm_h is not None:
            ht = self.layerNorm_h(ht)
            ct = self.layerNorm_c(ct)
        ########################################
        ht = self.drop(ht)    # [seq_len=1, batch_size, hidden_size]
        decoded = self.decoder(ht.view(ht.size(0) * ht.size(1), ht.size(2)))
        decoded = decoded.view(ht.size(0), ht.size(1), decoded.size(1))

        return decoded.squeeze(0), ht, ct

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(1, batch_size, self.nhid), weight.new_zeros(1, batch_size, self.nhid))