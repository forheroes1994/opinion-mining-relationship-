import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
torch.manual_seed(123)
random.seed(123)

class LSTM(nn.Module):
    def __init__(self, config, params):
        super(LSTM, self).__init__()
        self.static = params.static
        self.use_cuda = params.use_cuda


        self.word_num = params.word_num
        self.label_num = params.label_num
        self.sparse_num = params.sparse_num

        self.word_dims = config.word_dims
        self.sparse_word_dims = config.sparse_word_dims

        self.lstm_hiddens = config.lstm_hiddens

        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.batch_size

        self.embedding = nn.Embedding(self.word_num, self.word_dims)
        self.sparse_embedding = nn.Embedding(self.sparse_num, self.sparse_word_dims)

        self.embedding.weight.requires_grad = True
        if self.static:
            self.embedding_static = nn.Embedding(self.word_num, self.word_dims)
            self.embedding_static.weight.requires_grad = False

        if params.pretrain_word_embedding is not None:
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        if self.static:
            self.lstm = nn.LSTM(self.word_dims*2, self.lstm_hiddens // 2, bidirectional=True, dropout=config.dropout_lstm)
        else:
            self.lstm = nn.LSTM(self.word_dims, self.lstm_hiddens // 2, bidirectional=True, dropout=config.dropout_lstm)

        self.hidden2label = nn.Linear(self.lstm_hiddens * 3, self.label_num)
        # self.hidden = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)).cuda(),
                     Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)).cuda())
        else:
            return (Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)),
                     Variable(torch.zeros(2, batch_size, self.lstm_hiddens // 2)))

    def forward(self, sentence_var, sentence_mask_var, pairs_var, pairs_mask_var, sparse_var, batch_length, pairs_length):
        sentence_word_emb = self.embedding(sentence_var)
        pair_word_emb = self.embedding(pairs_var)
        sparse_word_emb = self.sparse_embedding(sparse_var)

        sentence_out = torch.transpose(sentence_word_emb, 0, 1)
        # print(sentence_x)
        # print(batch_length)
        sentence_packed_words = pack_padded_sequence(sentence_out, batch_length)
        sentence_out, self.hidden = self.lstm(sentence_packed_words)
        sentence_out, _ = pad_packed_sequence(sentence_out)
        sentence_out = torch.transpose(sentence_out, 0, 2)

        sentence_out = F.max_pool1d(sentence_out, sentence_out.size(2)).squeeze(2)
        sentence_out = torch.transpose(sentence_out, 0, 1)

        # sentence_out = self.dropout_lstm(sentence_out)

        pair_out = torch.transpose(pair_word_emb, 0, 1)
        pair_packed_words = pack_padded_sequence(pair_out, pairs_length)
        pair_out, self.hidden = self.lstm(pair_packed_words)
        pair_out, _ = pad_packed_sequence(pair_out)
        pair_out = torch.transpose(pair_out, 0, 2)

        pair_out = F.max_pool1d(pair_out, pair_out.size(2)).squeeze(2)
        pair_out = torch.transpose(pair_out, 0, 1)
        # pair_out = self.dropout_lstm(pair_out)



        # print(sparse_word_emb)
        sparse_out = torch.transpose(sparse_word_emb, 1, 2)
        sparse_out = F.max_pool1d(sparse_out, sparse_out.size(2)).squeeze(2)
        output = torch.cat((sentence_out, sparse_out,sparse_out), 1)

        output = self.dropout_lstm(output)
        output = self.hidden2label(F.tanh(output))

        return output