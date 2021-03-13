import torch
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF
from utils import *


class LSTMCRF(nn.Module):
    def __init__(self, words_num, embed_dim, hidden_dim, num_layers, out_class, word2idx, dropout = 0.2, bi_direction = True):
        super(LSTMCRF, self).__init__()
        self.word2idx = word2idx
        self.bi_direction = bi_direction
        self.hidden_dim = hidden_dim
        self.embed_layer = nn.Embedding(words_num, embed_dim)
        if bi_direction:
            self.rnn = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, out_class)
        self.crf=CRF(out_class)
    
    def __build_lstm(self, sentences):
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = nn.utils.rnn.pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks
    def loss(self, sentences, tags):
        lstm_out, masks = self.__build_lstm(sentences)
        feats = self.fc(lstm_out)
        loss = self.crf.loss(feats, tags, masks=masks)
        return loss
    def forward(self, sentences):
        lstm_out, masks = self.__build_lstm(sentences)
        feats = self.fc(lstm_out)
        scores, tag_seq = self.crf(feats, masks)
        return scores, tag_seq


        
