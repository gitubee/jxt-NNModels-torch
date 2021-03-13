import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

START_TAG_IDX = 0
END_TAG_IDX = 4

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(x):
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()
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


        
class CRF(nn.Module):
    def __init__(self, tag_size):
        super(CRF, self).__init__()
        #set the Parameters
        self.tag_size = tag_size
        #crf transform matrix, trans[i, j] means transform score from j to i
        self.crf_trans=nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.crf_trans.data[START_TAG_IDX, :] = -10000.
        self.crf_trans.data[:, END_TAG_IDX] = -10000.

    #forward calculate the all tags partition
    #input feats is each tags PR of sentence
    def _forward_alg(self, feats, masks):
        # get all paths scores sum
        # feats:[B, S, T]
        n_batchs, seq_len, tag_s = feats.size()
        #init the partition vector set the start_pos to 1.
        forward_scores = torch.fill((n_batchs, self.tag_size), -10000.)
        forward_scores[:, START_TAG_IDX] = 0.
        trans = self.crf_trans.unsqueeze(0)
        for i in range(seq_len):
            # emit matrix from the feat, feats[B,i,T] ->[B, T, 1]
            # forward matrix is [B, T], unsqueeze to [B, 1, T]
            emit_score=feats[:, i].unsqueeze(2)
            # cal the new score matrix and log_sum_exp
            # trans + emits + forward : [1, T, T] -> [B, T, T] + [B, T, 1] -> [B, T, T] + [B, 1, T] -> [B, T, T]
            t_scores = emit_score + forward_scores.unsqueeze(1) + trans
            t_scores = log_sum_exp(t_scores)
            # mask block seq
            # this position mask [B, 1]
            mask = mask[:, i].unsqueeze(1)
            forward_scores = t_scores * mask + forward_scores * (1 - mask)

        return log_sum_exp(forward_scores + self.crf_trans[END_TAG_IDX])
    def _score_sentence(self, feats, tags, masks):
        # get score of given path
        # sentence: [B, S, T]
        # cal sentences score
        n_batchs, seq_len, tag_s = feats.size()
        # select tags emits in feats
        emits = feats.gather(2, tags).view(n_batchs, -1)
        start_tag = torch.full((n_batchs, 1), START_TAG_IDX, dtype=torch.long)
        # select tags trans scores in crf_trans
        tags = torch.cat([start_tag, tags], dim=1)
        trans = self.transitions[tags[:, 1:], tags[:, :-1]].view(n_batchs,-1)
        # last transition score to End tag
        # find last tag of each batch, get last tag to End trans
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)
        last_score = self.transitions[END_TAG_IDX, last_tag]
        # why the score dont cal log_sum_exp
        
        score = ((trans + emits) * masks).sum(1) + last_score
        return score
    def __vetebi_decode(self, feats, masks):
        # decode path of max score
        # feats:[B, S, T]
        n_batchs, seq_len, tag_s = feats.size()
        #init forward_scores, pre_save_matrix
        forward_scores=torch.fill((n_batchs, self.tag_size),-10000.)
        forward_scores[:, START_TAG_IDX] = 0.
        pre_save_matrix=torch.fill()
        trans = self.crf_trans.unsqueeze(0)
        for i in range(seq_len):
            # for each position 
            # emits from feats matrix: [B, T]
            emit_scores = feats[:, i]
            # forward matrix is [B, T], unsqueeze to [B, 1, T]
            # emits dont influence arg max, so add it at last
            t_scores = forward_scores.unsqueeze(1) + trans
            # get max scores and max arg
            t_scores, premax_line = t_scores.max(2)
            t_scores += emit_scores
            m = masks[:, i].unsqueeze(1)
            forward_scores = forward_scores * (1 - m) + t_scores * m
            pre_save_matrix.append(premax_line)
        # transfrom to End tag
        forward_scores = forward_scores + self.crf_trans[END_TAG_IDX]
        # transpose pre save matrix [S, B, T] -> [B, S, T]
        pre_save_matrix = torch.Tensor(pre_save_matrix).transpose(0, 1)
        # get max scores and end tag
        max_scores, next_tag = forward_scores.max(-1)
        # fill all batch tag seq of max score
        max_tag_seq = torch.zeros((n_batchs, seq_len))
        end_tag = torch.fill((n_batchs), END_TAG_IDX)
        max_tag_seq[:, -1] = next_tag
        
        for i in range(seq_len-1, 0, -1):
            # this position mask: [B]
            m = masks[:, i]
            # if mask is 0, tag_seq = End tag, else is next tag
            max_tag_seq[:, i] = m * next_tag + (1- m) * end_tag
            # refresh next tag, if i mask is 0, use pre tag; else get new tag
            next_tag = pre_save_matrix[:, i].gather(1, next_tag.unsqueeze(1)).squeeze() * m + (1 - m) * next_tag
            next_tag = next_tag.squeeze()
        max_tag_seq[:, 0] = next_tag
        return max_scores, max_tag_seq

    def loss(self, feats, tags, masks):
        # negative log likelihood loss
        # calculate total scores and obsvered tag sequence score
        total_scores = self._forward_alg(feats, masks)
        tag_score = self._score_sentence(feats, tags.long(), masks)
        loss = (tag_score - total_scores).mean()
        return loss
    
    def forward(self, feats, masks):
        # get max scores and paths
        return self.__vetebi_decode(feats, masks)

