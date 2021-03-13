import torch
import torch.nn as nn
from utils import *


class CRF(nn.Module):
    def __init__(self, tag_size):
        super(CRF, self).__init__()
        #set the Parameters
        self.tag_size = tag_size+2
        self.start_tag = self.tag_size - 2
        self.end_tag = self.tag_size - 1
        #crf transform matrix, trans[i, j] means transform score from j to i
        self.crf_trans=nn.Parameter(torch.randn(self.tag_size, self.tag_size))
        self.crf_trans.data[self.start_tag, :] = -10000.
        self.crf_trans.data[:, self.end_tag] = -10000.

    #forward calculate the all tags partition
    #input feats is each tags PR of sentence
    def _forward_alg(self, feats, masks):
        # get all paths scores sum
        # feats:[B, S, T]
        n_batchs, seq_len, tag_s = feats.size()
        #init the partition vector set the start_pos to 1.
        forward_scores = torch.fill((n_batchs, self.tag_size), -10000.)
        forward_scores[:, self.start_tag] = 0.
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

        return log_sum_exp(forward_scores + self.crf_trans[self.end_tag])
    def _score_sentence(self, feats, tags, masks):
        # get score of given path
        # sentence: [B, S, T]
        # cal sentences score
        n_batchs, seq_len, tag_s = feats.size()
        # select tags emits in feats
        emits = feats.gather(2, tags).view(n_batchs, -1)
        start_tag = torch.full((n_batchs, 1), self.start_tag, dtype=torch.long)
        # select tags trans scores in crf_trans
        tags = torch.cat([start_tag, tags], dim=1)
        trans = self.transitions[tags[:, 1:], tags[:, :-1]].view(n_batchs,-1)
        # last transition score to End tag
        # find last tag of each batch, get last tag to End trans
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)
        last_score = self.transitions[self.end_tag, last_tag]
        # why the score dont cal log_sum_exp
        
        score = ((trans + emits) * masks).sum(1) + last_score
        return score
    def __vetebi_decode(self, feats, masks):
        # decode path of max score
        # feats:[B, S, T]
        n_batchs, seq_len, tag_s = feats.size()
        #init forward_scores, pre_save_matrix
        forward_scores=torch.fill((n_batchs, self.tag_size),-10000.)
        forward_scores[:, self.start_tag] = 0.
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
        forward_scores = forward_scores + self.crf_trans[self.end_tag]
        # transpose pre save matrix [S, B, T] -> [B, S, T]
        pre_save_matrix = torch.Tensor(pre_save_matrix).transpose(0, 1)
        # get max scores and end tag
        max_scores, next_tag = forward_scores.max(-1)
        # fill all batch tag seq of max score
        max_tag_seq = torch.zeros((n_batchs, seq_len))
        end_tag = torch.fill((n_batchs), self.end_tag)
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

