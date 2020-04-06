import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from einops import rearrange



class ContextMatching(nn.Module):
    def __init__(self, channel_size):
        super(ContextMatching, self).__init__()

    def similarity(self, s1, l1, s2, l2):
        s = torch.bmm(s1, s2.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).byte()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0

        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data.byte(), -float("inf"))
        return s

    @classmethod
    def get_u_tile(cls, s, s2):
        a_weight = F.softmax(s, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        u_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        return u_tile

    def forward(self, s1, l1, s2, l2):
        s = self.similarity(s1, l1, s2, l2)
        u_tile = self.get_u_tile(s, s2)
        return u_tile

class CharMatching(nn.Module):
    def __init__(self, heads, hidden, d_model, dropout = 0.1):
        super(CharMatching,self).__init__()

        self.mhatt = MHAttn(heads, hidden, d_model, dropout)
        self.ffn = FFN(d_model, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = Norm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = Norm(d_model)

    def forward(self, q, kv, mask_len):
        att_v = kv
        mask,_ = self.len_to_mask(mask_len, mask_len.max())
        for i in range(1):
            att_v = self.norm1(att_v+self.dropout1(self.mhatt(q, kv, kv, mask))) 

            att_v.masked_fill_(mask.unsqueeze(2).repeat(1,1,att_v.shape[-1]), 0)
        '''
        att_v = []
        for i in range(seq_len):
            att_v_i = self.norm1(q + self.dropout1(self.mhatt(kv[:,i,:], kv[:,i,:], q)))
            att_v.append(att_v_i.unsqueeze(1)) # batch, 1, dim
        att_v = torch.cat(att_v, dim=1) # batch, len, dim

        att_v = self.norm2(att_v + self.dropout2(
            self.ffn(att_v)
        ))
        '''
        return att_v

    def len_to_mask(self, lengths, len_max):
        #len_max = lengths.max().item()


        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)

        return mask, len_max


class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r=0.1):
        super(FFN, self).__init__()

        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)
        self.linear2 = nn.Linear(ff_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return self.linear2(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
          # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class MHAttn(nn.Module):
    def __init__(self, heads, hidden, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = int(hidden/heads)
        self.h = heads

        self.q_linear = nn.Linear(d_model, hidden)
        self.v_linear = nn.Linear(d_model, hidden)
        self.k_linear = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_k*self.h)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1,self.h,1,1)
            scores = scores.masked_fill_(mask, -float("inf"))

        scores = F.softmax(scores, dim=-1)

        scores = scores.transpose(-2, -1).repeat(1,1,1,self.d_k)

        if dropout is not None:
            scores = dropout(scores)

        #output = torch.matmul(scores, v)
        output = scores * v

        return output

