__author__ = "Jie Lei"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class BidafAttnLayer(nn.Module):
    def __init__(self, n_dim, d_model, dropout = 0.1):
        super(BidafAttnLayer,self).__init__()

        self.bidaf = BidafAttn(n_dim * 3, method="dot")
        self.ffn = FFN(d_model, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = Norm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = Norm(d_model)

    def forward(self, s1, l1, s2, l2):
        att_v = s1
        mask,_ = self.len_to_mask(l1, l1.max())
        
        for i in range(2):
            att_v = self.norm1(att_v+self.dropout1(self.bidaf(att_v, l1, s2, l2)))        
            att_v = self.norm2(att_v+self.dropout2(self.ffn(att_v)))

            #att_v.masked_fill_(mask.unsqueeze(2).repeat(1,1,att_v.shape[-1]), 0)

        return att_v

    def len_to_mask(self, lengths, len_max):
        #len_max = lengths.max().item()


        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)

        return mask, len_max

class BidafAttn(nn.Module):
    """from the BiDAF paper https://arxiv.org/abs/1611.01603.
    Implemented by @easonnie and @jayleicn
    """
    def __init__(self, channel_size, method="original", get_h=False):
        super(BidafAttn, self).__init__()
        """
        This method do biDaf from s2 to s1:
            The return value will have the same size as s1.
        :param channel_size: Hidden size of the input
        """
        self.method = method
        self.get_h = get_h
        if method == "original":
            self.mlp = nn.Linear(channel_size * 3, 1, bias=False)

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        if self.method == "original":
            t1 = s1.size(1)
            t2 = s2.size(1)
            repeat_s1 = s1.unsqueeze(2).repeat(1, 1, t2, 1)  # [B, T1, T2, D]
            repeat_s2 = s2.unsqueeze(1).repeat(1, t1, 1, 1)  # [B, T1, T2, D]
            packed_s1_s2 = torch.cat([repeat_s1, repeat_s2, repeat_s1 * repeat_s2], dim=3)  # [B, T1, T2, D*3]
            s = self.mlp(packed_s1_s2).squeeze()  # s is the similarity matrix from biDAF paper. [B, T1, T2]
        elif self.method == "dot":
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
        """
        attended vectors of s2 for each word in s1,
        signify which words in s2 are most relevant to words in s1
        """
        a_weight = F.softmax(s, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        u_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        return u_tile

    @classmethod
    def get_h_tile(cls, s, s1):
        """
        attended vectors of s1
        which words in s1 is most similar to each words in s2
        """
        t1 = s1.size(1)
        b_weight = F.softmax(torch.max(s, dim=2)[0], dim=-1).unsqueeze(1)  # [b, t2]
        h_tile = torch.bmm(b_weight, s1).repeat(1, t1, 1)  # repeat to match s1 # [B, t1, D]
        return h_tile

    def forward(self, s1, l1, s2, l2):
        s = self.similarity(s1, l1, s2, l2)
        u_tile = self.get_u_tile(s, s2)
        # h_tile = self.get_h_tile(s, s1)
        h_tile = self.get_h_tile(s, s1) if self.get_h else None
        #return u_tile, h_tile
        return u_tile


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class FFN(nn.Module):
    def __init__(self, hidden_size, ff_size, dropout_r=0.1):
        super(FFN, self).__init__()

        self.fc = FC(hidden_size, ff_size, dropout_r=dropout_r, use_relu=True)
        self.linear = nn.Linear(ff_size, hidden_size)

    def forward(self, x):
        return self.linear(self.fc(x))



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

