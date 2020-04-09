import torch
import torch.nn as nn
from .mlp import MLP

import torch.nn.functional as F
import math
from .rnn import mean_along_time

class BaselineMLP(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        V, D = vocab.shape
        self.embedding = nn.Embedding(V, D)
        # self.embedding = nn.Embedding(V, D)
        # self.load_embedding(vocab)

        device = args.device
        self.device = device
        self.mlp = nn.ModuleList([MLP(2 * D, 1, 50, 2) for i in range(5)])
        # self.mlp = nn.ModuleList([
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2),
        #     MLP(2 * D, 1, 50, 2)
        # ])
        self.to(device)

    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        # self.embedding.weight.data.copy_(pretrained_embedding)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab)

    def forward(self, que, answers, **features):
        batch_size = que.size(0)

        q = que
        a = answers
        ql = features['que_len']
        al = features['ans_len']

        q = self.embedding(q)
        a = self.embedding(a)

        q = mean_along_time(q,  ql)
        a = torch.stack([mean_along_time(a[:, i], al[:, i]) for i in range(5)])
        # a = a.transpose(0, 1)
        
        q = q.unsqueeze(0).repeat(5, 1, 1)
        qa = torch.cat([q, a], dim=2)

        out = torch.zeros(batch_size, 5).to(self.device)
        for i in range(5):
            out[:, i] = self.mlp[i](qa[i]).squeeze(1)

        return out
