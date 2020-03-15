import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
#from torch_geometric.nn import GCNConv

import math
from . rnn import RNNEncoder, max_along_time
from . bidaf import BidafAttn, BidafAttnLayer
'''
multichoice version
baseline model:
2-layer single-directional encoder-decoder GRU
fusion with linear layer
'''


class MultiChoice(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()

        self.text_feature_names = args.text_feature_names
        self.feature_names = args.use_inputs

        self.vocab = vocab
        V = len(vocab)
        D = n_dim

        self.embedding = nn.Embedding(V, D)
        n_dim = args.n_dim
        image_dim = args.image_dim


        self.bidaf = BidafAttn(n_dim * 3, method="dot")  # no parameter for dot
        #self.bidaf = BidafAttnLayer(n_dim, n_dim)
        self.lstm_raw = RNNEncoder(300, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

        self.lstm_script = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
        self.script_on = "script" in args.stream_type
        self.vbb_on = "visual_bb" in args.stream_type
        self.vmeta_on = "visual_meta" in args.stream_type
        self.conv_pool = conv1d(n_dim*4+1, n_dim*2)

        self.character = nn.Parameter(torch.randn(22, D, device=args.device, dtype=torch.float), requires_grad=True)
        self.norm1 = Norm(D)

        if self.script_on:
            self.lstm_sub = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")
            self.lstm_mature_sub = RNNEncoder(n_dim  * 5, n_dim, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_sub = nn.Sequential(MLP(n_dim*2, 1, 500, 1), nn.Softmax(dim=1))
            self.mhattn_sub = MHAttnLayer(4, D, D)

        if self.vmeta_on:            
            self.lstm_vmeta = RNNEncoder(321, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

            self.lstm_mature_vmeta = RNNEncoder(n_dim  * 5, n_dim, bidirectional=True,
                                               dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vmeta = nn.Sequential(MLP(n_dim*2, 1, 500, 1), nn.Softmax(dim=1))
            self.mhattn_vmeta = MHAttnLayer(4, D, D)

        if self.vbb_on:
            self.lstm_vbb = RNNEncoder(image_dim+21, 150, bidirectional=True, dropout_p=0, n_layers=1, rnn_type="lstm")

            self.vbb_fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(image_dim, n_dim),
                nn.Tanh(),
            )
            self.lstm_mature_vbb = RNNEncoder(n_dim * 2 * 5, n_dim, bidirectional=True,
                                              dropout_p=0, n_layers=1, rnn_type="lstm")
            self.classifier_vbb = nn.Sequential(MLP(n_dim*2, 1, 500, 1), nn.Softmax(dim=1))

            self.mhattn_vbb = MHAttnLayer(4, D, D)

        self.q_attn_mlp = (MLP(D, D, D, 1))
        self.q_attn_conv = conv1d(D, D)
        self.q_attn_mlp2 = (MLP(D, 3, D, 1))
        self.q_attn_softmax = nn.Softmax(dim=1)


    def _to_one_hot(self, y, n_dims, mask, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).type(dtype)
        out = zeros.scatter(scatter_dim, y_tensor, 1)

        out_mask,_ = self.len_to_mask(mask, out.shape[1])
        out_mask = out_mask.unsqueeze(2).repeat(1, 1, n_dims)

        return out.masked_fill_(out_mask, 0)


    def load_embedding(self, pretrained_embedding):
        print('Load pretrained embedding ...')
        self.embedding.weight.data.copy_(pretrained_embedding)
        #self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))

    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def process_feature(self, q, name, feature):
        if name in self.text_feature_names:
            feature = self.embedding(feature)

        feature = self.feature_encoders[name](feature)
        feature = feature.mean(dim=1)
        q = self.feature_fusers[name](q, feature)
        return q


    def len_to_mask(self, lengths, len_max):
        #len_max = lengths.max().item()


        mask = torch.arange(len_max, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), len_max) >= lengths.unsqueeze(1)
        mask = torch.as_tensor(mask, dtype=torch.uint8, device=lengths.device)

        return mask, len_max

    def forward(self, que, answers, **features):
        '''
        filtered_sub (B, max_sub_len)
        filtered_sub_len (B)
        filtered_speaker (B, max_sub_len)

        filtered_visual (B, max_v_len*3)
        filtered_visual_len (B)

        filtered_image (B, max_v_len, 512)
        filtered_image_len (12)

        que (B, max_que_len)
        que_len (B)

        answers (B, 5, max_ans_len)
        ans_len (B, 5)
        
        adjacency (B, 21, 21)
        print(que.shape)
        print(answers.shape)
        for key, value in features.items():
            print(key, value.shape)
            

        '''

        B = que.shape[0]

        # -------------------------------- #
        e_q = self.embedding(que)
        q_len = features['que_len']
        e_q, _ = self.lstm_raw(e_q, q_len)

        # -------------------------------- #
        e_ans = self.embedding(answers).transpose(0, 1)
        ans_len = features['ans_len'].transpose(0, 1)
        e_ans_list = [self.lstm_raw(e_a, ans_len[idx])[0] for idx, e_a in enumerate(e_ans)]

        #e_ans = torch.stack([self.lstm_raw(e_a, ans_len[:,idx]) ], dim=1)
        
        
        #print(self.get_name(que, q_len))
        concat_qa = [(self.get_name(que, q_len) + self.get_name(answers.transpose(0,1)[i], ans_len[i])).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [(torch.sum(concat_qa[i], dim=1) == 0).unsqueeze(1).type(torch.cuda.FloatTensor) for i in range(5)]
        concat_qa_none = [torch.cat([concat_qa[i], concat_qa_none[i]], dim=1) for i in range(5)]
        qa_character = [torch.matmul(concat_qa_none[i], self.character) for i in range(5)]
        qa_character = [self.norm1(qa_character[i]) for i in range(5)]

        # A = features['adjacency']
        #A_ = self.gcn(self.character, range(21), A)

        if self.script_on:
            e_s = self.embedding(features['filtered_sub'])
            #print(e_s.shape)
            s_len = features['filtered_sub_len']

            # -------------------------------- #
            spk = features['filtered_speaker']
            spk_onehot = self._to_one_hot(spk, 21, mask=s_len)
            e_s = torch.cat([e_s, spk_onehot], dim=2)

            spk_flag = [torch.matmul(spk_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            spk_flag = [(spk_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            # -------------------------------- #
            raw_out_sub, _ = self.lstm_sub(e_s, s_len)
            #raw_out_sub, _ = self.lstm_raw(e_s, s_len)
            # -------------------------------- #

            if torch.sum(q_len < 2) > 0:
                print(que)

            s_out = self.stream_processor(self.classifier_sub,self.mhattn_sub, spk_flag, raw_out_sub, s_len, qa_character, e_q, q_len, e_ans_list, ans_len)
        else:
            s_out = 0

        if self.vmeta_on:
            vmeta = features['filtered_visual'].view(B, -1, 3)
            #print(vmeta.shape)
            vmeta_len = features['filtered_visual_len']*2/3
            vp = vmeta[:,:,0]
            vp = vp.unsqueeze(2).repeat(1,1,2).view(B, -1)
            vbe = vmeta[:,:,1:3].contiguous()
            vbe = vbe.view(B, -1)
            e_vbe = self.embedding(vbe)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vmeta_len)
            e_vbe = torch.cat([e_vbe, vp_onehot], dim=2)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            # -------------------------------- #
            raw_out_vmeta, _ = self.lstm_vmeta(e_vbe, vmeta_len)
            #raw_out_vmeta, _ = self.lstm_raw(e_vbe, vmeta_len)
            m_out = self.stream_processor(self.classifier_vmeta, self.mhattn_vmeta, vp_flag, raw_out_vmeta, vmeta_len, qa_character, e_q, q_len, e_ans_list, ans_len)
        else:
            m_out = 0

        if self.vbb_on:
            e_vbb = features['filtered_person_full']
            vbb_len = features['filtered_person_full_len']

            vp = features['filtered_visual'].view(B, -1, 3)[:,:,0]
            vp = vp.unsqueeze(2).view(B, -1)
            # -------------------------------- #
            vp_onehot = self._to_one_hot(vp, 21, mask=vbb_len)
            #e_vbb =self.vbb_fc(vbb)
            e_vbb = torch.cat([e_vbb, vp_onehot], dim=2)
            vp_flag = [torch.matmul(vp_onehot, concat_qa[i].unsqueeze(2)) for i in range(5)]
            vp_flag = [(vp_flag[i] > 0).type(torch.cuda.FloatTensor) for i in range(5)]
            # -------------------------------- #
            raw_out_vbb, _ = self.lstm_vbb(e_vbb, vbb_len)
            #raw_out_vbb, _ = self.lstm_raw(e_vbb, vbb_len)
            b_out = self.stream_processor(self.classifier_vbb, self.mhattn_vbb, vp_flag, raw_out_vbb, vbb_len, qa_character, e_q, q_len, e_ans_list, ans_len)

        else:
            b_out = 0

        '''
        q_attn = self.q_attn_conv(self.q_attn_mlp(e_q), q_len)
        q_attn = self.q_attn_softmax(self.q_attn_mlp2(q_attn))
        q_attn = q_attn.unsqueeze(2).repeat(1,1,5).unsqueeze(3)
        
        out = q_attn[:,0]*s_out + q_attn[:,1]*m_out + q_attn[:,2]*b_out
        '''
        out = s_out + m_out + b_out 

        return out.squeeze()


        return o
       
    def stream_processor(self, classifier, mhattn, ctx_flag, ctx, ctx_l,
                         qa_character, q_embed, q_l, a_embed, a_l):

        u_q = self.bidaf(ctx, ctx_l, q_embed, q_l)
        u_a = [self.bidaf(ctx, ctx_l, a_embed[i], a_l[i]) for i in range(5)]
        u_ch = [mhattn(qa_character[i], ctx, ctx_l) for i in range(5)]

        concat_a = [torch.cat([ctx, u_a[i], u_q, ctx_flag[i], u_ch[i]], dim=-1) for i in range(5)]
        maxout = [self.conv_pool(concat_a[i], ctx_l) for i in range(5)]

        answers = torch.stack(maxout, dim=1)
        out = classifier(answers)  # (B, 5)

        return out 

    def get_name(self, x, x_l):
        x_mask = x.masked_fill(x>20, 21)
        x_onehot = self._to_one_hot(x_mask, 22, x_l)
        x_sum = torch.sum(x_onehot[:,:,:21], dim=1)
        return x_sum > 0



class conv1d(nn.Module):
    def __init__(self, n_dim, out_dim):
        super().__init__()
        out_dim = int(out_dim/4)
        self.conv_k1 = nn.Conv1d(n_dim, out_dim, kernel_size=1, stride=1)
        self.conv_k2 = nn.Conv1d(n_dim, out_dim, kernel_size=2, stride=1)
        self.conv_k3 = nn.Conv1d(n_dim, out_dim, kernel_size=3, stride=1)
        self.conv_k4 = nn.Conv1d(n_dim, out_dim, kernel_size=4, stride=1)
        #self.maxpool = nn.MaxPool1d(kernel_size = )

    def forward(self, x, x_l):
        # x : (B, T, 5*D)
        x_pad = torch.zeros(x.shape[0],3,x.shape[2]).type(torch.cuda.FloatTensor)
        x = torch.cat([x, x_pad], dim=1)
        x1 = F.relu(self.conv_k1(x.transpose(1,2)))[:,:,:-3]
        x2 = F.relu(self.conv_k2(x.transpose(1,2)))[:,:,:-2]
        x3 = F.relu(self.conv_k3(x.transpose(1,2)))[:,:,:-1]
        x4 = F.relu(self.conv_k4(x.transpose(1,2)))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = out.transpose(1,2)
        return max_along_time(out, x_l)


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


class ConvEncoder(nn.Module):
    def __init__(self, n_dim, max_len):
        super().__init__()

        self.conv1d = nn.Conv1d(n_dim, n_dim, 1)
        self.maxpool = nn.MaxPool1d(max_len)

    def forward(self, x):
        return self.maxpool(self.conv1d(x))


class Encoder(nn.Module):
    def __init__(self, n_dim, layers, dropout=0):
        super().__init__()

        self.rnn = nn.GRU(n_dim, n_dim, layers, batch_first=True, dropout=dropout)

    def run(self, x):
        output, hn = self.rnn(x)
        hn = hn.transpose(0, 1)
        return output, hn

    def forward(self, x):
        return self.run(x)[1]

class LinearFuser(nn.Module): 
    def __init__(self, in_dim1, in_dim2, out_dim):
        super().__init__()

        self.linear1 = nn.Linear(in_dim1, out_dim, bias=False)
        self.linear2 = nn.Linear(in_dim2, out_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x1, x2):
        out = self.linear1(x1) + self.linear2(x2) 
        out = self.act(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hsz, n_layers):
        super(MLP, self).__init__()

        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hsz),
                    nn.ReLU(True),
                    nn.Dropout(0.5)
                ])
                prev_dim = hsz

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MHAttnLayer(nn.Module):
    def __init__(self, heads, hidden, d_model, dropout = 0.1):
        super(MHAttnLayer,self).__init__()

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