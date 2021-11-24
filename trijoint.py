import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from args import get_parser
from torch.autograd import Variable
import sys

# =============================================================================
parser = get_parser()
opts = parser.parse_args()


# # ======================================================================

def mode_product(tensor, matrix_1, matrix_2, matrix_3, matrix_4, n_way=3):
    # mode-1 tensor product
    tensor_1 = tensor.transpose(3, 2).contiguous().view(tensor.size(0), tensor.size(1),
                                                        tensor.size(2) * tensor.size(3) * tensor.size(4))
    tensor_product = torch.matmul(matrix_1, tensor_1)
    tensor_1 = tensor_product.view(-1, tensor_product.size(1), tensor.size(4), tensor.size(3),
                                   tensor.size(2)).transpose(4, 2)

    # mode-2 tensor product
    tensor_2 = tensor_1.transpose(2, 1).transpose(4, 2).contiguous().view(-1, tensor_1.size(2),
                                                                          tensor_1.size(1) * tensor_1.size(
                                                                              3) * tensor_1.size(4))
    tensor_product = torch.matmul(matrix_2, tensor_2.float())
    tensor_2 = tensor_product.view(-1, tensor_product.size(1), tensor_1.size(4), tensor_1.size(3),
                                   tensor_1.size(1)).transpose(4, 1).transpose(4, 2)
    tensor_product = tensor_2
    if n_way > 2:
        # mode-3 tensor product
        tensor_3 = tensor_2.transpose(3, 1).transpose(4, 2).transpose(4, 3).contiguous().view(-1, tensor_2.size(3),
                                                                                              tensor_2.size(
                                                                                                  2) * tensor_2.size(
                                                                                                  1) * tensor_2.size(4))
        tensor_product = torch.matmul(matrix_3, tensor_3.float())
        tensor_3 = tensor_product.view(-1, tensor_product.size(1), tensor_2.size(4), tensor_2.size(2),
                                       tensor_2.size(1)).transpose(1, 4).transpose(4, 2).transpose(3, 2)
        tensor_product = tensor_3
    if n_way > 3:
        # mode-4 tensor product
        tensor_4 = tensor_3.transpose(4, 1).transpose(3, 2).contiguous().view(-1, tensor_3.size(4),
                                                                              tensor_3.size(3) * tensor_3.size(
                                                                                  2) * tensor_3.size(1))
        tensor_product = torch.matmul(matrix_4, tensor_4)
        tensor_4 = tensor_product.view(-1, tensor_product.size(1), tensor_3.size(3), tensor_3.size(2),
                                       tensor_3.size(1)).transpose(4, 1).transpose(3, 2)
        tensor_product = tensor_4

    return tensor_product

def d_norm(input, dim=-1, p=2, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)


# Skip-thoughts LSTM
class StRNN(nn.Module):
    def __init__(self):
        super(StRNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=opts.stDim,
            hidden_size=opts.srnnDim,
            bidirectional=False,
            batch_first=True)

    def forward(self, x, sq_lengths):
        # here we use a previous LSTM to get the representation of each instruction
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx \
            .view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
        # we get the last index of each sequence in the batch
        idx = (sq_lengths - 1).view(-1,
                                    1).expand(unpacked.size(0),
                                              unpacked.size(2)).unsqueeze(1)
        # we sort and get the last element of each sequence
        output = unpacked.gather(0, unsorted_idx.long()).gather(1, idx.long())
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output

class IngRNN(nn.Module):
    def __init__(self):
        super(IngRNN, self).__init__()
        self.irnn = nn.LSTM(
            input_size=opts.ingrW2VDim,
            hidden_size=opts.irnnDim,
            bidirectional=True,
            batch_first=True)
        _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        self.embs = nn.Embedding(
            vec.size(0),
            opts.ingrW2VDim,
            padding_idx=0)  # not sure about the padding idx
        self.embs.weight.data.copy_(vec)

    def forward(self, x, sq_lengths):
        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx \
            .view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1, -1, 1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd
        # dimension
        output = hidden[0].gather(1, unsorted_idx).transpose(0, 1).contiguous()
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims, act='Tanh', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class TCNet(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, act='Tanh', dropout=[.2, .5], k=1):
        super(TCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim  # 1024
        self.a_dim = a_dim  # 1024
        self.h_out = h_out  # 1
        self.rank = rank  # 32
        self.h_dim = h_dim * k  # 512 * 1
        self.hv_dim = int(h_dim / rank)  # Tm的Wzv
        self.hq_dim = int(h_dim / rank)
        self.ha_dim = int(h_dim / rank)

        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[1])
        self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
        # only do paralind decomposition while h_dim < 1024
        if self.h_dim < 1024:
            self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
            self.v_net = nn.ModuleList(
                [FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[1]) for _ in range(rank)])
            self.q_net = nn.ModuleList(
                [FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            self.a_net = nn.ModuleList(
                [FCNet([self.h_dim, self.ha_dim], act=act, dropout=dropout[0]) for _ in range(rank)])

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(
                torch.Tensor(1, rank, self.hv_dim, self.hq_dim, self.ha_dim, glimpse, h_out).normal_())
        self.dropout = nn.Dropout(dropout[1])

    def forward(self, v, q, a):
        f_emb = 0
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        a_tucker = self.a_tucker(a)
        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)
            a_ = self.a_net[r](a_tucker)
            f_emb = f_emb + mode_product(self.T_g[:, r, :, :, :, :, :], v_, q_, a_, None)

        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q, a, w):
        # v: [batch, num_objs, obj_dim]
        # q, a: [batch, q_len, q_dim]
        # w: b x v x q x a
        v_ = self.v_tucker(v).transpose(2, 1)  # b x v_dim x num_v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3)  # b x q_dim x q_len x 1
        a_ = self.a_tucker(a).transpose(2, 1).unsqueeze(3)  # b x d x a

        logits = torch.einsum('bdv,bvqa,bdqi,bdaj->bdij', [v_, w, q_, a_])
        logits = logits.squeeze(3).squeeze(2)
        return logits


class TriAttention(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, k, dropout=[.2, .5]):
        super(TriAttention, self).__init__()
        self.glimpse = glimpse
        self.TriAtt = TCNet(v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, dropout=dropout, k=k)

    def forward(self, v, q, a):
        v_num = v.size(1)
        q_num = q.size(1)
        a_num = a.size(1)
        logits = self.TriAtt(v, q, a)  # T_m
        # mask = (0 == v.abs().sum(2)).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(logits.size())
        mask = (0 == v.abs().sum(2)).unsqueeze(2).unsqueeze(3).expand(logits.size())
        logits.data.masked_fill_(mask.data, -float('inf'))

        p = torch.softmax(logits.contiguous().view(-1, v_num * q_num * a_num, self.glimpse), 1)
        return p.view(-1, v_num, q_num, a_num, self.glimpse), logits


class Embedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Embedding, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x


class ImgExtractor(nn.Module):
    def __init__(self):
        super(ImgExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.visionMLP = nn.Sequential(*modules)

    def forward(self, x):
        x = self.visionMLP(x)
        return x  # [batch_size, 2048, 1, 1]


class ImgRegionExtractor(nn.Module):
    def __init__(self):
        super(ImgRegionExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visionMLP = nn.Sequential(*modules)
        # self.avgpool = nn.AdaptiveAvgPool2d(2)

    def forward(self, x):
        x = self.visionMLP(x)  # ([batch_size, 2048, 7, 7)]
        # x = self.avgpool(x)
        return x


# Im2recipe model

class im2recipe(nn.Module):
    def __init__(self):
        super(im2recipe, self).__init__()

        self.batch_size = opts.batch_size
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.visionMLP = nn.Sequential(*modules)
        # self.visionMLP = ImgRegionExtractor()
        # self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.stGRU_ = StGRU()
        self.stRNN_ = StRNN()
        self.ingRNN_ = IngRNN()
        # self.ingGRU_ = IngGRU()

        self.inst_embedding = Embedding(opts.srnnDim, opts.embDim)
        self.ing_embedding = Embedding(1024, opts.embDim)
        self.v_embedding = Embedding(opts.imfeatDim, opts.embDim)

        self.dropout = [.2, 0.5]
        self.v_att= TriAttention(opts.embDim, opts.embDim, opts.embDim, opts.hDim, opts.hOut, opts.rank, opts.glimpse, k=1, dropout=self.dropout)
        self.t_net = TCNet(opts.embDim, opts.embDim, opts.embDim, opts.hDim, opts.hOut, opts.rank,  opts.glimpse, k=2, dropout=self.dropout)
        print('dropout:', self.dropout)
        self.inst_prj = FCNet([opts.numHid, opts.numHid], '', .2)
        self.ing_prj = FCNet([opts.numHid, opts.numHid], '', .2)

        self.linear_sim = nn.Linear(opts.numHid, 1)

    def _similarity(self, x):
        # x: [batch_size, batch_size, numHid]
        x = self.linear_sim(x)
        x = x.view(x.size(0), x.size(1))
        x = torch.sigmoid(x)

        return x

    def forward(self, x, y1, y2, z1, z2, hard=False, hard_index=None):
        if hard:
            hard_index = torch.tensor(hard_index).cuda()
            x = torch.index_select(x, 0, hard_index)
            y1 = torch.index_select(y1, 0, hard_index)
            y2 = torch.index_select(y2, 0, hard_index)
            z1 = torch.index_select(z1, 0, hard_index)
            z2 = torch.index_select(z2, 0, hard_index)
        # input_size=[(bs, 3, 224, 224),(bs, 20, 1024),(bs), (bs,20), (bs)])
        inst_emb = self.stRNN_(y1, y2)  # [batch_size, srnnDim]
        # inst_emb = self.stGRU_(y1, y2) # [batch_size, 20, 1024]
        inst_emb = self.inst_embedding(inst_emb)
        inst_emb = d_norm(inst_emb, dim=-1)
        inst_emb = inst_emb.unsqueeze(1)  # [batch_size,1,1024]

        #ing_emb = self.ingRNN_(z1, z2)  # [batch_size, irnnDim*2]
        ing_emb = self.ingGRU_(z1, z2)  # [batch_size, 20, 1024]
        ing_emb = self.ing_embedding(ing_emb)
        ing_emb = d_norm(ing_emb, dim=-1)
        #ing_emb = ing_emb.unsqueeze(1)  # [batch_size,1,1024]

        # no region
        v_emb = self.visionMLP(x)  # [batch_size, imgfeatDim]
        v_emb = v_emb.view(v_emb.size(0), -1)
        v_emb = self.v_embedding(v_emb)
        v_emb = d_norm(v_emb, dim=-1)
        v_emb = v_emb.unsqueeze(1)  # [batch_size,1,1024]

        z_emb = []
        att, logits = self.v_att(v_emb, inst_emb, ing_emb)
        for b in range(v_emb.size(0)):
            v_b_emb = v_emb[b].expand(v_emb.size())
            # torch.Size([batch_size, 1, 1, 20, 1])
            b_emb = self.t_net.forward_with_weights(v_b_emb, inst_emb, ing_emb,att[:, :, :, :, 0])
            z_emb.append(b_emb.sum(1)) # [batch_size, h_dim]

        z_emb = torch.stack(z_emb)  # z_emb: [batch_size, batch_size, h_dim]
        score = self._similarity(z_emb)
        return score