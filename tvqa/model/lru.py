from .transformer import GRUCell, Linear, GRU, BertLayer, BertConfig
from .tvqa_abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

class LRU(ABC):
    def __init__(self, opt):
        super().__init__(opt)
        config = BertConfig(0, hidden_size=opt.hidden_size,
                               intermediate_size=opt.hidden_size * 4,
                               num_attention_heads=opt.num_attention_heads)
        self.sent_gru = GRU(input_size=opt.embedding_size,
                            hidden_size=config.hidden_size,
                            batch_first=True)
        self.q_gru = GRU(input_size=opt.embedding_size,
                           hidden_size=config.hidden_size,
                           batch_first=True)
        self.sent_linear = Linear(in_features=opt.vid_feat_size,
                                  out_features=config.hidden_size)
        self.q_linear = Linear(in_features=opt.vid_feat_size,
                               out_features=config.hidden_size)

        self.gamma_linear = Linear(in_features=config.hidden_size,
                                   out_features=1)
        self.gamma_sigmoid = nn.Sigmoid()


    def sub_embedding(self, e):
        e_sub = self.embedding(e)
        _, hid = self.sent_gru(e_sub.unsqueeze(0))
        return hid.squeeze(0)

    def q_embedding(self, e):
        e_sub = self.embedding(e)
        _, hid = self.q_gru(e_sub.unsqueeze(0))
        return hid.squeeze(0)

    # Using given embeddings, compute logit and value
    def mem_forward(self, vid_feature, sub_feature, v_tm1, attention_mask):
        memory_feature = self.sent_linear(vid_feature[:, :-1])
        memory_feature = memory_feature + sub_feature[:, :-1]

        q_feature = self.q_linear(vid_feature[:, -1])
        q_feature = q_feature + sub_feature[:, -1]
        q_feature = q_feature.unsqueeze(0)

        lru_gamma = self.gamma_linear(q_feature)
        lru_gamma = self.gamma_sigmoid(lru_gamma.squeeze(2))

        batch_size, memory_num, hidden_dim = memory_feature.size()

        z_t = F.softmax(torch.bmm(memory_feature, q_feature.permute(0, 2, 1)).squeeze(-1), 1)

        v_tm1 = v_tm1[:, :-1]
        v_t = 0.1 * v_tm1 + 0.9 * z_t

        g = z_t - lru_gamma * v_tm1
        V = g.data.new(1, 1).fill_(0.0)
        # Logit, Value computing part
        return g, V, v_t
