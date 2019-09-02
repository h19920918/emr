from .transformer import GRUCell, Linear, GRU, BertLayer, BertConfig
from .tvqa_abc import ABC
import torch
import torch.nn as nn

class T_EMR(ABC):
    def __init__(self, opt):
        super().__init__(opt)
        config = BertConfig(0, hidden_size=opt.hidden_size,
                               intermediate_size=opt.hidden_size * 4,
                               num_attention_heads=opt.num_attention_heads,
                               demo=opt.demo)
        self.sent_gru = GRU(input_size=opt.embedding_size,
                            hidden_size=config.hidden_size,
                            batch_first=True)
        self.sent_linear = Linear(in_features=opt.vid_feat_size,
                                  out_features=config.hidden_size)
        self.transformer = BertLayer(config)

        if self.opt.deep:
            self.g_linear = nn.Sequential(Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=config.hidden_size,
                                                 out_features=1))

            self.V_linear = nn.Sequential(Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=config.hidden_size,
                                                 out_features=1))
        else:
            self.g_linear = Linear(in_features=config.hidden_size,
                                   out_features=1)
            self.V_linear = Linear(in_features=config.hidden_size,
                                   out_features=1)



    def sub_embedding(self, e):
        e_sub = self.embedding(e)
        _, hid = self.sent_gru(e_sub.unsqueeze(0))
        return hid.squeeze(0)

    # Using given embeddings, compute logit and value
    def mem_forward(self, vid_feature, sub_feature, temporal_hidden, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        memory_feature = self.sent_linear(vid_feature)
        memory_feature = memory_feature + sub_feature

        batch_size, memory_num, hidden_dim = memory_feature.size()
        if self.opt.demo:
            e, att = self.transformer(memory_feature, extended_attention_mask)
        else:
            e = self.transformer(memory_feature, extended_attention_mask)
        e_i = e.view(batch_size * memory_num, hidden_dim)

        h_t_i = e_i
        if self.opt.give_chance_to_last:
            h_t_i = e_i[:-1, :]
            g_i = self.g_linear(h_t_i)
            g = g_i.view(batch_size, memory_num-1)
            V_i = self.V_linear(h_t_i)
            V = V_i.view(batch_size, memory_num-1).mean(1)
        else:
            g_i = self.g_linear(h_t_i)
            g = g_i.view(batch_size, memory_num)
            V_i = self.V_linear(h_t_i)
            V = V_i.view(batch_size, memory_num).mean(1)
        h_t = temporal_hidden

        if self.opt.demo:
            return g, V, h_t, att
        else:
            return g, V, h_t
