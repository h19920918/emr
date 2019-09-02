from .transformer import GRUCell, Linear, GRU, BertLayer, BertConfig
from .tvqa_abc import ABC

import torch

class FIFO(ABC):
    def __init__(self, opt):
        super().__init__(opt)
        self.sent_gru = GRU(input_size=opt.embedding_size,
                            hidden_size=opt.hidden_size,
                            batch_first=True)

    def sub_embedding(self, e):
        e_sub = self.embedding(e)
        _, hid = self.sent_gru(e_sub.unsqueeze(0))
        return hid.squeeze(0)

    # Using given embeddings, compute logit and value
    def mem_forward(self, vid_feature, sub_feature, temporal_hidden, attention_mask):
        g = temporal_hidden.data.new(temporal_hidden.size()).fill_(0.0)
        g[:, -1] = 1e+10

        V = g.new(1, 1).fill_(0.0)
        g = g.new(1, self.opt.memory_num).fill_(0.0)
        h_t = temporal_hidden
        # Logit, Value computing part
        return g, V, h_t
