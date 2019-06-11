import torch
from torch import nn
import torch.nn.functional as F

from model.base import Base
from util import Linear, GRU, GRUCell


class R_EMR(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_gru = GRU(input_size=self.cfg.bert_hidden_size,
                            hidden_size=self.cfg.hidden_size,
                            batch_first=True,
                            dropout=self.cfg.drop_rate)

        self.e_gru = GRU(input_size=self.cfg.hidden_size,
                         hidden_size=self.cfg.hidden_size,
                         bidirectional=True,
                         batch_first=True,
                         dropout=self.cfg.drop_rate)

        self.g_linear = nn.Sequential(Linear(in_features=self.cfg.hidden_size*2,
                                             out_features=self.cfg.hidden_size),
                                      nn.ReLU(),
                                      Linear(in_features=self.cfg.hidden_size,
                                             out_features=self.cfg.hidden_size),
                                      nn.ReLU(),
                                      Linear(in_features=self.cfg.hidden_size,
                                             out_features=1))

        if self.cfg.rl_method == 'a3c':
            self.V_rho = nn.Sequential(Linear(in_features=self.cfg.hidden_size*2,
                                              out_features=self.cfg.hidden_size*2),
                                       nn.ReLU(),
                                       Linear(in_features=self.cfg.hidden_size*2,
                                              out_features=self.cfg.hidden_size*2))

            self.val_gru = GRUCell(input_size=self.cfg.hidden_size*2,
                                   hidden_size=self.cfg.hidden_size*2)

            self.V_linear = nn.Sequential(Linear(in_features=self.cfg.hidden_size*2,
                                                 out_features=self.cfg.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=self.cfg.hidden_size,
                                                 out_features=self.cfg.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=self.cfg.hidden_size,
                                                 out_features=1))

    def forward_sent(self, ctx_words, mask):
        hid = self.sent_gru(ctx_words, mask)
        hid = F.max_pool1d(hid.permute(0, 2, 1), self.cfg.memory_len)
        hid = hid.permute(0, 2, 1).squeeze(1)
        return hid

    def mem_forward(self, **batch):
        e = batch['ctx_sent_vec']
        hidden_state = batch['hidden_state']
        attn_logit_mvavg = batch['attn_logit_mvavg']

        batch_size, memory_num, _ = e.size()
        e = self.e_gru(e)
        e_i = e.view(batch_size * memory_num, self.cfg.hidden_size*2)

        g_i = self.g_linear(e_i)
        g = g_i.view(batch_size, memory_num)
        if self.cfg.rl_method == 'a3c':
            e_v = torch.sum(e_i, dim=0, keepdim=True)
            v_rho = self.V_rho(e_v)
            hidden_state = self.val_gru(v_rho, hidden_state)
            V = self.V_linear(hidden_state)
        else:
            V = g.data.new(1, 1).fill_(0.0)
        return dict(logit=g,
                    value=V,
                    attn_logit_mvavg=attn_logit_mvavg,
                    hidden_state=hidden_state,
                    )
