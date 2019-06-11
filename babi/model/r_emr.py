import torch
from torch import nn

from model.base import Base
from util import Linear, GRU, GRUCell


class R_EMR(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_gru = GRU(input_size=self.cfg.memory_dim,
                         hidden_size=self.cfg.memory_dim//2,
                         bidirectional=True,
                         batch_first=True,
                         dropout=self.cfg.drop_prob)

        self.g_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

        self.V_rho = Linear(in_features=self.cfg.memory_dim, out_features=self.cfg.memory_dim)

        self.val_gru = GRUCell(input_size=self.cfg.memory_dim,
                               hidden_size=self.cfg.memory_dim)

        self.V_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

    def mem_forward(self, **batch):
        e = batch['val_mems'].sum(1)
        hidden_state = batch['hidden_state']
        attn_logit_mvavg = batch['attn_logit_mvavg']

        batch_size, memory_size, _ = e.size()
        e = self.e_gru(e)
        e_i = e.view(batch_size * memory_size, self.cfg.memory_dim)

        g_i = self.g_linear(e_i)
        g = g_i.view(batch_size, memory_size)
        e_v = torch.sum(e_i, dim=0, keepdim=True)
        v_rho = self.V_rho(e_v)
        hidden_state = self.val_gru(v_rho, hidden_state)
        V = self.V_linear(hidden_state)
        return dict(logit=g,
                    value=V,
                    attn_logit_mvavg=attn_logit_mvavg,
                    hidden_state=hidden_state,
                    )
