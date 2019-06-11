import torch
from torch import nn
import torch.nn.functional as F

from model.base import Base
from util import Linear, GRU, GRUCell


class LRU_DNTM(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

        self.gamma_sigmoid = nn.Sigmoid()

        self.g_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

        self.V_rho = Linear(in_features=self.cfg.memory_dim, out_features=self.cfg.memory_dim)

        self.val_gru = GRUCell(input_size=self.cfg.memory_dim,
                               hidden_size=self.cfg.memory_dim)

        self.V_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

    def lru_query_embedding(self, query):
        lru_gamma = self.gamma_linear(query)
        lru_gamma = self.gamma_sigmoid(lru_gamma)
        return query, lru_gamma

    def mem_forward(self, **batch):
        e = batch['val_mems'][:, :, :-1, :].sum(1)
        hidden_state = batch['hidden_state']
        attn_logit_mvavg = batch['attn_logit_mvavg']
        lru_query = batch['lru_query'].sum(2).squeeze(1)

        lru_query, lru_gamma = self.lru_query_embedding(lru_query)

        batch_size, memory_size, _ = e.size()
        e_i = e.view(batch_size * memory_size, self.cfg.memory_dim)

        z_t = F.softmax(torch.bmm(e, lru_query.unsqueeze(2)).squeeze(-1), 1)

        gamma_t = lru_gamma
        v_tm1 = attn_logit_mvavg[:, :-1]
        v_t = 0.1*v_tm1 + 0.9*z_t

        g = z_t - gamma_t * v_tm1
        e_i = e.view(batch_size * memory_size, self.cfg.memory_dim)
        e_v = torch.sum(e_i, dim=0, keepdim=True)
        v_rho = self.V_rho(e_v)
        hidden_state = self.val_gru(v_rho, hidden_state)
        V = self.V_linear(hidden_state)
        return dict(logit=g,
                    value=V,
                    attn_logit_mvavg=attn_logit_mvavg,
                    hidden_state=hidden_state,
                    )
