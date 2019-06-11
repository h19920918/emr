import torch
from torch import nn

from model.base import Base
from util import Linear, Transformer, GRUCell, get_sinusoid_encoding_table


class T_EMR(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = Transformer(hidden_size=self.cfg.memory_dim,
                                       intermediate_size=self.cfg.memory_dim*4,
                                       num_attention_heads=self.cfg.num_attention_heads,
                                       dropout=self.cfg.drop_prob)

        self.g_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

        self.V_rho = Linear(in_features=self.cfg.memory_dim, out_features=self.cfg.memory_dim)

        self.val_gru = GRUCell(input_size=self.cfg.memory_dim,
                               hidden_size=self.cfg.memory_dim)

        self.V_linear = Linear(in_features=self.cfg.memory_dim, out_features=1)

        table = get_sinusoid_encoding_table(self.cfg.num_positions, self.cfg.memory_dim, padding_idx=0)
        self.position_enc = nn.Embedding.from_pretrained(table, freeze=True)

    def mem_forward(self, **batch):
        e = batch['val_mems'].sum(1)
        hidden_state = batch['hidden_state']
        attn_logit_mvavg = batch['attn_logit_mvavg']

        seq_length = e.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=e.device)
        position_embeddings = self.position_enc(position_ids)
        e = e + position_embeddings.unsqueeze(0)

        batch_size, memory_size, _ = e.size()
        e = self.transformer(e)
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
