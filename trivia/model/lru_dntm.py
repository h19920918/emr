import torch
from torch import nn
import torch.nn.functional as F

from model.base import Base
from util import GRU, Linear, GRUCell


class LRU_DNTM(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_gru = GRU(input_size=self.cfg.bert_hidden_size,
                            hidden_size=self.cfg.hidden_size*2,
                            batch_first=True,
                            dropout=self.cfg.drop_rate)
        self.q_write = GRU(input_size=self.cfg.bert_hidden_size,
                           hidden_size=self.cfg.hidden_size*2,
                           batch_first=True,
                           dropout=self.cfg.drop_rate)
        self.gamma_linear = nn.Sequential(Linear(in_features=self.cfg.hidden_size*2,
                                                 out_features=self.cfg.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=self.cfg.hidden_size,
                                                 out_features=self.cfg.hidden_size),
                                          nn.ReLU(),
                                          Linear(in_features=self.cfg.hidden_size,
                                                 out_features=1))
        self.gamma_sigmoid = nn.Sigmoid()
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

    def query_embedding(self, ques, mask):
        hid = self.q_write(ques, mask)
        hid = F.max_pool1d(hid.permute(0, 2, 1), self.cfg.memory_len)
        ques = hid.permute(0, 2, 1).squeeze(1)

        lru_gamma = self.gamma_linear(ques)
        lru_gamma = self.gamma_sigmoid(lru_gamma)
        return ques, lru_gamma

    def mem_forward(self, **batch):
        ques_input_ids = batch['ques_input_ids']
        ques_input_mask = batch['ques_input_mask']
        ques_segment_ids = batch['ques_segment_ids']

        e = batch['ctx_sent_vec'][:, :-1, :]
        hidden_state = batch['hidden_state']
        attn_logit_mvavg = batch['attn_logit_mvavg']

        batch_size, memory_num, _ = e.size()
        ques = self.forward_ques(ques_input_ids, ques_segment_ids)
        ques, lru_gamma = self.query_embedding(ques, ques_input_mask)

        z_t = F.softmax(torch.bmm(e, ques.unsqueeze(2)).squeeze(-1), 1)

        gamma_t = lru_gamma
        v_tm1 = attn_logit_mvavg[:, :-1]
        v_t = 0.1*v_tm1 + 0.9*z_t

        g = z_t - gamma_t * v_tm1
        e_i = e.view(batch_size * memory_num, self.cfg.hidden_size*2)
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
