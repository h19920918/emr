from functools import partial

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.init as init

from util import position_encoding
from vocab import Vocab


WEIGHT_TYING_TYPES = (ADJACENT, LAYERWISE) = ('adjacent', 'layerwise')


class Memory(nn.Module):
    def __init__(self, memory_size, max_ctx_len, max_sent_len, memory_dim, vocab_size,
                 pos_enc):
        super().__init__()
        self.memory_size = memory_size
        self.max_ctx_len = max_ctx_len
        self.max_sent_len = max_sent_len
        self.memory_dim = memory_dim
        self.vocab_size = vocab_size
        self.pos_enc = pos_enc

    def forward(self, ctx, time):
        m, m_ = self._memory_embedding(self.A, self.TA, ctx, time)
        c, c_ = self._memory_embedding(self.C, self.TC, ctx, time)
        return (m, m_), (c, c_)

    def _memory_embedding(self, embedding, temporal_embedding, ctx, time):
        m = embedding(ctx.view([-1, self.max_sent_len]))
        m = m.view([-1, self.memory_size, self.max_sent_len, self.memory_dim])

        # Position encoding
        pos_enc = Variable(m.data.new(self.pos_enc))
        m = (m * pos_enc).sum(2)

        # Temporal embedding
        temp_m = m + temporal_embedding(time)
        return temp_m, m

    def sent_embedding(self, ctx):
        m = self._sent_embedding(self.A, ctx)
        c = self._sent_embedding(self.C, ctx)
        return m, c

    def _sent_embedding(self, embedding, ctx):
        batch_size, num_sents, _ = ctx.size()
        m = embedding(ctx.view([batch_size * num_sents, self.max_sent_len]))
        m = m.view([batch_size, num_sents, self.max_sent_len, self.memory_dim])

        # Position encoding
        pos_enc = Variable(m.data.new(self.pos_enc))
        m = (m * pos_enc).sum(2)
        return m

    def temp_embedding(self, time):
        ta = self._temp_embedding(self.TA, time)
        tc = self._temp_embedding(self.TC, time)
        return ta, tc

    def _temp_embedding(self, temporal_embedding, time):
        return temporal_embedding(time)


def _embedding(num_embeddings, embedding_dim):
    e = nn.Embedding(num_embeddings, embedding_dim, padding_idx=Vocab.PAD_ID)
    e.weight.data.normal_(0, 0.1)
    e.weight.data[Vocab.PAD_ID].fill_(0)
    return e


class AdjacentMemory(Memory):
    def __init__(self, prev, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if prev:
            self.A = prev.C
            self.C = _embedding(self.vocab_size, self.memory_dim)

            self.TA = prev.TC
            self.TC = _embedding(self.max_ctx_len + 1, self.memory_dim)
        else:
            self.A = _embedding(self.vocab_size, self.memory_dim)
            self.C = _embedding(self.vocab_size, self.memory_dim)

            self.TA = _embedding(self.max_ctx_len + 1, self.memory_dim)
            self.TC = _embedding(self.max_ctx_len + 1, self.memory_dim)


class LayerwiseMemory(Memory):
    def __init__(self, prev, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if prev:
            self.A = prev.A
            self.C = prev.C

            self.TA = prev.TA
            self.TC = prev.TC
        else:
            self.A = _embedding(self.vocab_size, self.memory_dim)
            self.C = _embedding(self.vocab_size, self.memory_dim)

            self.TA = _embedding(self.max_ctx_len + 1, self.memory_dim)
            self.TC = _embedding(self.max_ctx_len + 1, self.memory_dim)


class Base(nn.Module):
    def __init__(self, cfg, num_hops, memory_size, max_ctx_len, max_sent_len, memory_dim,
                 vocab_size, weight_tying=ADJACENT):
        super().__init__()
        self.cfg = cfg
        self.drop_prob = self.cfg.drop_prob
        self.num_hops = num_hops
        self.memory_size = memory_size
        self.max_ctx_len = max_ctx_len
        self.max_sent_len = max_sent_len
        self.memory_dim = memory_dim
        self.vocab_size = vocab_size
        self.weight_tying = weight_tying

        self.pos_enc = position_encoding(max_sent_len, memory_dim)
        self.memories = nn.ModuleList()

        prev = None
        for i in range(num_hops):
            if self.adjacent_weight_tying:
                memory = AdjacentMemory(prev, memory_size, max_ctx_len, max_sent_len,
                                        memory_dim, vocab_size, self.pos_enc)
            elif self.layerwise_weight_tying:
                memory = LayerwiseMemory(prev, memory_size, max_ctx_len, max_sent_len,
                                         memory_dim, vocab_size, self.pos_enc)
            else:
                assert(False)
            self.memories.append(memory)
            prev = memory

        if self.adjacent_weight_tying:
            self.B = self.memories[0].A
        else:
            self.B = _embedding(vocab_size, memory_dim)

        self.W = nn.Linear(self.memory_dim, self.vocab_size)
        if self.adjacent_weight_tying:
            self.W.weight = self.memories[-1].C.weight
        else:
            self.W.weight.data.normal_(0, 0.1)

    def get_mem_vec(self, ctx):
        key_mem, val_mem = [], []

        for memory in self.memories:
            key, val = memory.sent_embedding(ctx)
            key_mem.append(key)
            val_mem.append(val)

        key_mem = torch.cat(key_mem, 1)
        val_mem = torch.cat(val_mem, 1)
        return key_mem, val_mem

    def forward(self, query, rel_time, key_mems, val_mems, **kwargs):
        # Query embedding
        u, lru_gamma, ctrl_hidden = self.query_embedding(query, **kwargs)
        logits = []
        attns = []
        read_vectors = []
        attn_sum = 0

        for i, memory in enumerate(self.memories):
            key_time, val_time = memory.temp_embedding(rel_time)
            key_mem = key_mems[:, i, :, :] + key_time
            val_mem = val_mems[:, i, :, :] + val_time

            # Match memory and query
            logit = (key_mem*u).sum(2)
            logits.append(logit)
            p = F.softmax(logit, 1)
            attns.append(p)
            val_mem = val_mem * p.unsqueeze(2)

            # Output
            o = val_mem.sum(1, keepdim=True)
            read_vectors.append(o.squeeze(1))
            u = o + u
            attn_sum = attn_sum + p
        attns = torch.stack(attns, 1)
        attn_avg = attns.mean(1)
        attn_logit = torch.stack(logits, 1).mean(1)
        read_vector = torch.cat(read_vectors, -1)
        return dict(u=u, attns=attns, attn_avg=attn_avg, attn_logit=attn_logit,
                    read_vector=read_vector,
                    ctrl_hidden=ctrl_hidden,
                    lru_gamma=lru_gamma)

    def qa_forward(self, **kwargs):
        result = self.forward(**kwargs)
        logit = self.W(result['u']).squeeze(1)
        return dict(logit=logit, **result)

    def mem_forward(self, **kwargs):
        raise NotImplementedError()

    def query_embedding(self, query, **kwargs):
        query = self.B(query)
        pos_enc = Variable(query.data.new(self.pos_enc))
        u = (query * pos_enc).sum(1, keepdim=True)
        lru_gamma = ctrl_hidden = None
        return u, lru_gamma, ctrl_hidden

    @property
    def adjacent_weight_tying(self):
        return self.weight_tying == ADJACENT

    @property
    def layerwise_weight_tying(self):
        return self.weight_tying == LAYERWISE
