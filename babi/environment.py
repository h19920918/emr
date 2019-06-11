import os
import re
from tqdm import tqdm

import numpy as  np
import random
import torch
from torch.utils.data.dataloader import default_collate

from dataset import BabiDataset
from vocab import Vocab


class Environment(object):
    def __init__(self, cfg, set_id, vocab, stats, shuffle):
        self.cfg = cfg
        self.set_id = set_id
        self.task_id = cfg.task_id
        self.vocab = vocab
        self.stats = stats
        self.max_sent_len = stats['max_sent_len']
        self.max_ques_len = stats['max_ques_len']
        self.shuffle = shuffle

        task_path = os.path.join(cfg.prepro_dir, 'task%s' % (self.task_id), set_id)
        self.dataset = BabiDataset(task_path, vocab)
        self.data_idx = 0
        self._model = None

    def set_model(self, model):
        self._model = model

    def set_gpu_id(self, gpu_id):
        self._gpu_id = gpu_id

    def reset(self, data_idx=None):
        if data_idx is not None:
            self.data_idx = data_idx
        elif self.shuffle:
            self.data_idx = np.random.randint(len(self.dataset))
        self.data = self.dataset[self.data_idx]

        self.memory = []
        self.sent_ptr = 0
        self.qa_ptr = 0
        self.hidden_state = torch.zeros(1, self.cfg.memory_dim)
        self.unique_id = 1

    def update_read(self, read_output):
        self.read_vector = read_output['read_vector']
        self.ctrl_hidden = read_output['ctrl_hidden']

    def _append_current(self):
        mem = dict(sents=self.data.ctx.sents[self.sent_ptr],
                   attn_logit_mvavg=torch.zeros(1),
                   sents_rev=self.data.ctx.sents_rev[self.sent_ptr],
                   sents_idx=self.data.ctx.sents_idx[self.sent_ptr],
                   unique_id=self.unique_id,
                   )
        mem['key_mem'], mem['val_mem'] = self._get_mem_vec(mem)
        self.memory.append(mem)
        self.unique_id += 1

    def _get_mem_vec(self, mem_cell):
        sent = np.full([1, 1, self.max_sent_len], Vocab.PAD_ID)
        sent[0, 0, :len(mem_cell['sents'])] = mem_cell['sents']
        sent = torch.LongTensor(sent).cuda(self._gpu_id)

        key_mem, val_mem = self._model.get_mem_vec(sent)
        return key_mem, val_mem

    def is_done(self):
        assert(self.sent_ptr <= len(self.data.ctx.sents))
        return self.sent_ptr == len(self.data.ctx.sents) - 1

    @property
    def current_qa(self):
        return self.data.qas[self.qa_ptr]

    def is_supp_fact(self, sent_idx):
        for qa in self.data.qas:
            if sent_idx in qa.supp_idxs:
                return True
        return False

    def observe(self):
        sents = []
        sents_rev = []
        sents_idx = []
        attn_logit_mvavg = []
        key_mems, val_mems = [], []
        supp_flags = []
        unique_ids = []

        for mem in self.memory:
            sents.append(mem['sents'])
            sents_rev.append(mem['sents_rev'])
            sents_idx.append(mem['sents_idx'])
            attn_logit_mvavg.append(mem['attn_logit_mvavg'])
            key_mems.append(mem['key_mem'])
            val_mems.append(mem['val_mem'])
            supp_flags.append(self.is_supp_fact(mem['sents_idx']))
            unique_ids.append(mem['unique_id'])

        ctx_len = len(sents)
        ctx = np.full([ctx_len, self.max_sent_len], Vocab.PAD_ID)
        ctx_rev = np.full([ctx_len, self.max_sent_len], Vocab.PAD_ID)
        sent_lens = np.full([ctx_len], 0)
        rel_time = np.zeros([ctx_len], dtype=np.int64)
        abs_time = np.zeros([ctx_len], dtype=np.int64)
        uni_ids = np.zeros([ctx_len], dtype=np.int64)

        for i, (sent, sent_rev, sent_idx, uni_id) in enumerate(zip(sents, sents_rev, sents_idx, unique_ids)):
            sent_lens[i] = len(sent)
            ctx[i, :sent_lens[i]] = sent
            ctx_rev[i, :sent_lens[i]] = sent_rev
            rel_time[i] = len(sents)-i-1
            abs_time[i] = self.sent_ptr+1-sent_idx
            uni_ids[i] = uni_id

        query = np.full([self.max_ques_len], Vocab.PAD_ID)
        query_rev = np.full([self.max_ques_len], Vocab.PAD_ID)
        query_len = self.current_qa.ques_len
        query[:query_len] = self.current_qa.ques
        query_rev[:query_len] = self.current_qa.ques_rev

        query = np.full([self.max_ques_len], Vocab.PAD_ID)
        query_rev = np.full([self.max_ques_len], Vocab.PAD_ID)
        query_len = len(self.data.ctx.sents[self.sent_ptr])
        query[:query_len] = self.data.ctx.sents[self.sent_ptr]
        query_rev[:query_len] = self.data.ctx.sents_rev[self.sent_ptr]

        lru_query = self.memory[-1]['val_mem']

        batch = dict(ctx=ctx, query=query,
                     rel_time=rel_time,
                     abs_time=abs_time,
                     sents_idx=np.array(sents_idx),
                     ctx_rev=ctx_rev, query_rev=query_rev,
                     sent_lens=sent_lens, ctx_len=ctx_len,
                     query_len=query_len,
                     supp_flags=np.array(supp_flags, dtype=np.int64),
                     lru_query=lru_query,
                     unique_ids=uni_ids,
                     )

        batch['answ_idx'] = np.array([self.current_qa.answ_idx])
        batch = default_collate([batch])
        batch['attn_logit_mvavg'] = torch.stack(attn_logit_mvavg, 1)
        batch['key_mems'] = torch.stack(key_mems, 2)
        batch['val_mems'] = torch.stack(val_mems, 2)
        batch['hidden_state'] = self.hidden_state
        return batch

    def check_solvable(self):
        supp = set(self.current_qa.supp_idxs)
        mem = set([mem['sents_idx'] for mem in self.memory])
        return len(supp - mem) == 0

    def step(self, action, **result):
        attn_logit_mvavg = result['attn_logit_mvavg']
        hidden_state = result['hidden_state']
        self.hidden_state = hidden_state
        for i, mem in enumerate(self.memory):
            mem['attn_logit_mvavg'] = attn_logit_mvavg[0][[i]].cpu()
        assert(0 <= action < self.cfg.memory_size)
        if self.cfg.model == 'UNIFORM':
            action = random.randint(0, self.cfg.memory_size-1)
        del self.memory[action]
        self.action_idx = action

    def is_qa_step(self):
        return self.data.ctx.sents_idx[self.sent_ptr] == self.current_qa.sent_idx

    def is_next_qa_step(self):
        return self.data.ctx.sents_idx[self.sent_ptr+1] == self.current_qa.sent_idx

    def qa_step(self, attns):
        attn = attns.mean(0)
        for i, mem in enumerate(self.memory):
            mem['atten_accum'] += attn[i]

        self.qa_ptr += 1


def create_env(cfg, set_id, vocab, stats, shuffle):
    env = Environment(cfg, set_id, vocab, stats, shuffle)
    return env
