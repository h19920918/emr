import multiprocessing as mp
import os
import random

import numpy as  np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from dataset import TriviaDataset


PAD_ID = 0


class Environment(object):
    def __init__(self, cfg, set_id, tokenizer, shuffle):
        self.cfg = cfg
        self.set_id = set_id
        self.task = cfg.task
        self.max_word_len = self.cfg.memory_num * self.cfg.memory_len + self.cfg.max_ques_token + 3
        self.max_ques_len = cfg.max_ques_token
        self.shuffle = shuffle

        dataset_path = os.path.join(cfg.prepro_dir, cfg.task, set_id)
        self.dataset = TriviaDataset(cfg, dataset_path, tokenizer)

        self.data_idx = 0

    def set_model(self, model):
        self._model = model

    def set_gpu_id(self, gpu_id):
        self._gpu_id = gpu_id

    def reset(self, idx=None):
        if idx is not None:
            self.data_idx = idx
        elif self.shuffle:
            self.data_idx = np.random.randint(len(self.dataset))
        self.data = self.dataset[self.data_idx]

        self.memory = []
        self.sent_ptr = 0
        self.action_idx = None
        self.hidden_state = torch.zeros(1, self.cfg.hidden_size*2)

    def _append_current(self):
        mem = dict(sent_words=self.data.sents_words[self.sent_ptr],
                   attn_logit_mvavg=torch.zeros(1),
                   sent_ptr=self.sent_ptr,
                   )
        mem['ctx_words_vec'], mem['ctx_sent_vec'] = self._get_ctx_vec(mem)
        self.memory.append(mem)
        # if self.action_idx == None or self.cfg.model in ['FIFO', 'LIFO']:
        #     self.memory.append(mem)
        # elif self.sent_ptr == self.data.ctx_sent_len-1:
        #     self.memory.append(mem)
        # else:
        #     self.memory.insert(self.action_idx, mem)

    def _memory_reset(self):
        for i in range(len(self.memory)):
            del self.memory[i]['ctx_words_vec']
            del self.memory[i]['ctx_sent_vec']
            ctx_words_vec, ctx_sent_vec = self._get_ctx_vec(self.memory[i])
            self.memory[i]['ctx_words_vec'] = ctx_words_vec
            self.memory[i]['ctx_sent_vec'] = ctx_sent_vec
            self.memory[i]['attn_logit_mvavg'] = self.memory[i]['attn_logit_mvavg'].data
        self.hidden_state = self.hidden_state.data

    def _get_ctx_vec(self, mem_cell):
        sent_words = mem_cell['sent_words']
        input_ids = np.full([1, self.cfg.memory_len], PAD_ID)
        input_ids[0, :len(sent_words)] = sent_words
        input_mask = (input_ids != PAD_ID) * 1
        input_ids = torch.LongTensor(input_ids).cuda(self._gpu_id)
        input_mask = torch.LongTensor(input_mask).cuda(self._gpu_id)
        segment_ids = torch.LongTensor(np.full([1, self.cfg.memory_len], 1)).cuda(self._gpu_id)

        ctx_words = self._model.forward_ctx(input_ids, segment_ids)
        if self.cfg.model in ['T_EMR', 'R_EMR', 'LRU_DNTM']:
            ctx_sent = self._model.forward_sent(ctx_words, input_mask).cpu()
        else:
            ctx_sent = torch.zeros(1, self.cfg.hidden_size*2)
        ctx_words = ctx_words.cpu()
        return ctx_words, ctx_sent

    def observe(self, qa_step=False):
        sents_words = []
        ctx_words_vec = []
        ctx_sent_vec = []
        attn_logit_mvavg = []
        mem_solvable = []
        s_idxes = []
        e_idxes = []
        answ_words = []
        sent_ptrs = []
        for indice in self.data.indices:
            answ_words.append(self.data.ctx_words[indice[0]:indice[1]+1])
            s_idxes.append(indice[0])
            e_idxes.append(indice[1])

        # import random
        # random.shuffle(self.memory)
        for mem in self.memory:
            sents_words.append(mem['sent_words'])
            ctx_words_vec.append(mem['ctx_words_vec'])
            ctx_sent_vec.append(mem['ctx_sent_vec'])
            attn_logit_mvavg.append(mem['attn_logit_mvavg'])
            sent_ptrs.append(mem['sent_ptr'])

            solvables, _, _ = self._find_answ(np.array(mem['sent_words']), answ_words)
            solvable = True if sum(solvables) >= 1.0 else False
            mem_solvable.append(solvable)

        sent_ptrs = np.array(sent_ptrs)
        ques_input_ids = np.full([self.max_ques_len+2], PAD_ID)
        ques_input_mask = np.full([self.max_ques_len+2], PAD_ID)
        ques_segment_ids = np.full([self.max_ques_len+2], PAD_ID)
        if qa_step:
            max_word_len = len(self.memory) * self.cfg.memory_len + self.cfg.max_ques_token + 3
            input_ids = np.full([max_word_len], PAD_ID)
            input_mask = np.full([max_word_len], PAD_ID)
            segment_ids = np.full([max_word_len], PAD_ID)

            _sents_words = []
            for sent in sents_words:
                _sents_words += sent
            sents_words = _sents_words
            sents_words = np.array(sents_words)
            ctx_word_len = len(sents_words)
            ques_word_len = self.data.ques_word_len
            t_length = ques_word_len + ctx_word_len + 3
            input_ids[0] = self.data.cls
            input_ids[1:ques_word_len+1] = self.data.ques_words
            input_ids[ques_word_len+1] = self.data.sep
            input_ids[ques_word_len+2:ques_word_len+2+ctx_word_len] = sents_words
            input_ids[ques_word_len+2+ctx_word_len] = self.data.sep

            segment_ids[:1+ques_word_len+1] = 0
            segment_ids[ques_word_len+2:ques_word_len+2+ctx_word_len+1] = 1
            input_mask[:t_length] = 1
        elif self.cfg.model in ['LRU_DNTM']:
            max_word_len = len(self.memory) * self.cfg.memory_len
            input_ids = np.full([max_word_len], PAD_ID)
            input_mask = np.full([max_word_len], PAD_ID)
            segment_ids = np.full([max_word_len], PAD_ID)

            _sents_words = []
            for sent in sents_words:
                _sents_words += sent
            sents_words = _sents_words
            sents_words = np.array(sents_words)
            ctx_word_len = len(sents_words)
            input_ids[:ctx_word_len] = sents_words
            segment_ids[:ctx_word_len] = 1
            input_mask[:ctx_word_len] = 1

            ques_word_len = len(self.memory[-1]['sent_words'])
            ques_input_ids[:ques_word_len] = self.memory[-1]['sent_words']
            ques_segment_ids[:ques_word_len] = 1
            ques_input_mask[:ques_word_len] = 1
        else:
            max_word_len = len(self.memory) * self.cfg.memory_len
            input_ids = np.full([max_word_len], PAD_ID)
            input_mask = np.full([max_word_len], PAD_ID)
            segment_ids = np.full([max_word_len], PAD_ID)

            _sents_words = []
            for sent in sents_words:
                _sents_words += sent
            sents_words = _sents_words
            sents_words = np.array(sents_words)
            ctx_word_len = len(sents_words)
            input_ids[:ctx_word_len] = sents_words
            segment_ids[:ctx_word_len] = 1
            input_mask[:ctx_word_len] = 1

        batch = dict(input_ids=input_ids,
                     input_mask=input_mask,
                     segment_ids=segment_ids,
                     ques_input_ids=ques_input_ids,
                     ques_input_mask=ques_input_mask,
                     ques_segment_ids=ques_segment_ids,
                     sent_ptrs=sent_ptrs,
                     )

        solvables, s_idxes, e_idxes = self._find_answ(sents_words, answ_words)
        if len(s_idxes) == 0:
            s_idx = 1e10
            e_idx = 1e10
            solvable = False
        else:
            solvable = True if sum(solvables) >= 1.0 else False
            s_idx = min(s_idxes)
            e_idx = e_idxes[s_idxes.index(s_idx)]
            if qa_step and solvable:
                s_idx = s_idx + ques_word_len + 2
                e_idx = e_idx + ques_word_len + 2
        batch['s_idx'] = np.array([s_idx])
        batch['e_idx'] = np.array([e_idx])

        batch = default_collate([batch])
        batch['s_idx'] = batch['s_idx'].long()
        batch['e_idx'] = batch['e_idx'].long()
        batch['ctx_words_vec'] = torch.cat(ctx_words_vec, 1)
        batch['ctx_sent_vec'] = torch.stack(ctx_sent_vec, 1)
        batch['hidden_state'] = self.hidden_state
        batch['attn_logit_mvavg'] = torch.stack(attn_logit_mvavg, 1)
        return batch, solvable, mem_solvable

    def _find_answ(self, ctx_words, answ_words):
        s_idxes = []
        e_idxes = []
        solvables = []
        for answ_word in answ_words:
            answ_word_len = len(answ_word)
            s_idx, e_idx = None, None
            solvable = False
            for i in range(len(ctx_words)-answ_word_len+1):
                if all(ctx_words[i:i+answ_word_len] == answ_word):
                    s_idx = i
                    e_idx = i+answ_word_len-1
                    solvable = True
                    s_idxes.append(s_idx)
                    e_idxes.append(e_idx)
                    solvables.append(solvable)
                    break
        return solvables, s_idxes, e_idxes

    def step(self, action, **result):
        attn_logit_mvavg = result['attn_logit_mvavg']
        hidden_state = result['hidden_state']
        self.hidden_state = hidden_state
        for i, mem in enumerate(self.memory):
            mem['attn_logit_mvavg'] = attn_logit_mvavg[0][[i]].cpu()
        assert(0 <= action < self.cfg.memory_num)
        if self.cfg.model == 'UNIFORM':
            action = random.randint(0, self.cfg.memory_num-1)
        del self.memory[action]
        self.action_idx = action

    def is_done(self):
        assert(self.sent_ptr <= self.data.ctx_sent_len)
        return self.sent_ptr == self.data.ctx_sent_len
