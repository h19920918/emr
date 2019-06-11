from collections import namedtuple
from glob import glob
import os
import re

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from vocab import Vocab


Context = namedtuple('BabiContext', ['sents', 'sents_rev', 'sents_idx', 'sent_lens', 'ctx_len'])
QA = namedtuple('BabiQA', ['ques', 'ques_rev', 'ques_len', 'answ_idx', 'sent_idx', 'supp_idxs'])
BabiDataRow = namedtuple('BabiDataRow', ['ctx', 'qas'])


class BabiDataset(Dataset):
    def __init__(self, path, vocab):
        self.path = path
        self.vocab = vocab

        fname_pattern = 'story_*.txt'
        fnames = glob(os.path.join(self.path, fname_pattern))
        self.data_ids = dict([(int(re.search('\w+_(\d+).txt', fname).group(1))-1,
                               fname)
                              for fname in fnames])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        with open(self.data_ids[idx]) as f:
            qas = []

            while True:
                line = next(f).strip()
                if line == '':
                    break

                sent_idx, ques, answ, supp = line.split('\t')
                ques = self.vocab.ws2is(ques.split())
                supp_idxs = [int(idx) for idx in supp.split(' ')]

                qas.append(QA(ques=ques,
                              ques_rev=list(reversed(ques)),
                              ques_len=len(ques),
                              answ_idx=self.vocab.w2i(answ),
                              sent_idx=int(sent_idx),
                              supp_idxs=supp_idxs,
                              ))

            sents = []
            sents_rev = []
            sents_idx = []
            for sent in f:
                line = sent.strip()
                idx, sent = line.split('\t')
                sent = sent.split()
                sents.append(sent)
                sents_rev.append(list(reversed(sent)))
                sents_idx.append(int(idx))

        sent_lens = list(map(len, sents))
        sents = self.vocab.wss2iss(sents)
        sents_rev = self.vocab.wss2iss(sents_rev)

        ctx = Context(sents=sents, sents_rev=sents_rev, sents_idx=sents_idx,
                      sent_lens=sent_lens, ctx_len=len(sents))
        return BabiDataRow(ctx=ctx, qas=qas)


class BatchCollator(object):
    def __init__(self, max_ctx_len=None, max_sent_len=None, max_ques_len=None):
        self.max_ctx_len = max_ctx_len
        self.max_sent_len = max_sent_len
        self.max_ques_len = max_ques_len

    def __call__(self, batch):
        batch = sorted(batch, key=lambda b: b.ctx.ctx_len, reverse=True)

        batch_size = len(batch)

        max_ctx_len = self.max_ctx_len
        if max_ctx_len is None:
            max_ctx_len = max(map(lambda b: b.ctx.ctx_len, batch))

        max_sent_len = self.max_sent_len
        if max_sent_len is None:
            max_sent_len = max(map(lambda b: max(b.ctx.sent_lens), batch))

        max_ques_len = self.max_ques_len
        if max_ques_len is None:
            max_ques_len = max(map(lambda b: b.qas[-1].ques_len, batch))

        batch_padded = []
        for b in batch:
            ctx = np.full([max_ctx_len, max_sent_len], Vocab.PAD_ID)
            ctx_rev = np.full([max_ctx_len, max_sent_len], Vocab.PAD_ID)
            time = np.zeros([max_ctx_len], dtype=np.int64)

            ques = np.full([max_ques_len], Vocab.PAD_ID)
            ques_rev = np.full([max_ques_len], Vocab.PAD_ID)

            sent_lens = np.full([max_ctx_len], 0)

            sents = b.ctx.sents[-max_ctx_len:]
            sents_rev = b.ctx.sents_rev[-max_ctx_len:]
            ctx_len = len(sents)
            for i, (sent, sent_rev, sent_len) in enumerate(zip(sents,
                                                               sents_rev,
                                                               b.ctx.sent_lens[-max_ctx_len:])):
                ctx[i, :sent_len] = sent
                ctx_rev[i, :sent_len] = sent_rev
                sent_lens[i] = sent_len
                time[i] = len(sents) - i

            ques[:b.qas[-1].ques_len] = b.qas[-1].ques
            ques_rev[:b.qas[-1].ques_len] = b.qas[-1].ques_rev

            batch_padded.append(dict(ctx=ctx, ques=ques,
                                     time=time,
                                     ctx_rev=ctx_rev, ques_rev=ques_rev,
                                     sent_lens=sent_lens, ctx_len=ctx_len,
                                     ques_len=b.qas[-1].ques_len,
                                     answ_idx=b.qas[-1].answ_idx))

        return default_collate(batch_padded)


def create_batch_collator(cfg, stats):
    collator = BatchCollator(max_ctx_len=stats['max_ctx_len'],
                             max_sent_len=stats['max_sent_len'],
                             max_ques_len=stats['max_sent_len'])
    return collator
