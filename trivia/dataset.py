from collections import namedtuple
from glob import glob
import os

from math import ceil
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

TriviaDataId = namedtuple('TriviaDataId', ['doc_id', 'ques_id'])
TriviaDataRow = namedtuple('TriviaDataRow',
                          ['ctx_words', 'ques_words', 'sents_words',
                           'ctx_word_len', 'ques_word_len', 'ctx_sent_len', 'sents_word_len',
                           'indices',
                           'cls', 'sep'])

SEP = ['[SEP]']
CLS = ['[CLS]']
PAD_ID = 0


class TriviaDataset(Dataset):
    def __init__(self, cfg, path, tokenizer):
        self.cfg = cfg
        self.path = path
        self.tokenizer = tokenizer
        self.example_ids = [fname.split('/')[-1].split('.')[-2] for fname in
                glob(os.path.join(self.path, '*.index'))]
        self.example_ids.sort()

    def __len__(self):
        return len(self.example_ids)

    def __getitem__(self, idx):
        ques_fname = os.path.join(self.path, '%s.ques.words' % self.example_ids[idx])
        with open(ques_fname) as f:
            ques_words = f.readlines()
            ques_words = [word.strip() for word in ques_words]
            ques_word_len = len(ques_words)
            ques_words = self.tokenizer.convert_tokens_to_ids(ques_words)

        ctx_fname = os.path.join(self.path, '%s.story.words' % self.example_ids[idx])
        with open(ctx_fname) as f:
            ctx_words = f.readlines()
            ctx_words = [word.strip() for word in ctx_words]
            ctx_words = self.tokenizer.convert_tokens_to_ids(ctx_words)
            ctx_word_len = len(ctx_words)
            ctx_sent_len = ceil(ctx_word_len / self.cfg.memory_len)

            sents_words = [ctx_words[i*self.cfg.memory_len:(i+1)*self.cfg.memory_len]
                           for i in range(ctx_sent_len)]
            sents_word_len = [len(sent) for sent in sents_words]

        idx_fname = os.path.join(self.path, '%s.index' % self.example_ids[idx])
        indices = []
        with open(idx_fname) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                indices.append((int(line[0].strip()), int(line[1].strip())))

        cls = self.tokenizer.convert_tokens_to_ids(CLS)[0]
        sep = self.tokenizer.convert_tokens_to_ids(SEP)[0]

        return TriviaDataRow(ctx_words=ctx_words, ques_words=ques_words, sents_words=sents_words,
                             ctx_word_len=ctx_word_len, ques_word_len=ques_word_len,
                             ctx_sent_len=ctx_sent_len, sents_word_len=sents_word_len,
                             indices=indices, cls=cls, sep=sep)


class BatchCollator(object):
    def __init__(self, max_token_length):
        self.max_token_length = max_token_length

    def __call__(self, batch):
        batch_padded = []
        for b in batch:
            input_ids = np.full([self.max_token_length], PAD_ID)
            input_mask = np.full([self.max_token_length], PAD_ID)
            segment_ids = np.full([self.max_token_length], PAD_ID)

            t_length = b.ques_word_len+b.ctx_word_len+3
            input_ids[0] = b.cls
            input_ids[1:b.ques_word_len+1] = b.ques_words
            input_ids[b.ques_word_len+1] = b.sep
            input_ids[b.ques_word_len+2:b.ques_word_len+2+b.ctx_word_len] = b.ctx_words
            input_ids[b.ques_word_len+2+b.ctx_word_len] = b.sep

            segment_ids[:1+b.ques_word_len+1] = 0
            segment_ids[b.ques_word_len+2:b.ques_word_len+2+b.ctx_word_len+1] = 1
            input_mask[:t_length] = 1

            assert len(input_ids) == self.max_token_length
            assert len(input_mask) == self.max_token_length
            assert len(segment_ids) == self.max_token_length

            doc_offset = b.ques_word_len + 2
            start_position = b.s_idx + doc_offset
            end_position = b.e_idx + doc_offset

            batch_padded.append(dict(input_ids=input_ids,
                                     input_mask=input_mask,
                                     segment_ids=segment_ids,
                                     start_position=start_position,
                                     end_position=end_position))
        return default_collate(batch_padded)
