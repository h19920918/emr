from collections import Counter
import csv
import os
import pickle
import re
from tqdm import tqdm

from util import get_line_num
from vocab import Vocab


def prepro_babi(cfg):
    for task_id in map(str, range(cfg.task_id, cfg.task_id+1)):
        for set_id in ['train', 'valid', 'test']:
            task_path = os.path.join(cfg.prepro_dir, 'task%s' % task_id, set_id)
            if not os.path.exists(task_path):
                os.makedirs(task_path)
            _prepro_task(cfg, task_id, set_id)
        _build_vocab(cfg, task_id)
        _build_stats(cfg, task_id)


def _write_to_file(story_fname, sents, qas):
    with open(story_fname, 'w+') as f:
        for qa in qas:
            sent_idx, que, ans, sup = qa
            f.write('{i}\t{q}\t{a}\t{s}\n'.format(
                i=sent_idx, q=que, a=ans, s=sup))
        f.write('\n')
        for i, sent in enumerate(sents):
            f.write('{i}\t{s}\n'.format(i=i+1, s=sent))


def _prepro_task(cfg, task_id, set_id):
    fname = os.path.join(cfg.babi_dir, 'qa%s_%s.txt' % (task_id, set_id))
    task_path = os.path.join(cfg.prepro_dir, 'task%s' % (task_id), set_id)
    vocab = Counter()
    example_id = story_id = 1
    line_num = get_line_num(fname)
    max_ctx_len = max_sent_len = 0
    sents = []

    def tokenize(s):
        # exclude . and ?
        return [x for x in re.split('(\W+)', s) if x.strip() not in ['', '.', '?']]

    with open(fname, 'r') as lines:
        for line in tqdm(lines, desc=fname, total=line_num):
            line = line.strip().lower().split('\t')

            if len(line) == 1:  # context
                tokens = tokenize(line[0])
                if tokens[0] == '1':
                    if sents:
                        story_fname = os.path.join(task_path, 'story_%d.txt' % (story_id))
                        _write_to_file(story_fname, sents, qas)
                        story_id += 1
                    sents = []
                    qas = []
                tokens = tokens[1:]
                max_sent_len = max(len(tokens), max_sent_len)
                sents.append(' '.join(tokens))
                vocab.update(tokens)
            else:  # question, answer, supporting facts
                tokens = tokenize(line[0])[1:]
                max_sent_len = max(len(tokens), max_sent_len)
                que = ' '.join(tokens)
                ans = line[1]
                sup = line[2]
                qas.append((len(sents)+1, que, ans, sup))

                example_fname = os.path.join(task_path, 'example_%d.txt' % (example_id))
                _write_to_file(example_fname, sents, qas[-1:])
                example_id += 1

                max_ctx_len = max(max_ctx_len, len(sents))
                vocab.update(tokens)
                vocab.update([ans])

                sents.append(que)
        else:
            story_fname = os.path.join(task_path, 'story_%d.txt' % (story_id))
            _write_to_file(story_fname, sents, qas)

    vocab_path = os.path.join(task_path, 'vocab.pk')
    with open(vocab_path, 'wb+') as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    tsv_path = os.path.join(task_path, 'vocab.tsv')
    with open(tsv_path, 'w+') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

    stats_path = os.path.join(task_path, 'stats.pk')
    with open(stats_path, 'wb+') as f:
        stats = dict(max_ctx_len=max_ctx_len,
                     max_sent_len=max_sent_len)
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

def _build_vocab(cfg, task_id):
    task_path = os.path.join(cfg.prepro_dir, 'task%s' % (task_id))

    with open(os.path.join(task_path, 'train', 'vocab.pk'), 'rb') as f:
        train = pickle.load(f)

    with open(os.path.join(task_path, 'valid', 'vocab.pk'), 'rb') as f:
        valid = pickle.load(f)

    with open(os.path.join(task_path, 'test', 'vocab.pk'), 'rb') as f:
        test = pickle.load(f)

    """with open(os.path.join(task_path, 'test_large', 'vocab.pk'), 'rb') as f:
        test_large = pickle.load(f)"""

    vocab = train + valid + test

    # Vocab TSV
    with open(os.path.join(task_path, 'vocab.tsv'), 'w+') as f:
        writer = writer = csv.writer(f, delimiter='\t')
        writer.writerows(sorted(vocab.items(), key=lambda x: x[1], reverse=True))

    v = Vocab(vocab)
    # Vocab pickle
    with open(os.path.join(task_path, 'vocab.pk'), 'wb+') as f:
        pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)


def _build_stats(cfg, task_id):
    task_path = os.path.join(cfg.prepro_dir, 'task%s' % (task_id))

    with open(os.path.join(task_path, 'train', 'stats.pk'), 'rb') as f:
        train = pickle.load(f)

    with open(os.path.join(task_path, 'valid', 'stats.pk'), 'rb') as f:
        valid = pickle.load(f)

    with open(os.path.join(task_path, 'test', 'stats.pk'), 'rb') as f:
        test = pickle.load(f)

    with open(os.path.join(task_path, 'stats.pk'), 'wb+') as f:
        stats = dict(max_ctx_len=max(train['max_ctx_len'],
                                     valid['max_ctx_len'],
                                     test['max_ctx_len'],),
                     max_sent_len=max(train['max_sent_len'],
                                      valid['max_sent_len'],
                                      test['max_sent_len'],))
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cfg = set_config()
    prepro_babi(cfg)
