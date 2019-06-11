import os
import pickle
from pprint import pprint

import numpy as np
import torch
from torch.nn import functional as F

from environment import create_env
from model.util import create_a3c_model
from util import set_seed
from vocab import Vocab


def demo(cfg):
    set_seed(cfg.seed)

    if not os.path.exists(cfg.ckpt):
        print('Invalid ckpt path:', cfg.ckpt)
        exit(1)
    ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
    print(cfg.ckpt, 'loaded')
    cfg.__dict__.update(ckpt['cfg'].__dict__)
    cfg.model = cfg.model.upper()
    pprint(cfg.__dict__)

    prepro_dir = os.path.join(cfg.prepro_dir, 'task%s' % (cfg.task_id))
    with open(os.path.join(prepro_dir, 'vocab.pk'), 'rb') as f:
        vocab = pickle.load(f)
        print()
        print(f.name, 'loaded')

    with open(os.path.join(prepro_dir, 'stats.pk'), 'rb') as f:
        stats = pickle.load(f)
        print(f.name, 'loaded')
        stats['max_ques_len'] = stats['max_sent_len']

    model = create_a3c_model(cfg, vocab, stats)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()

    env = create_env(cfg, 'test', vocab, stats, shuffle=False)
    env.set_model(model)
    env.set_gpu_id(None)
    print(env.dataset.path, 'loaded')

    while True:
        idx = np.random.randint(0, len(env.dataset))
        env.reset(idx)
        print('-'*80)
        print('Data ID:', env.data_idx)
        print()
        print('[Context]')
        qa_ptr = 0
        supp_ptrs, qa_ptrs = [], []
        for si, sent in enumerate(vocab.iss2wss(env.data.ctx.sents), start=1):
            if si == int(env.data.qas[qa_ptr].sent_idx):
                qa = env.data.qas[qa_ptr]
                ques = ' '.join(vocab.is2ws(qa.ques))
                answ = vocab.i2w(qa.answ_idx)
                supp = qa.supp_idxs
                print('{i:2d}\t{q}\t{a} {s}'.format(i=si, q=ques, a=answ, s=supp))
                qa_ptr += 1
                supp_ptrs.append(supp[0])
                qa_ptrs.append(qa.sent_idx)
            else:
                print('{i:2d} {s}'.format(i=si, s=' '.join(sent)))

        input('\nPress enter to continue\n')

        while not env.is_done():
            if len(env.memory) < cfg.memory_size-1:
                env._append_current()
                env.sent_ptr += 1

                # _print_addition_memory(env.memory, vocab)

                if env.is_qa_step():
                    read_output = _qa_forward(cfg, vocab, env, model)
                    env.qa_ptr += 1
                continue
            else:
                env._append_current()
                env.sent_ptr += 1

                # _print_addition_memory(env.memory, vocab)

                batch = env.observe()
                batch = {k: v.cuda() for k, v in batch.items()}

                with torch.no_grad():
                    write_output = model.mem_forward(**batch)
                act_logit, value = write_output['logit'], write_output['value']
                prob = F.softmax(act_logit, 1)
                _, action = prob.max(1, keepdim=True)

                _print_deletion_memory(env.memory, vocab, action)
                env.step(action=action.item(), **write_output)

                if env.is_qa_step():
                    read_output = _qa_forward(cfg, vocab, env, model)
                    env.qa_ptr += 1
        input('\nPress enter to continue\n')


def _qa_forward(cfg, vocab, env, model):
    batch = env.observe()
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        read_output = model.qa_forward(**batch)

    if env.is_qa_step():
        print('='*84)
        print()
        print('[Question]')
        ques = ' '.join(vocab.is2ws(env.current_qa.ques))
        supp = env.current_qa.supp_idxs
        print('     {ques} {supp}'.format(ques=ques, supp=supp))
        logit = read_output['logit']
        _, top5 = logit.topk(5, 1)
        top5 = top5.data[0]
        _print_qa_result(cfg, vocab, batch, read_output)
        print()
        print('Pred:', vocab.i2w(top5[0]), '(%s)' % (', '.join(vocab.is2ws(top5[1:]))))
        print('Answ:', vocab.i2w(env.current_qa.answ_idx))
        print('='*84)
    return read_output


def _print_deletion_memory(memory, vocab, action):
    action = action.item()
    print()
    print('-'*84)
    print('[Status]')
    print('ui\tmi\tcontents')
    for i, mem in enumerate(memory):
        uni_id = mem['unique_id']
        sent = mem['sents']
        sent = ' '.join(vocab.is2ws(sent))
        if i == action:
            print('*   {u_i:2d}\t{i:2d}\t{s}'.format(u_i=uni_id, i=i, s=sent))
        else:
            print('    {u_i:2d}\t{i:2d}\t{s}'.format(u_i=uni_id, i=i, s=sent))


def _print_addition_memory(memory, vocab):
    print()
    print('-'*84)
    print('[Status]')
    print('ui\tmi\tcontents')
    for i, mem in enumerate(memory):
        uni_id = mem['unique_id']
        sent = mem['sents']
        sent = ' '.join(vocab.is2ws(sent))
        print('{u_i:2d}\t{i:2d}\t{s}'.format(u_i=uni_id, i=i, s=sent))


def _print_qa_result(cfg, vocab, batch, result):
    attn = result['attns'].data[0].mean(0)
    print('\n[Result]')
    print('mi si attn atn1 atn2 atn3 memory')
    print('-'*50)
    for mi, (sent, sent_len) in enumerate(zip(batch['ctx'].data[0],
                                              batch['sent_lens'].data[0])):
        sent = ' '.join(vocab.is2ws(sent[:sent_len]))
        print('{mi:2d} {si:2d} {attn:.2f} {atn1:.2f} {atn2:.2f} {atn3:.2f} '
              '{sent}'.format(
                mi=mi,
                si=batch['sents_idx'].data[0][mi],
                attn=attn[mi],
                atn1=result['attns'].data[0][min(cfg.num_hops-1, 0)][mi],
                atn2=result['attns'].data[0][min(cfg.num_hops-1, 1)][mi],
                atn3=result['attns'].data[0][min(cfg.num_hops-1, 2)][mi],
                sent=sent))
