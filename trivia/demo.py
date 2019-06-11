from termcolor import colored
import os
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F

from environment import Environment
from model.util import create_a3c_model
from tokenization import BertTokenizer


def demo(cfg):
    if not os.path.exists(cfg.ckpt):
        print('Invalid ckpt path:', cfg.ckpt)
        exit(1)
    ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
    print(cfg.ckpt, 'loaded')
    loaded_cfg = ckpt['cfg'].__dict__

    del loaded_cfg['test_set']
    del loaded_cfg['use_pretrain']
    del loaded_cfg['num_workers']
    del loaded_cfg['num_episodes']
    del loaded_cfg['memory_num']
    del loaded_cfg['memory_len']

    cfg.__dict__.update(loaded_cfg)
    cfg.model = cfg.model.upper()
    pprint(cfg.__dict__)

    model = create_a3c_model(cfg)
    model.load_state_dict(ckpt['model'])
    model.cuda()

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)

    env = Environment(cfg, cfg.test_set, tokenizer, shuffle=True)
    env.set_model(model)
    env.set_gpu_id(torch.cuda.current_device())
    print(env.dataset.path, 'loaded')
    while True:
        model.eval()
        env.reset()
        print('-'*80)
        print('Data ID:', env.data_idx)
        print()
        print('[Context]')
        print(' '.join(tokenizer.convert_ids_to_tokens(env.data.ctx_words)))
        print()

        ques = ' '.join(tokenizer.convert_ids_to_tokens(env.data.ques_words))
        ques = ques.replace(' ##', '')
        ques = ques.replace('##', '')
        print('[Question]')
        print(ques)
        print()
        answs = []
        indices = list(set(env.data.indices))
        for indice in indices:
            s_idx = indice[0]
            e_idx = indice[1]
            answ = ' '.join(tokenizer.convert_ids_to_tokens(env.data.ctx_words[s_idx:e_idx+1]))
            answ = answ.replace(' ##', '')
            answ = answ.replace('##', '')
            answs.append(answ)
        print('[Answer]')
        for i in range(len(answs)):
            print('%d.' % (i+1), answs[i])

        input('\nPress enter to continue\n')
        while not env.is_done():
            if len(env.memory) < cfg.memory_num-1:
                env._append_current()
                env.sent_ptr += 1
            else:
                if cfg.model == 'LIFO':
                    break
                env._append_current()
                env.sent_ptr += 1

                batch, solvable, mem_solvable = env.observe()
                batch = {k: v.cuda() for k, v in batch.items()}

                result = model.mem_forward(**batch)
                logit, value = result['logit'], result['value']

                prob = F.softmax(logit, 1)
                _, action = prob.max(1, keepdim=True)

                env.step(action=action.item(), **result)

                _print_mem_result(cfg, tokenizer, batch, prob, action, result, solvable, mem_solvable, answs)
                # input()

        assert(len(env.memory) <= cfg.memory_num)
        result = _qa_forward(env, model)
        batch = result['batch']
        _print_qa_result(cfg, tokenizer, batch, result, answs, env)
        input()


def _qa_forward(env, model):
    batch, solvable, mem_solvable = env.observe(qa_step=True)
    if solvable:
        batch = {k: v.cuda() for k, v in batch.items()}
        result = model.qa_forward(**batch)
        s_idx, e_idx = batch['s_idx'], batch['e_idx']
        p1_logit, p2_logit = result['start_logits'], result['end_logits']

        batch_size, c_len = p1_logit.size()
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len) * float('-inf')).cuda().tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
        score = (ls(p1_logit).unsqueeze(2) + ls(p2_logit).unsqueeze(1)) + mask
        score, p1_pred = score.max(dim=1)
        score, p2_pred = score.max(dim=1)
        p1_pred = torch.gather(p1_pred, 1, p2_pred.view(-1, 1)).squeeze(-1)
    else:
        p1_pred = None
        p2_pred = None
    return dict(p1_pred=p1_pred, p2_pred=p2_pred, solvable=solvable, batch=batch, mem_solvable=mem_solvable)


def _print_mem_result(cfg, tokenizer, batch, prob, action, result, solvable, mem_solvable, answs):
    print('\n[Result]')
    print('Value:', result['value'].item())
    print('Total solvable:', solvable)
    print('-'*150)
    memory_num = batch['input_ids'].size()[1] // cfg.memory_len
    sents = batch['input_ids'].view(1, memory_num, cfg.memory_len)
    sents = sents.squeeze(0)
    prob = prob.squeeze(0)
    ptrs = batch['sent_ptrs'].squeeze(0)
    print('   si mi prob    memory')
    for i, sent in enumerate(sents):
        ptr = ptrs[i].item()
        sent = ' '.join(tokenizer.convert_ids_to_tokens(sent[:cfg.memory_len].tolist()))
        sent = sent.replace(' ##', '')
        sent = sent.replace('##', '')
        sent = replace_answ_to_color(answs, sent)
        mark = '*' if i == action.item() else ' '
        p = prob[i].item()
        print('{mark} {ptr:3d} {i:2d} {prob:.5f} {sent}'.format(
            i=i, ptr=ptr, mark=mark, prob=p, sent=sent))
        if mem_solvable[i]:
            print('Solvable:', mem_solvable[i])
        print()
        print('-'*150)


def replace_answ_to_color(answs, sent):
    for answ in answs:
        sent = sent.replace(answ, colored(answ, 'red'))
    return sent


def _print_qa_result(cfg, tokenizer, batch, result, answs, env):
    print('\n[Result]')
    print('-'*150)
    print('mi memory')
    print('-'*150)
    sents = batch['input_ids'].squeeze(0)
    seg_ids = batch['segment_ids'].squeeze(0)
    mask_ids = batch['input_mask'].squeeze(0)
    ques_word_len = (mask_ids == seg_ids).tolist().count(0)

    sents = sents[ques_word_len-1:]
    sents_len = (cfg.memory_num-1) * cfg.memory_len
    sents = sents[:sents_len]
    # if sents.size()[0] < cfg.memory_num * cfg.memory_len:
    #     sents = ' '.join(tokenizer.convert_ids_to_tokens[sents.tolist()])
    sents = sents.view(sents.size()[0]//cfg.memory_len, -1)
    for i, sent in enumerate(sents):
        sent = ' '.join(tokenizer.convert_ids_to_tokens(sent[:cfg.memory_len].tolist()))
        sent = sent.replace(' ##', '')
        sent = sent.replace('##', '')
        sent = replace_answ_to_color(answs, sent)
        print('{i:2d} {sent}'.format(i=i, sent=sent))
        print('Solvable:', result['mem_solvable'][i])
        print()

    if result['solvable']:
        s_idx, e_idx = batch['s_idx'].item(), batch['e_idx'].item()
        gt = batch['input_ids'].squeeze(0)[s_idx:e_idx+1]
        gt = tokenizer.convert_ids_to_tokens(gt.tolist())
        gt = ' '.join(gt)
        gt = gt.replace(' ##', '')
        gt = gt.replace('##', '')

        p1_pred, p2_pred = result['p1_pred'].item(), result['p2_pred'].item()
        if p2_pred < p1_pred:
            print()
            print('p2 < p1')
            print()
            return
        pred = batch['input_ids'].squeeze(0)[p1_pred:p2_pred+1]
        pred = tokenizer.convert_ids_to_tokens(pred.tolist())
        pred = ' '.join(pred)
        pred = pred.replace(' ##', '')
        pred = pred.replace('##', '')

        print()
        print('[Prediction]')
        print('  ', pred)
        print('[Ground truth]')
        print('  ', gt)
    else:
        print()
        print('Fail to remaining the sentence in memory')
        print()
        return
