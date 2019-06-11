import argparse
from datetime import datetime
import multiprocessing
import os
import pickle
from pprint import pprint
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

from demo import demo
from environment import create_env
from model.util import create_a3c_model
from prepro import prepro_babi
from shared_adam import SharedAdam
from test_in_parallel import test
from util import set_seed
from vocab import Vocab
from worker import TensorboardWorker, TrainWorker, ValidWorker


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--babi-dir', default='./data/babi/en-valid-10k')
    parser.add_argument('--prepro-dir', default='./prepro/babi')
    parser.add_argument('--prepro', action='store_true')
    parser.add_argument('--task-id', type=int, default=2)

    parser.add_argument('--num-workers', type=int, default=9)
    parser.add_argument('--log-dir', default='./logs-test/%s' % datetime.now().strftime('%b%d_%H-%M-%S'))
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test-set', type=str, default='test')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=int(time.mktime(datetime.now().timetuple())))

    parser.add_argument('--model', type=str, default='fifo')

    parser.add_argument('--num-valid-episodes', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=400000)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-grad-norm', type=float, default=40)
    parser.add_argument('--drop-prob', type=float, default=0.1)

    parser.add_argument('--memory-size', type=int, default=6)
    parser.add_argument('--memory-dim', type=int, default=20)
    parser.add_argument('--num-attention-heads', type=int, default=4)
    parser.add_argument('--num-positions', type=int, default=6)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1.00)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--qa-loss-coef', type=float, default=1.0)

    parser.add_argument('--num-hops', type=int, default=3)

    parser.add_argument('--pre-ckpt', default=None)

    cfg = parser.parse_args()
    cfg.model = cfg.model.upper()
    return cfg


def main(cfg):
    ckpt = None
    if cfg.ckpt:
        if not os.path.exists(cfg.ckpt):
            print('Invalid ckpt path:', cfg.ckpt)
            exit(1)
        ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)

        print(cfg.ckpt, 'loaded')
        loaded_cfg = ckpt['cfg'].__dict__
        pprint(loaded_cfg)

        del loaded_cfg['num_episodes']
        del loaded_cfg['num_workers']
        del loaded_cfg['test_set']
        del loaded_cfg['pre_ckpt']

        cfg.__dict__.update(loaded_cfg)
        cfg.model = cfg.model.upper()

        print()
        print('Merged Config')
        pprint(cfg.__dict__)
    else:
        os.makedirs(os.path.join(cfg.log_dir, 'ckpt'))

    prepro_dir = os.path.join(cfg.prepro_dir, 'task%s' % (cfg.task_id))
    with open(os.path.join(prepro_dir, 'vocab.pk'), 'rb') as f:
        vocab = pickle.load(f)

    with open(os.path.join(prepro_dir, 'stats.pk'), 'rb') as f:
        stats = pickle.load(f)
        stats['max_ques_len'] = stats['max_sent_len']

    shared_model = create_a3c_model(cfg, vocab, stats)

    if cfg.pre_ckpt is not None:
        pretrain_param = torch.load(cfg.pre_ckpt, map_location=lambda storage, loc: storage)
        pretrain_param = pretrain_param['model']
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        new_pretrain_param = pretrain_param.copy()
        pretrain_param = new_pretrain_param.copy()

        metadata = getattr(pretrain_param, '_metadata', None)
        if metadata is not None:
            pretrain_param._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                pretrain_param, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(shared_model, prefix='')
        print("Weights of {} not initialized from pretrained model: {}".format(
                shared_model.__class__.__name__, missing_keys))
        print("Weights from pretrained model not used in {}: {}".format(
                shared_model.__class__.__name__, unexpected_keys))

    if ckpt is not None:
        shared_model.load_state_dict(ckpt['model'])
    shared_model.share_memory()

    params = filter(lambda p: p.requires_grad, shared_model.parameters())
    optim = SharedAdam(params, lr=cfg.lr)

    if ckpt is not None:
        optim.load_state_dict(ckpt['optim'])
    optim.share_memory()

    set_seed(cfg.seed)

    done = mp.Value('i', False)
    if ckpt is not None:
        gstep = mp.Value('i', ckpt['step'])
    else:
        gstep = mp.Value('i', 0)
    queue = mp.Queue()

    train_env = create_env(cfg, 'train', vocab, stats, shuffle=True)
    valid_shuffle = False if cfg.num_valid_episodes == 0 else True
    valid_env = create_env(cfg, 'valid', vocab, stats, shuffle=valid_shuffle)

    procs = []
    if cfg.debug:
        p = TrainWorker(cfg, len(procs), done, shared_model, optim, vocab, stats, train_env, queue, gstep)
        # p = ValidWorker(cfg, len(procs), done, shared_model, optim, vocab, stats, valid_env, gstep)
        p.run()
        return

    p = ValidWorker(cfg, len(procs), done, shared_model, optim, vocab, stats, valid_env, gstep)
    p.start()
    procs.append(p)

    for _ in range(cfg.num_workers-1):
        p = TrainWorker(cfg, len(procs), done, shared_model, optim, vocab, stats, train_env, queue, gstep)
        p.start()
        procs.append(p)

    p = TensorboardWorker(cfg, len(procs), queue, done, gstep)
    p.start()
    procs.append(p)

    for p in procs:
        p.join()
    print('All processes is finished:', cfg.log_dir)


if __name__ == '__main__':
    cfg = _parse_args()
    print('Config')
    pprint(cfg.__dict__)
    print()
    if cfg.prepro:
        prepro_babi(cfg)

    assert(not (cfg.demo and cfg.test))
    if cfg.demo:
        demo(cfg)
    elif cfg.test:
        test(cfg)
    else:
        main(cfg)
