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
from environment import Environment
from model.util import create_a3c_model
from shared_adam import SharedBertAdam, SharedAdam
from test_in_parallel import test
from util import set_seed
from tokenization import BertTokenizer
from worker import TensorboardWorker, TrainWorker, ValidWorker
from dataset import TriviaDataset


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trivia-dir', default='./data/trivia_qa')
    parser.add_argument('--prepro-dir', type=str, default='./prepro/trivia_qa')
    parser.add_argument('--task', type=str, default='wikipedia')
    parser.add_argument('--train-story-token', default=1200)
    parser.add_argument('--test-story-token', default='all')
    parser.add_argument('--max-ques-token', type=int, default=39)

    parser.add_argument('--seed', type=int, default=int(time.mktime(datetime.now().timetuple())))
    parser.add_argument('--log-dir', default='./logs-test/%s' % datetime.now().strftime('%b%d_%H-%M-%S'))
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--prediction-file', type=str, default='prediction.json')
    parser.add_argument('--valid-set', type=str, default='verified-dev')
    parser.add_argument('--test-set', type=str, default='verified-dev')
    parser.add_argument('--use-pretrain', action='store_true')

    parser.add_argument('--num-episodes', type=int, default=500000)
    parser.add_argument('--num-workers', type=int, default=5)

    parser.add_argument('--rl-method', type=str, default='a3c')
    parser.add_argument('--model', type=str, default='LIFO')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased')
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--bert-hidden-size', type=int, default=768)
    parser.add_argument('--num-attention-heads', type=int, default=8)
    parser.add_argument('--memory-num', type=int, default=21)
    parser.add_argument('--memory-len', type=int, default=20)
    parser.add_argument('--drop-rate', type=float, default=0.1)

    parser.add_argument('--num-steps', type=int, default=1000000)
    parser.add_argument('--num-positions', type=int, default=64)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max-grad-norm', type=float, default=40)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1.00)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--qa-loss-coef', type=float, default=1.0)
    parser.add_argument('--rl-loss-coef', type=float, default=1.0)

    parser.add_argument('--debug', action='store_true')
    cfg = parser.parse_args()

    cfg.model = cfg.model.upper()
    if cfg.train_story_token == 'all' and cfg.test_story_token == 'all':
        cfg.prepro_dir = os.path.join(cfg.prepro_dir, 'train-all-test-all')
        cfg.train_story_token = 100000000
        cfg.test_story_token = 100000000
    elif cfg.test_story_token == 'all':
        cfg.prepro_dir = os.path.join(cfg.prepro_dir, 'train-%d-test-all' % (cfg.train_story_token))
        cfg.test_story_token = 100000000
    else:
        cfg.prepro_dir = os.path.join(cfg.prepro_dir, 'train-%d-test-%d' % (cfg.train_story_token, cfg.test_story_token))
    cfg.prediction_file = os.path.join(cfg.log_dir, cfg.prediction_file)
    return cfg


def main(cfg):
    ckpt = None
    if cfg.ckpt:
        if not os.path.exists(cfg.ckpt):
            print('Invalid ckpt path:', cfg.ckpt)
            exit(1)
        ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc:storage)

        print(cfg.ckpt, 'loaded')
        loaded_cfg = ckpt['cfg'].__dict__
        pprint(loaded_cfg)

        del loaded_cfg['num_workers']
        del loaded_cfg['use_pretrain']
        del loaded_cfg['test_set']

        cfg.__dict__.update(loaded_cfg)
        cfg.model = cfg.model.upper()

        print()
        print('Merged Config')
        pprint(cfg.__dict__)
    else:
        os.makedirs(os.path.join(cfg.log_dir, 'ckpt'))

    shared_model = create_a3c_model(cfg)

    if cfg.use_pretrain:
        print("LOAD pretrain parameter for BERT from ./pretrain/pytorch_model.bin...")
        pretrain_param = torch.load('./pretrain/pytorch_model.bin', map_location=lambda storage, loc: storage)
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        new_pretrain_param = pretrain_param.copy()
        for k, v in pretrain_param.items():
            new_key = 'model.' + k
            new_pretrain_param[new_key] = v
            del new_pretrain_param[k]
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

    if cfg.qa_loss_coef == 0.0:
        for name, param in shared_model.named_parameters():
            if 'bert' in name or 'qa_outputs' in name:
                param.requires_grad = False
    shared_model.share_memory()

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)
    optim = SharedAdam(shared_model.parameters(), lr=cfg.lr)
    if ckpt is not None:
        optim.load_state_dict(ckpt['optim'])
    optim.share_memory()

    set_seed(cfg.seed)

    train_env = Environment(cfg, 'train', tokenizer, shuffle=True)
    valid_env = Environment(cfg, cfg.valid_set, tokenizer, shuffle=False)

    done = mp.Value('i', False)
    if ckpt is not None:
        gstep = mp.Value('i', ckpt['step'])
    else:
        gstep = mp.Value('i', 0)
    queue = mp.Queue()

    procs = []
    # p = TrainWorker(cfg, len(procs), done, shared_model, optim, tokenizer, train_env, queue, gstep)
    p = ValidWorker(cfg, len(procs), done, shared_model, optim, tokenizer, valid_env, gstep)
    if cfg.debug:
        p.run()
    else:
        p.start()
    procs.append(p)

    for _ in range(cfg.num_workers-1):
        p = TrainWorker(cfg, len(procs), done, shared_model, optim, tokenizer, train_env, queue, gstep)
        p.start()
        procs.append(p)

    p = TensorboardWorker(cfg, len(procs), queue, done, gstep)
    p.start()
    procs.append(p)

    for p in procs:
        p.join()
    print('All processes is finished:', cfg.log_dir)


if __name__ == '__main__':
    cfg = config()
    print('Config')
    pprint(cfg.__dict__)
    print()

    assert(not (cfg.demo and cfg.test))
    if cfg.demo:
        demo(cfg)
    elif cfg.test:
        test(cfg)
    else:
        main(cfg)
