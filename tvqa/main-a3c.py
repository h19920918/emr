import argparse
from datetime import datetime
import multiprocessing
import os
import pickle
from pprint import pprint
import time
import pickle

import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

from environment import Environment
from shared_adam import SharedAdam, SharedBertAdam
from util import set_seed
from worker import TensorboardWorker, TrainWorker, ValidWorker
from tvqa_dataset import TVQADataset
from model.tvqa_abc import ABC
from model.util import create_model
from vqa_pretrain import pretrain
from visu import visu
from test_in_parallel import test

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/tvqa_train_processed.json",
                         help="train set path")
    parser.add_argument("--valid_path", type=str, default="./data/tvqa_val_processed.json",
                         help="valid set path")
    parser.add_argument("--test_path", type=str, default="./data/tvqa_test_public_processed.json",
                         help="test set path")
    parser.add_argument("--glove_path", type=str, default="./data/glove.6B.300d.txt",
                         help="GloVe pretrained vector path")
    parser.add_argument("--vid_feat_path", type=str, default="./data/tvqa_imagenet_pool5_hq.h5",
                         help="imagenet feature path")
    parser.add_argument("--vid_feat_size", type=int, default=2048,
                         help="visual feature dimension")
    parser.add_argument("--word2idx_path", type=str, default="./cache/word2idx.pickle",
                         help="word2idx cache path")
    parser.add_argument("--idx2word_path", type=str, default="./cache/idx2word.pickle",
                         help="idx2word cache path")
    parser.add_argument("--vocab_embedding_path", type=str, default="./cache/vocab_embedding.pickle",
                         help="vocab_embedding cache path")
    parser.add_argument("--vcpt_path", type=str, default="./data/det_visual_concepts_hq.pickle",
                         help="visual concepts feature path")

    parser.add_argument('--model', type=str, choices=["FIFO", "LIFO", "UNIFORM", "LRU_DNTM", "R_EMR", "T_EMR"], default="T_EMR",
                         help='Set the model')
    parser.add_argument('--spv', type=float, default=1,
                        help='Rate of adding reward by random question showing')

    parser.add_argument("--memory_length", default=40, type=int,
                        help="The maximum total input sequence length of memory slot after WordPiece tokenization. "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=39, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs to run")
    parser.add_argument('--num-episodes', type=int, default=500000,
            help='Total number of training document')
    parser.add_argument('--num-workers', type=int, default=3,
            help='The number of LEMN model to use')
    parser.add_argument('--log-dir', default='./logs-test/%s' % datetime.now().strftime('%b%d_%H-%M-%S'),
            help='Path of the log file to be saved')

    parser.add_argument('--ckpt', default=None,
            help='Load the checkpoint file to process continue training or demo or test')
    parser.add_argument('--pretrain-dir', default=None,
            help='Load the checkpoint file for using pretrained qa model')

    parser.add_argument('--demo', action='store_true',
            help='Run the demo version for trained model')
    parser.add_argument('--test', action='store_true',
            help='Test trained model')
    parser.add_argument('--pretrain', action='store_true',
            help='Do pretraining for vqa model')
    parser.add_argument('--seed', type=int, default=int(time.mktime(datetime.now().timetuple())))

    parser.add_argument('--memory-num', type=int, default=20,
            help='The number of memory entry')

    parser.add_argument('--lr', type=float, default=0.0001,
            help='Learning rate with Adam optimizer')
    parser.add_argument('--max-grad-norm', type=float, default=50,
            help='Value loss coefficient')
    parser.add_argument('--l2-loss-coef', type=float, default=0,
            help='L2 loss coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
            help='Discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.00,
            help='Parameter for GAE')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
            help='Entropy term coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
            help='Value loss coefficient')
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")

    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden size in memory model')
    parser.add_argument('--num-attention-heads', type=int, default=8,
                        help='The number of attention heads')

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                         "of training.")
    parser.add_argument('--dataset_ready', action='store_true',
            help='If dataset is ready, just load saved features')
    parser.add_argument('--debug', action='store_true',
            help='Enter debugging mode')

    parser.add_argument("--device", type=int, default=0, help="gpu ordinal, -1 indicates cpu")

    parser.add_argument("--bsz", type=int, default=8, help="mini-batch size")
    parser.add_argument("--test_bsz", type=int, default=25, help="mini-batch size for testing")
    parser.add_argument("--log_freq", type=int, default=400, help="print, save training info")

    parser.add_argument("--no_glove", action="store_true", help="not use glove vectors")
    parser.add_argument("--no_ts", action="store_true", help="no timestep annotation, use full length feature")
    parser.add_argument("--input_streams", type=str, nargs="+", choices=["vcpt", "sub", "imagenet"],
                         help="input streams for the model, will use both `vcpt` and `sub` streams")
    parser.add_argument("--n_layers_cls", type=int, default=1, help="number of layers in classifier")
    parser.add_argument("--hsz1", type=int, default=150, help="hidden size for the first lstm")
    parser.add_argument("--hsz2", type=int, default=300, help="hidden size for the second lstm")
    parser.add_argument("--embedding_size", type=int, default=300, help="word embedding dim")
    parser.add_argument("--max_sub_l", type=int, default=300, help="max length for subtitle")
    parser.add_argument("--max_vcpt_l", type=int, default=300, help="max length for visual concepts")
    parser.add_argument("--max_vid_l", type=int, default=480, help="max length for video feature")
    parser.add_argument("--vocab_size", type=int, default=0, help="vocabulary size")
    parser.add_argument("--no_normalize_v", action="store_true", help="do not normalize video featrue")
    parser.add_argument("--no_core_driver", action="store_true",
                         help="hdf5 driver, default use `core` (load into RAM), if specified, use `None`")

    parser.add_argument("--random_ts", action="store_true", help="pick supporting frame randomly (for pretraining)")
    parser.add_argument("--give_chance_to_last", action="store_true", help="do not consider last pick when deleting (test)")
    parser.add_argument("--deep", action="store_true", help="use deep network for value and policy (test)")
    parser.add_argument("--large", action="store_true", help="use large validation set (valid)")
    cfg = parser.parse_args()

    cfg.normalize_v = not cfg.no_normalize_v
    cfg.with_ts = not cfg.no_ts
    cfg.input_streams = [] if cfg.input_streams is None else cfg.input_streams
    cfg.vid_feat_flag = True if "imagenet" in cfg.input_streams else False
    cfg.h5driver = None if cfg.no_core_driver else "core"

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

        del loaded_cfg['num_episodes']
        del loaded_cfg['num_workers']
        del loaded_cfg['prepro_dir']

        cfg.__dict__.update(loaded_cfg)
        cfg.model = cfg.model.upper()

        print('Merged Config')
        pprint(cfg.__dict__)

    else:
        os.makedirs(os.path.join(cfg.log_dir, 'ckpt'), exist_ok=True)

    prepro_ckpt = None
    if cfg.pretrain_dir is not None:
        if not os.path.exists(cfg.pretrain_dir):
            print('Invalid pretraining ckpt path:', cfg.pretrain_dir)
            exit(1)
        prepro_ckpt = torch.load(os.path.join(cfg.pretrain_dir, "best_valid.pth"), map_location=lambda storage, loc:storage)


    dset = TVQADataset(cfg)
    cfg.vocab_size = len(dset.word2idx)
    dset_valid = TVQADataset(cfg)

    # Prepare model
    shared_model = create_model(cfg)

    if ckpt is not None:
        shared_model.load_state_dict(ckpt['model'])

    # Load TVQA ABC part
    if prepro_ckpt is not None:
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(prepro_ckpt, '_metadata', None)
        prepro_ckpt = prepro_ckpt.copy()
        if metadata is not None:
            prepro_ckpt._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                prepro_ckpt, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(shared_model, prefix='')
        print("Weights of {} not initialized from pretrained model: {}".format(
                shared_model.__class__.__name__, missing_keys))
        print("Weights from pretrained model not used in {}: {}".format(
                shared_model.__class__.__name__, unexpected_keys))
        # shared_model.load_state_dict(prepro_ckpt)

    shared_model.share_memory()

    optim = SharedAdam(filter(lambda p: p.requires_grad, shared_model.parameters()), lr=cfg.lr)

    if ckpt is not None:
        optim.load_state_dict(ckpt['optim'])

    optim.share_memory()

    set_seed(cfg.seed)
    dset.set_mode("train")
    train_env = Environment(cfg, 'train', dset, shuffle=True)
    dset_valid.set_mode("valid")
    valid_env = Environment(cfg, 'valid', dset_valid, shuffle=False)

    done = mp.Value('i', False)
    if ckpt is not None:
        gstep = mp.Value('i', ckpt['step'])
    else:
        gstep = mp.Value('i', 0)
    queue = mp.Queue()

    if cfg.debug:
        procs = []
        p = ValidWorker(cfg, len(procs), done, shared_model, optim, valid_env, gstep)
        # p = TrainWorker(cfg, len(procs), done, shared_model, optim, train_env, queue, gstep)
        p.run()
    else:
        procs = []
        p = ValidWorker(cfg, len(procs), done, shared_model, optim, valid_env, gstep)
        p.start()
        procs.append(p)

        for _ in range(cfg.num_workers-1):
            p = TrainWorker(cfg, len(procs), done, shared_model, optim, train_env, queue, gstep)
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
        visu(cfg)
    elif cfg.test:
        test(cfg)
    elif cfg.pretrain:
        pretrain(cfg)
    else:
        main(cfg)
