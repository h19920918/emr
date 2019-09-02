from math import floor
import os
import pickle
from pprint import pprint

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn

from environment import Environment
from util import set_seed
from util import F1score, ExactMatch

from tvqa_dataset import TVQADataset
from model.tvqa_abc import ABC
from model.util import create_model
from tqdm import tqdm


def visu(cfg):
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    set_seed(cfg.seed)
    mn = cfg.memory_num
    demo = cfg.demo
    large = cfg.large
    if not os.path.exists(cfg.ckpt):
        print('Invalid ckpt path:', cfg.ckpt)
        exit(1)
    ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
    print(cfg.ckpt, 'loaded')
    cfg.__dict__.update(ckpt['cfg'].__dict__)
    pprint(cfg.__dict__)
    cfg.memory_num = mn
    cfg.large = large
    cfg.demo = demo
    dset = TVQADataset(cfg)
    cfg.vocab_size = len(dset.word2idx)

    model = create_model(cfg)

    model.load_state_dict(ckpt['model'])
    model.cuda()

    dset.set_mode("valid")
    env = Environment(cfg, 'test', dset, shuffle=False)
    env.set_model(model)
    env.set_gpu_id(torch.cuda.current_device())

    f1_score = F1score()
    exact_match = ExactMatch()
    criterion = nn.CrossEntropyLoss()
    accs = []

    while True:
        data_idx = np.random.randint(len(dset))
        model.eval()
        env.reset(data_idx)

        print('-'*80)
        print('Data ID:', env.data_idx)
        print()

        input('\nPress enter to continue\n')

        train_data, solvable = env.observe()
        # For visualizing frame. Download frames data for this code and set path
        pic_path = '/st2/mshan/preprocessing/data/tv_qa/tv_qa_uncompressed/frames_hq'
        env.print_memory(pic_path)
        x = 'a'
        for i, entry in enumerate(train_data):
            # No batch. Compute one entry at once
            if entry.data is None:
                # Null entry...
                # But if there is none entry, can reach here if there is some None entry
                continue
            if entry.feature is None:
                # Do Mem forward and compute features (if feature is not computed yet)
                # 1 x 768(hidden_dim)

                # Video feature
                vid_feat = entry.data[0].cuda()
                # Sub feature
                if entry.data[1] is None:
                    sub_feat = torch.zeros(1, cfg.hidden_size, dtype=torch.float).cuda()
                else:
                    if cfg.model == "LRU_DNTM":
                        if i == cfg.memory_num-1:
                            sub_feat = model.q_embedding(torch.LongTensor(entry.data[1]).cuda())
                        else:
                            sub_feat = model.sub_embedding(torch.LongTensor(entry.data[1]).cuda())
                    else:
                        sub_feat = model.sub_embedding(torch.LongTensor(entry.data[1]).cuda())

                entry.feature = (vid_feat, sub_feat)

            if entry.hidden is None:
                if cfg.model == "LRU":
                    entry.hidden = torch.zeros(1, dtype=torch.float).cuda()
                else:
                    entry.hidden = torch.zeros(1, cfg.hidden_size * 2, dtype=torch.float).cuda()

        while not env.is_done():
            if x != 'c':
                x = input('\nPress c to skip through\n')

            entry = train_data[-1]
            if entry.data is None:
                # Null entry...
                # But if there is none entry, cannot reach here! (Because it will stuck on while condition)
                assert False
            if entry.feature is None:
                # Do Mem forward and compute features (if feature is not computed yet)
                # Feature is going to embedding
                # 1 x 768(hidden_dim)
                vid_feat = entry.data[0].cuda()
                # Sub feature
                if entry.data[1] is None:
                    sub_feat = torch.zeros(1, cfg.hidden_size, dtype=torch.float).cuda()
                else:
                    if cfg.model == "LRU_DNTM":
                        sub_feat = model.q_embedding(torch.LongTensor(entry.data[1]).cuda())
                        if train_data[-2].data[1] is not None:
                            train_data[-2].feature[1] = model.sub_embedding(torch.LongTensor(train_data[-2].data[1]).cuda())
                    else:
                        sub_feat = model.sub_embedding(torch.LongTensor(entry.data[1]).cuda())

                entry.feature = (vid_feat, sub_feat)

            if entry.hidden is None:
                if cfg.model == "LRU_DNTM":
                    entry.hidden = torch.zeros(1, dtype=torch.float).cuda()
                else:
                    entry.hidden = torch.zeros(1, cfg.hidden_size * 2, dtype=torch.float).cuda()

            # At here, all memory entries have feature (for spatial transformer) and hidden (for temporal GRU)
            # Stack

            modelargs = []
            input_mask = torch.ones([cfg.memory_num], dtype=torch.long).unsqueeze(0).cuda()
            vid_feature = torch.stack([entry.feature[0] for entry in train_data], 0).unsqueeze(0)
            sub_feature = torch.stack([entry.feature[1] for entry in train_data], 1)
            temporal_hidden = torch.stack([entry.hidden for entry in train_data], 1)

            modelargs.append(vid_feature)
            modelargs.append(sub_feature)
            modelargs.append(temporal_hidden)
            modelargs.append(input_mask)

            with torch.no_grad():
                if cfg.model == "T_LRU":
                    logit, value, temporal_hidden, att = model.mem_forward(*modelargs)
                else:
                    logit, value, temporal_hidden = model.mem_forward(*modelargs)
                    att = None

            # Reassigning temporal hidden
            if cfg.model == "LRU_DNTM":
                for i, entry in enumerate(train_data):
                    if i == cfg.memory_num - 1:
                        entry.hidden = torch.zeros(1, dtype=torch.float).cuda()
                    else:
                        entry.hidden = temporal_hidden[:, i]

            prob = F.softmax(logit, 1)
            log_prob = F.log_softmax(logit, 1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            #action = prob.multinomial(num_samples=1)
            _, action = prob.max(1, keepdim=True)
            log_prob = log_prob.gather(1, action)

            if x != 'c':
                env.print_memory(pic_path, prob, att=att)

            env.step(action.item())
            env.step_append()

            train_data, solvable = env.observe()

        model_in_list, targets, _ = env.qa_construct(0)
        with torch.no_grad():
            outputs = model(*model_in_list)
        if outputs.max(0)[1].item() == targets.item():
            acc = 1
        else :
            acc = 0

        env.print_memory(pic_path, answer_set=(outputs.max(0)[1].item(), targets.item()))

        print("=== QA Result ===")
        print("Prediction: %s" % outputs.max(0)[1].item())
        print("Truth: %s" % targets.item())
        print("Accuracy: %.2f" % acc)

    print("Total mean accuracy : %.2f" % (sum(accs) / len(accs)))
    print("Test instance amount : %d" % (len(accs)))
