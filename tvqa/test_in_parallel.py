from math import ceil, floor
import os
import pickle
from pprint import pprint
import random

import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
from torch.nn import functional as F
from tqdm import tqdm

from environment import Environment
from util import get_num_gpus, set_seed

from tvqa_dataset import TVQADataset
from model.tvqa_abc import ABC
from model.util import create_model

EMR = ["LRU_DNTM", "R_EMR", "T_EMR"]
NAIVE = ["LIFO", "FIFO", "UNIFORM"]

class TestWorker(mp.Process):
    def __init__(self, cfg, worker_id, model, env, queue):
        super().__init__(name='test-worker-%02d' % (worker_id))
        self.cfg = cfg
        self.worker_id = worker_id
        self.gpu_id = self.worker_id % get_num_gpus()
        self.model = model
        self.env = env
        self.queue = queue

        batch_size = ceil(len(self.env.dset) / self.cfg.num_workers)
        start = batch_size * worker_id
        end = min(batch_size*(worker_id+1), len(self.env.dset))
        self.data_idxs = range(start, end)

    def run(self):
        cfg = self.cfg
        self.model.eval()
        self.model.cuda(self.gpu_id)
        self.env.set_model(self.model)
        self.env.set_gpu_id(self.gpu_id)

        for idx in tqdm(self.data_idxs, desc=self.name, position=self.worker_id):
            self.env.reset(idx)

            train_data, solvable = self.env.observe()

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
                    vid_feat = entry.data[0].cuda(self.gpu_id)
                    # Sub feature
                    if entry.data[1] is None:
                        sub_feat = torch.zeros(1, self.cfg.hidden_size, dtype=torch.float).cuda(self.gpu_id)
                    else:
                        if self.cfg.model == "LRU_DNTM":
                            if i == self.cfg.memory_num-1:
                                sub_feat = self.model.q_embedding(torch.LongTensor(entry.data[1]).cuda(self.gpu_id))
                            else:
                                sub_feat = self.model.sub_embedding(torch.LongTensor(entry.data[1]).cuda(self.gpu_id))
                        else:
                            sub_feat = self.model.sub_embedding(torch.LongTensor(entry.data[1]).cuda(self.gpu_id))

                    entry.feature = (vid_feat, sub_feat)

                if entry.hidden is None:
                    if self.cfg.model == "LRU_DNTM":
                        entry.hidden = torch.zeros(1, dtype=torch.float).cuda(self.gpu_id)
                    else:
                        entry.hidden = torch.zeros(1, self.cfg.hidden_size * 2, dtype=torch.float).cuda(self.gpu_id)

            while not self.env.is_done():
                entry = train_data[-1]
                if entry.data is None:
                    # Null entry...
                    # But if there is none entry, cannot reach here! (Because it will stuck on while condition)
                    assert False
                if entry.feature is None:
                    # Do Mem forward and compute features (if feature is not computed yet)
                    # Feature is going to embedding
                    # 1 x 768(hidden_dim)
                    vid_feat = entry.data[0].cuda(self.gpu_id)
                    # Sub feature
                    if entry.data[1] is None:
                        sub_feat = torch.zeros(1, self.cfg.hidden_size, dtype=torch.float).cuda(self.gpu_id)
                    else:
                        if self.cfg.model == "LRU_DNTM":
                            sub_feat = self.model.q_embedding(torch.LongTensor(entry.data[1]).cuda(self.gpu_id))
                            if train_data[-2].data[1] is not None:
                                train_data[-2].feature[1] = self.model.sub_embedding(torch.LongTensor(train_data[-2].data[1]).cuda(self.gpu_id))
                        else:
                            sub_feat = self.model.sub_embedding(torch.LongTensor(entry.data[1]).cuda(self.gpu_id))

                    entry.feature = (vid_feat, sub_feat)

                if entry.hidden is None:
                    if self.cfg.model == "LRU_DNTM":
                        entry.hidden = torch.zeros(1, dtype=torch.float).cuda(self.gpu_id)
                    else:
                        entry.hidden = torch.zeros(1, self.cfg.hidden_size * 2, dtype=torch.float).cuda(self.gpu_id)

                # Stack
                modelargs = []
                input_mask = torch.ones([cfg.memory_num], dtype=torch.long).unsqueeze(0).cuda(self.gpu_id)
                vid_feature = torch.stack([entry.feature[0] for entry in train_data], 0).unsqueeze(0)
                sub_feature = torch.stack([entry.feature[1] for entry in train_data], 1)
                temporal_hidden = torch.stack([entry.hidden for entry in train_data], 1)

                modelargs.append(vid_feature)
                modelargs.append(sub_feature)
                modelargs.append(temporal_hidden)
                modelargs.append(input_mask)

                with torch.no_grad():
                    logit, value, temporal_hidden = self.model.mem_forward(*modelargs)

                # Reassigning temporal hidden
                if self.cfg.model == "LRU_DNTM":
                    for i, entry in enumerate(train_data):
                        if i == self.cfg.memory_num - 1:
                            entry.hidden = torch.zeros(1, dtype=torch.float).cuda(self.gpu_id)
                        else:
                            entry.hidden = temporal_hidden[:, i]

                prob = F.softmax(logit, 1)
                log_prob = F.log_softmax(logit, 1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                _, action = prob.max(1, keepdim=True)
                log_prob = log_prob.gather(1, action)

                self.env.step(action.item())
                self.env.step_append()
                train_data, solvable = self.env.observe()

            model_in_list, targets, _ = self.env.qa_construct(self.gpu_id)
            with torch.no_grad():
                outputs = self.model(*model_in_list)
            if outputs.max(0)[1].item() == targets.item():
                acc = 1
            else :
                acc = 0

            qa_id = self.env.data[9]
            pred = outputs.max(0)[1].item()
            dataset_cnt = 1
            solvable = self.env.invest_memory()
            if solvable > 0:
                solve_binary = 1
            else:
                solve_binary = 0

            self.queue.put_nowait(dict(acc=acc,
                                       dataset_cnt=dataset_cnt,
                                       solvable=solvable,
                                       solve_binary=solve_binary,
                                       qa_id=qa_id,
                                       pred=pred))


def test(cfg):
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    set_seed(cfg.seed)
    mn = cfg.memory_num
    large = cfg.large
    if not os.path.exists(cfg.ckpt):
        print('Invalid ckpt path:', cfg.ckpt)
        exit(1)
    ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
    print(cfg.ckpt, 'loaded')
    loaded_cfg = ckpt['cfg'].__dict__
    if loaded_cfg.get('num_workers') is not None:
        del loaded_cfg['num_workers']
    cfg.__dict__.update(loaded_cfg)
    cfg.model = cfg.model.upper()
    pprint(cfg.__dict__)
    cfg.memory_num = mn
    cfg.large = large

    dset = TVQADataset(cfg)
    cfg.vocab_size = len(dset.word2idx)

    model = create_model(cfg)

    model.load_state_dict(ckpt['model'])

    dset.set_mode("valid")
    print(len(dset))
    set_length = len(dset)
    env = Environment(cfg, 'test', dset, shuffle=False)
    env.set_model(model)
    env.set_gpu_id(torch.cuda.current_device())

    queue = mp.Queue()

    procs = []
    for i in range(cfg.num_workers):
        p = TestWorker(cfg, i, model, env, queue)
        p.start()
        procs.append(p)

    results = []
    for p in procs:
        while True:
            running = p.is_alive()
            if not queue.empty():
                result = queue.get()
                results.append(result)
            else:
                if not running:
                    break

    for p in procs:
        p.join()

    print('Processing duplicated factors')
    import pdb; pdb.set_trace()
    acc = 0
    total_length = 0
    solvable = 0
    solvable_success = 0
    solvable_fail = 0

    solve_binary = 0
    solve_binary_success = 0
    solve_binary_fail = 0

    for i in range(len(results)):
        acc += results[i]['acc']
        total_length += results[i]['dataset_cnt']
        solvable += results[i]['solvable']
        solve_binary += results[i]['solve_binary']

        if results[i]['acc'] == 1:
            solvable_success += results[i]['solvable']
        else:
            solvable_fail += results[i]['solvable']

        if results[i]['acc'] == 1:
            solve_binary_success += results[i]['solve_binary']
        else:
            solve_binary_fail += results[i]['solve_binary']

    # answer_dict = {}
    # for i in range(len(results)):
    #     qa_id = results[i]['qa_id']
    #     pred = results[i]['pred']
    #     answer_dict[str(qa_id)] = pred
    #
    # import json
    # with open('prediction_valid.json', 'w') as fp:
    #     json.dump(answer_dict, fp, sort_keys=True)

    print("Dump output to json")

    from time import sleep
    sleep(3)
    print('All processes is finished.')
    print('Solvable: %.2f' % (solvable / total_length * 100))
    print('Solve binary (whether at least 1 sf or not) : %.2f' % (solve_binary / total_length * 100))
    print('Accuracy: %.2f' % (acc / total_length * 100))

    print('Rate of supporting frames in memory when Success: %.2f' % (solvable_success / acc * 100))
    print('Rate of supporting frames in memory when Fail: %.2f' % (solvable_fail / (total_length - acc) * 100))

    print('Rate of memory including at least one sf when Success: %.2f' % (solve_binary_success / acc * 100))
    print('Rate of memory including at least one sf when Fail: %.2f' % (solve_binary_fail / (total_length - acc) * 100))

    # 43.65 at first try (Valid)
