from math import ceil
import os
import pickle
from pprint import pprint
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.nn import functional as F

from environment import create_env
from model.util import create_a3c_model
from util import get_num_gpus, set_seed
from vocab import Vocab


class TestWorker(mp.Process):
    def __init__(self, cfg, worker_id, model, env, queue):
        super().__init__(name='test-worker-%02d' % (worker_id))
        self.cfg = cfg
        self.worker_id = worker_id
        self.gpu_id = self.worker_id % get_num_gpus()
        self.model = model
        self.env = env
        self.queue = queue

        batch_size = ceil(len(self.env.dataset) / self.cfg.num_workers)
        start = batch_size*worker_id
        end = min(batch_size*(worker_id+1), len(self.env.dataset))
        self.name = self.name + f'-{start}-{end}'
        self.data_idxs = range(start, end)

    def run(self):
        self.model.eval()
        self.model.cuda(self.gpu_id)
        self.env.set_gpu_id(self.gpu_id)

        for idx in tqdm(self.data_idxs, desc=self.name, position=self.worker_id):
            self.env.reset(idx)
            accs = []
            solvs = []
            while not self.env.is_done():
                if len(self.env.memory) < self.cfg.memory_size-1:
                    self.env._append_current()
                    self.env.sent_ptr += 1

                    if self.env.is_qa_step():
                        read_output = self._qa_forward()

                        acc = (read_output['pred'] == read_output['target']).item()
                        accs.append(acc)
                        solvs.append(1.0 if self.env.check_solvable() else 0.0)
                        self.env.qa_ptr += 1
                    continue
                else:
                    self.env._append_current()
                    self.env.sent_ptr += 1

                    batch = self.env.observe()
                    batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}

                    with torch.no_grad():
                        write_output = self.model.mem_forward(**batch)
                    act_logit, value = write_output['logit'], write_output['value']

                    prob = F.softmax(act_logit, 1)
                    _, action = prob.max(1, keepdim=True)

                    self.env.step(action=action.item(), **write_output)

                    if self.env.is_qa_step():
                        with torch.no_grad():
                            read_output = self._qa_forward()

                        acc = (read_output['pred'] == read_output['target']).item()
                        accs.append(acc)
                        solvs.append(1.0 if self.env.check_solvable() else 0.0)
                        self.env.qa_ptr += 1

            assert(len(accs) == len(self.env.data.qas))
            result = dict()
            for i, (acc, solv) in enumerate(zip(accs, solvs)):
                result[f'acc_{i}'] = acc
                result[f'solv_{i}'] = solv
            result['acc'] = sum(accs) / len(accs)
            result['solv'] = sum(solvs) / len(solvs)
            if idx < 120:
                result['group_0'] = result['acc']
            elif idx < 140:
                result['group_1'] = result['acc']
            elif idx < 160:
                result['group_2'] = result['acc']
            elif idx < 180:
                result['group_3'] = result['acc']
            elif idx < 200:
                result['group_4'] = result['acc']

            self.queue.put_nowait(result)

    def _qa_forward(self):
        batch = self.env.observe()
        batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}

        with torch.no_grad():
            read_output = self.model.qa_forward(**batch)
        qa_logit = read_output['logit']

        target = batch['answ_idx']
        _, pred = qa_logit.max(1)
        return dict(pred=pred, target=target, **read_output)


def test(cfg):
    set_seed(cfg.seed)

    if not os.path.exists(cfg.ckpt):
        print('Invalid ckpt path:', cfg.ckpt)
        exit(1)
    ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)

    print(cfg.ckpt, 'loaded')
    loaded_cfg = ckpt['cfg'].__dict__
    if loaded_cfg.get('num_workers') is not None:
        del loaded_cfg['num_workers']
        del loaded_cfg['test_set']
        del loaded_cfg['pre_ckpt']

    cfg.__dict__.update(loaded_cfg)
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

    env = create_env(cfg, cfg.test_set, vocab, stats, shuffle=False)
    env.set_model(model)

    print(env.dataset.path, 'loaded')

    queue = mp.Queue()

    procs = []
    for i in range(cfg.num_workers):
        p = TestWorker(cfg, i, model, env, queue)
        p.start()
        procs.append(p)

    num_examples = len(env.dataset)
    dataset = dict()
    for _ in range(num_examples):
        example = queue.get()
        for key, val in example.items():
            dataset[key] = dataset.get(key, 0) + val

    for p in procs:
        p.join()

    from time import sleep
    sleep(3)
    print(f'All processes is finished ({num_examples} examples).')

    print()
    acc = dataset['acc'] / num_examples * 100
    solv = dataset['solv'] / num_examples * 100
    total_error = 100 - acc
    print('[Total]')
    print(f'Acc (Solv): {acc:.2f} ({solv:.2f})')
    print(f'Error : {total_error:.2f}')
    print(f'{total_error:.3f}')

    print()
    print(cfg.log_dir)

    return total_error
