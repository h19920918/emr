import json
import os
from pprint import pprint
from tqdm import tqdm

from math import ceil, floor
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from environment import Environment
from model.util import create_a3c_model
from util import get_num_gpus, get_score_from_trivia, f1_score, exact_match_score
from tokenization import BertTokenizer


class TestWorker(mp.Process):
    def __init__(self, cfg, worker_id, model, env, queue, tokenizer):
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
        self.data_idxs = range(start, end)

        self.f1_score = f1_score
        self.exact_match_score = exact_match_score

        self.tokenizer = tokenizer

    def get_score(self, batch, p1_pred, p2_pred, solvable):
        pred_s_idx = p1_pred.item()
        pred_e_idx = p2_pred.item()

        if pred_s_idx > pred_e_idx or not solvable:
            f1 = exact = 0.0
            return f1, exact

        example_id = self.env.dataset.example_ids[self.env.data_idx]
        example_id = os.path.join(self.cfg.prepro_dir, self.cfg.task, self.cfg.test_set, example_id)
        example_file = example_id + '.answ.words'
        with open(example_file, 'r') as f:
            lines = f.readlines()
        gts = []
        for line in lines:
            gts.append(line.strip())

        # gt_s_idx = batch['s_idx'].item()
        # gt_e_idx = batch['e_idx'].item()

        # gt_tokens = batch['input_ids'][0][gt_s_idx:gt_e_idx+1].cpu().data
        # gt_txt = self.tokenizer.convert_ids_to_tokens(gt_tokens.tolist())
        # gt_txt = ' '.join(gt_txt)
        # gt_txt = gt_txt.replace(' ##', '')
        # gt_txt = gt_txt.replace('##', '')

        pred_tokens = batch['input_ids'][0][pred_s_idx:pred_e_idx+1].cpu().data
        pred_txt = self.tokenizer.convert_ids_to_tokens(pred_tokens.tolist())
        pred_txt = ' '.join(pred_txt)
        pred_txt = pred_txt.replace(' ##', '')
        pred_txt = pred_txt.replace('##', '')

        f1s = []
        exacts = []
        for gt_txt in gts:
            f1 = self.f1_score(pred_txt, gt_txt)
            exact = self.exact_match_score(pred_txt, gt_txt)
            f1s.append(f1)
            if exact:
                exact = 1.0
            else:
                exact = 0.0
            exacts.append(exact)
        f1 = max(f1s)
        exact = max(exacts)
        return f1, exact

    def run(self):
        self.model.eval()
        self.model.cuda(self.gpu_id)
        self.env.set_model(self.model)
        self.env.set_gpu_id(self.gpu_id)
        actions = [0 for _ in range(self.cfg.memory_num)]

        for idx in tqdm(self.data_idxs, desc=self.name, position=self.worker_id):
            self.env.reset(idx)
            with torch.no_grad():
                while not self.env.is_done():
                    if len(self.env.memory) < self.cfg.memory_num-1:
                        self.env._append_current()
                        self.env.sent_ptr += 1
                    else:
                        if self.cfg.model in ['LIFO']:
                            actions[0] += 1
                            break
                        self.env._append_current()
                        self.env.sent_ptr += 1

                        batch, solvable, _ = self.env.observe()
                        batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}

                        result = self.model.mem_forward(**batch)
                        logit, value = result['logit'], result['value']

                        prob = F.softmax(logit, 1)
                        _, action = prob.max(1, keepdim=True)
                        self.env.step(action=action.item(), **result)
                        actions[action.item()] += 1

                batch, solvable, _ = self.env.observe(qa_step=True)
                batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}
                result = self.model.qa_forward(**batch)
                qa_loss = result['loss']
                s_idx, e_idx = batch['s_idx'], batch['e_idx']
                p1_logit, p2_logit = result['start_logits'], result['end_logits']

                batch_size, c_len = p1_logit.size()
                ls = nn.LogSoftmax(dim=1)
                mask = (torch.ones(c_len, c_len) * float('-inf')).cuda(self.gpu_id).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
                score = (ls(p1_logit).unsqueeze(2) + ls(p2_logit).unsqueeze(1)) + mask
                p1_score, p1_pred = score.max(dim=1)
                p2_score, p2_pred = p1_score.max(dim=1)
                p1_pred = torch.gather(p1_pred, 1, p2_pred.view(-1, 1)).squeeze(-1)

                score = p1_score[0][p1_pred.item()].item() + p2_score.item()
                answer_tokens = batch['input_ids'][0][p1_pred.item():p2_pred.item()+1].cpu().data
                answer = self.tokenizer.convert_ids_to_tokens(answer_tokens.tolist())
                answer = ' '.join(answer)
                answer = answer.replace(' ##', '')
                answer = answer.replace('##', '')

                if solvable:
                    p1_acc = (p1_pred == s_idx).item()
                    p2_acc = (p2_pred == e_idx).item()

                    f1, exact = self.get_score(batch, p1_pred, p2_pred, solvable)
                    acc = (p1_acc + p2_acc) / 2
                else:
                    acc = 0.0
                    p1_acc = 0.0
                    p2_acc = 0.0
                    f1 = 0.0
                    exact = 0.0

            solv = 1 if solvable else 0
            doc = self.env.dataset.example_ids[idx]

            self.queue.put_nowait(dict(exact=exact,
                                       f1=f1,
                                       score=score,
                                       doc=doc,
                                       answer=answer,
                                       solvable=solv,
                                       actions=actions))


def test(cfg):
    if cfg.ckpt is not None:
        if not os.path.exists(cfg.ckpt):
            print('Invalid ckpt path:', cfg.ckpt)
            exit(1)
        ckpt = torch.load(cfg.ckpt, map_location=lambda storage, loc: storage)
        print(cfg.ckpt, 'loaded')
        loaded_cfg = ckpt['cfg'].__dict__

        del loaded_cfg['num_workers']
        del loaded_cfg['test_set']
        del loaded_cfg['log_dir']
        del loaded_cfg['prediction_file']
        del loaded_cfg['num_episodes']
        del loaded_cfg['use_pretrain']
        del loaded_cfg['memory_num']
        del loaded_cfg['memory_len']
        del loaded_cfg['prepro_dir']
        del loaded_cfg['debug']

        cfg.__dict__.update(loaded_cfg)
        cfg.model = cfg.model.upper()

        print('Merged Config')
        pprint(cfg.__dict__)

        os.makedirs(cfg.log_dir)

        model = create_a3c_model(cfg)
        model.load_state_dict(ckpt['model'])
    else:
        os.makedirs(cfg.log_dir)
        model = create_a3c_model(cfg)

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

        load(model, prefix='')
        print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)

    env = Environment(cfg, cfg.test_set, tokenizer, shuffle=False)
    print(env.dataset.path, 'loaded')

    queue = mp.Queue()

    procs = []
    for i in range(cfg.num_workers):
        p = TestWorker(cfg, i, model, env, queue, tokenizer)
        if cfg.debug:
            p.run()
        else:
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

    exact_list = []
    f1_list = []
    full_action = [0 for _ in range(cfg.memory_num)]
    full_solvable = []
    id_list = []
    for i in range(len(results)):
        id_list.append(results[i]['doc'])
        full_solvable.append(results[i]['solvable'])
        exact_list.append(results[i]['exact'])
        f1_list.append(results[i]['f1'])
        for j in range(cfg.memory_num):
            full_action[j] += results[i]['actions'][j]
    qa_list = list(set(['_'.join(doc_id.split('_')[:-1]) for doc_id in id_list]))
    answers = dict()
    for qa_id in qa_list:
        answers[qa_id] = ('', -100000000)

    for i in range(len(results)):
        qa_id = '_'.join(id_list[i].split('_')[:-1])
        score = results[i]['score']
        answer = results[i]['answer']

        if answers[qa_id][1] < score:
            answers[qa_id] = (answer, score)

    for qa_id in answers.keys():
        answers[qa_id] = answers[qa_id][0]

    key_list = list(set(answers.keys()))
    solvables = [[] for i in range(len(key_list))]
    for i in range(len(full_solvable)):
        id_ = '_'.join(id_list[i].split('_')[:-1])
        solv = full_solvable[i]
        idx = key_list.index(id_)
        solvables[idx].append(solv)

    for i in range(len(solvables)):
        if 1 in solvables[i]:
            solvables[i] = 1
        else:
            solvables[i] = 0

    with open(cfg.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)
    results = get_score_from_trivia(cfg, cfg.test_set)
    exact = results['exact_match']
    f1 = results['f1']

    total_action_num = 0
    for i in range(cfg.memory_num):
        total_action_num += full_action[i]
    avg_action = [0 for _ in range(cfg.memory_num)]
    for i in range(cfg.memory_num):
        avg_action[i] += full_action[i] / total_action_num
    print('All processes is finished.')
    print('ExactMatch: %.2f' % (sum(exact_list) / len(exact_list) * 100))
    print('F1score: %.2f' % (sum(f1_list) / len(f1_list) * 100))
    print()
    print('ExactMatch: %.2f' % (exact * 100))
    print('F1score: %.2f' % (f1 * 100))
    print()
    print('Solvables: %.2f' % (sum(full_solvable) / len(full_solvable) * 100))
    print('Non duplicated Solvables: %.2f' % (sum(solvables) / len(solvables) * 100))
    print()
    print('Total number of actions: %d' % (total_action_num))
    for i in range(cfg.memory_num):
        print('Action %d : %.2f' % (i, avg_action[i] * 100))
