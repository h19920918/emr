import json
import os
import shutil
import time
import random

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
from model.tvqa_abc import ABC
from model.util import create_model


from util import EMA, get_num_gpus, set_seed, ReduceScale, F1score, ExactMatch

NAIVE = ["FIFO", "LIFO", "UNIFORM"]
EMR = ["LRM_DNTM", "R_EMR", "T_EMR"]

class TensorboardWorker(mp.Process):
    def __init__(self, cfg, worker_id, queue, done, gstep):
        super().__init__(name='a3c-worker-tb')
        self.cfg = cfg
        self.worker_id = worker_id
        self.queue = queue
        self.done = done
        self.gstep = gstep

        reduce_factor = 0.999
        self.exact = ReduceScale(reduce_factor, 0)
        self.f1 = ReduceScale(reduce_factor, 0)
        self.acc = ReduceScale(reduce_factor, 0)
        self.solv = ReduceScale(reduce_factor, 0)
        self.loss = ReduceScale(reduce_factor)
        self.l2_loss = ReduceScale(reduce_factor)
        self.qa_loss = ReduceScale(reduce_factor)
        self.rl_loss = ReduceScale(reduce_factor)
        self.policy_loss = ReduceScale(reduce_factor)
        self.entropy = ReduceScale(reduce_factor)
        self.value_loss = ReduceScale(reduce_factor)
        self.grad_norm = ReduceScale(reduce_factor)
        self.reward = ReduceScale(reduce_factor)

    def run(self):
        log_dir = os.path.join(self.cfg.log_dir, 'train-tb')
        self.writer_tb = SummaryWriter(log_dir=log_dir)
        self.writer_tb.add_text('cfg', json.dumps(self.cfg.__dict__))

        with tqdm(desc=self.cfg.log_dir, initial=self.gstep.value,
                  total=self.cfg.num_episodes,
                  position=self.worker_id) as pbar:
            while self.gstep.value < self.cfg.num_episodes:
                while not self.queue.empty() and self.gstep.value < self.cfg.num_episodes:
                    item = self.queue.get()
                    step = self.gstep.value

                    exact = self.exact.update(item['exact'])
                    f1 = self.f1.update(item['f1'])
                    acc = self.acc.update(item['acc'])
                    solv = self.acc.update(item['solv'])
                    loss = self.loss.update(item['loss'])
                    l2_loss = self.l2_loss.update(item['l2_loss'])
                    qa_loss = self.qa_loss.update(item['qa_loss'])
                    rl_loss = self.rl_loss.update(item['rl_loss'])
                    policy_loss = self.policy_loss.update(item['policy_loss'])
                    entropy = self.entropy.update(item['entropy'])
                    value_loss = self.value_loss.update(item['value_loss'])
                    grad_norm = self.grad_norm.update(item['grad_norm'])
                    reward = self.reward.update(item['reward'])

                    self.writer_tb.add_scalar('exactmatch', exact, step)
                    self.writer_tb.add_scalar('f1score', f1, step)
                    self.writer_tb.add_scalar('accuracy', acc, step)
                    self.writer_tb.add_scalar('solvable', solv, step)
                    self.writer_tb.add_scalar('loss', loss, step)
                    self.writer_tb.add_scalar('l2_loss', l2_loss, step)
                    self.writer_tb.add_scalar('qa_loss', qa_loss, step)
                    self.writer_tb.add_scalar('rl_loss', rl_loss, step)
                    self.writer_tb.add_scalar('policy_loss', policy_loss, step)
                    self.writer_tb.add_scalar('entropy', entropy, step)
                    self.writer_tb.add_scalar('value_loss', value_loss, step)
                    self.writer_tb.add_scalar('grad_norm', grad_norm, step)
                    self.writer_tb.add_scalar('reward', reward, step)

                    self.gstep.value += 1
                    pbar.update()
        self.done.value = True


class WorkerBase(mp.Process):
    def __init__(self, cfg, worker_id, done, shared_model):
        super().__init__(name='a3c-worker-%02d' % (worker_id))
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.entropy_coef = cfg.entropy_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.l2_loss_coef = cfg.l2_loss_coef
        self.max_grad_norm = cfg.max_grad_norm

        self.worker_id = worker_id
        self.gpu_id = self.worker_id % get_num_gpus()
        self.seed = cfg.seed + worker_id
        self.done = done
        self.shared_model = shared_model

        self.criterion = nn.CrossEntropyLoss(size_average=False).cuda(self.gpu_id)
        self.f1_score = F1score()
        self.exact_match = ExactMatch()

        self.criterion = nn.CrossEntropyLoss()

    def sync_model(self):
        self.model.load_state_dict(self.shared_model.state_dict())

    def get_l2_loss(self):
        l2_loss = 0
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l2_loss += param.norm(2)
        return l2_loss

    def detokenize(self, tok_text):
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        return tok_text

    def get_score(self, batch, p1_pred, p2_pred, solvable):
        pred_s_idx = p1_pred
        pred_e_idx = p2_pred

        if pred_s_idx > pred_e_idx or not solvable:
            f1 = exact = 0
            return f1, exact

        f1, exact = [], []
        gt_s_idx = batch['s_idx'][0]
        gt_e_idx = batch['e_idx'][0]

        gt_txt = batch['ctx_words'][0][gt_s_idx:gt_e_idx+1].cpu()
        gt_txt = self.word_vocab.is2ws(gt_txt)
        gt_txt = ' '.join(gt_txt)

        pred_txt = batch['ctx_words'][0][pred_s_idx:pred_e_idx+1].cpu()
        pred_txt = self.word_vocab.is2ws(pred_txt)
        pred_txt = ' '.join(pred_txt)

        gt_txt = gt_txt.split()
        pred_txt = pred_txt.split()
        f1 = self.f1_score.calc_score(pred_txt, gt_txt)
        exact = self.exact_match.calc_score(pred_txt, gt_txt)

        return f1, exact


class ValidWorker(WorkerBase):
    def __init__(self, cfg, worker_id, done, shared_model, optim, env, gstep):
        super().__init__(cfg, worker_id, done, shared_model)
        self.optim = optim
        self.env = env
        self.gstep = gstep
        self.best_exact = 0
        self.num_episodes = len(self.env.dset)

    def init(self):
        log_dir = os.path.join(self.cfg.log_dir, 'valid')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text('cfg', json.dumps(self.cfg.__dict__))

    def run(self):
        # self.env.dset.load_vid()
        cfg = self.cfg
        self.init()
        set_seed(self.seed)
        # Prepare model
        self.model = create_model(cfg)
        self.model.cuda(self.gpu_id)
        self.model.eval()
        self.env.set_model(self.model)
        self.env.set_gpu_id(self.gpu_id)

        while not self.done.value:
            self.sync_model()
            step = self.gstep.value
            model_state = self.model.state_dict()
            optim_state = self.optim.state_dict()
            status = dict(exact=0,
                          f1=0,
                          acc=0,
                          solv=0,
                          loss=0,
                          l2_loss=0,
                          qa_loss=0,
                          rl_loss=0,
                          policy_loss=0,
                          entropy=0,
                          value_loss=0,
                          reward=0)
            for i in tqdm(range(self.num_episodes), desc=self.name,
                          position=self.worker_id):
                self.env.reset()
                result = self.run_episode()
                for k, v in result.items():
                    status[k] += v
            for k, v in status.items():
                status[k] /= self.num_episodes

            self._update_tensorboard(step, **status)

            self._save_checkpoint(step=step,
                                  exact=status['acc'],
                                  model_state=model_state,
                                  optim_state=optim_state)

    def _update_tensorboard(self, step, exact, f1, acc, solv, loss, l2_loss, qa_loss, rl_loss,
                            policy_loss, entropy, value_loss, reward):
        self.writer.add_scalar('exactmatch', exact, step)
        self.writer.add_scalar('f1score', f1, step)
        self.writer.add_scalar('accuracy', acc, step)
        self.writer.add_scalar('solvable', solv, step)
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('l2_loss', l2_loss, step)
        self.writer.add_scalar('qa_loss', qa_loss, step)
        self.writer.add_scalar('rl_loss', rl_loss, step)
        self.writer.add_scalar('policy_loss', policy_loss, step)
        self.writer.add_scalar('entropy', entropy, step)
        self.writer.add_scalar('value_loss', value_loss, step)
        self.writer.add_scalar('reward', reward, step)

    def _save_checkpoint(self, step, exact, model_state, optim_state):
        ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-%d.ckpt' % step)
        torch.save(dict(
            cfg=self.cfg,
            step=step,
            model=model_state,
            optim=optim_state,
        ), ckpt_path)

        if exact >= self.best_exact:
            self.best_exact = exact
            best_ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-best.ckpt')
            shutil.copyfile(ckpt_path, best_ckpt_path)
        last_ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-last.ckpt')
        shutil.copyfile(ckpt_path, last_ckpt_path)

    def run_episode(self):
        with torch.no_grad():
            log_probs = []
            values = []
            entropies = []
            rewards = []

            train_data, solvable = self.env.observe()

            for i, entry in enumerate(train_data):
                # No batch. Compute one entry at once
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
                        if self.cfg.model == "LRU":
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
                # value_list.append(value)

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

                # At here, all memory entries have feature (for spatial transformer) and hidden (for temporal GRU)
                # Stack
                modelargs = []
                input_mask = torch.ones([self.cfg.memory_num], dtype=torch.long).unsqueeze(0).cuda(self.gpu_id)
                vid_feature = torch.stack([entry.feature[0] for entry in train_data], 0).unsqueeze(0)
                sub_feature = torch.stack([entry.feature[1] for entry in train_data], 1)
                temporal_hidden = torch.stack([entry.hidden for entry in train_data], 1)

                modelargs.append(vid_feature)
                modelargs.append(sub_feature)
                modelargs.append(temporal_hidden)
                modelargs.append(input_mask)

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

                entropies.append(entropy)
                log_probs.append(log_prob)
                values.append(value)

                self.env.step(action.item())

                train_data, solvable = self.env.observe()
                if not self.env.is_done():
                    rewards.append(0)

                self.env.step_append()
                train_data, solvable = self.env.observe()


            num_sf = self.env.invest_memory()
            solvable = True
            if num_sf > 0:
                solv = 1
            else:
                solv = 0

            if solvable:
                model_in_list, targets, _ = self.env.qa_construct(self.gpu_id)
                outputs = self.model(*model_in_list)
                qa_loss = self.criterion(outputs.unsqueeze(0), targets)

                if len(values) > 0:
                    if outputs.max(0)[1].item() == targets.item():
                        acc = 1
                    else :
                        acc = 0
            else :
                qa_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                acc = 0

            solv = 1 if solvable else 0

            assert (len(values) == len(rewards) == len(log_probs) == len(entropies)), \
                   "value : %d  rewards : %d  log_probs : %d  entropies : %d" % \
                   (len(values), len(rewards), len(log_probs), len(entropies))

            if self.cfg.model in NAIVE:
                rl_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                entropy = torch.zeros(1, 1).cuda(self.gpu_id)

                l2_loss = self.get_l2_loss()
                qa_l2_loss = qa_loss + self.l2_loss_coef * l2_loss
                final_loss = qa_l2_loss

            elif self.cfg.model in EMR:
                episode_len = len(rewards)
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                policy_loss = 0
                for i in reversed(range(episode_len)):
                    policy_loss = policy_loss - rewards[i] * log_probs[i]

                if episode_len == 0:
                    rl_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                    policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                    value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                    entropy = torch.zeros(1, 1).cuda(self.gpu_id)
                else:
                    rl_loss = policy_loss
                l2_loss = self.get_l2_loss()
                qa_l2_loss = qa_loss + self.l2_loss_coef * l2_loss

                final_loss = qa_l2_loss + rl_loss


            ###

        return dict(exact=0,
                    f1=0,
                    acc=acc,
                    solv=solv,
                    loss=final_loss.item(),
                    l2_loss=l2_loss.item(),
                    qa_loss=qa_loss.item(),
                    rl_loss=rl_loss.item(),
                    policy_loss=policy_loss.item(),
                    entropy=entropy.item(),
                    value_loss=value_loss.item(),
                    reward=sum(rewards))


class TrainWorker(WorkerBase):
    def __init__(self, cfg, worker_id, done, shared_model, optim, env, queue, gstep):
        super().__init__(cfg, worker_id, done, shared_model)
        self.optim = optim
        self.env = env
        self.queue = queue
        self.gstep = gstep

    def ensure_shared_grads(self):
        for param, shared_param in zip(self.model.parameters(),
                                       self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def run(self):
        cfg = self.cfg
        set_seed(self.seed)
        # Prepare model
        self.model = create_model(cfg)
        self.model.train()
        self.env.set_gpu_id(self.gpu_id)
        self.env.set_model(self.model)

        init_approx = self.gstep.value // (self.cfg.num_workers - 1)
        total_approx = self.cfg.num_episodes // (self.cfg.num_workers - 1)
        with tqdm(desc=self.name, initial=init_approx, total=total_approx, position=self.worker_id) as pbar:
            while not self.done.value:
                self.sync_model()
                self.model.cuda(self.gpu_id)
                self.env.reset()
                self.run_episode()
                pbar.update()

    # One episode = One {Context, Question, Answer} pair
    def run_episode(self):
        log_probs = []
        values = []
        entropies = []
        rewards = []
        acc = 0

        # FIRST OBSERVE
        train_data, solvable = self.env.observe()

        ## Check solvable before start
        with torch.no_grad():
            model_in_list, targets, _ = self.env.qa_construct(self.gpu_id)
            outputs = self.model(*model_in_list)

        if outputs.max(0)[1].item() == targets.item():
            former_success = True
        else :
            former_success = False

        for i, entry in enumerate(train_data):
            # No batch. Compute one entry at once
            if entry.data is None:
                # Null entry...
                # But if there is none entry, cannot reach here! (Because it will stuck on while condition)
                # assert False
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
            # No batch. Compute one entry at once
            if entry.data is None:
                # Null entry...
                # But if there is none entry, cannot reach here! (Because it will stuck on while condition)
                assert False
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

            # At here, all memory entries have feature (for spatial transformer) and hidden (for temporal GRU)
            # Stack
            modelargs = []
            input_mask = torch.ones([self.cfg.memory_num], dtype=torch.long).unsqueeze(0).cuda(self.gpu_id)
            vid_feature = torch.stack([entry.feature[0] for entry in train_data], 0).unsqueeze(0)
            sub_feature = torch.stack([entry.feature[1] for entry in train_data], 1)
            temporal_hidden = torch.stack([entry.hidden for entry in train_data], 1)

            modelargs.append(vid_feature)
            modelargs.append(sub_feature)
            modelargs.append(temporal_hidden)
            modelargs.append(input_mask)

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
            action = prob.multinomial(num_samples=1)
            log_prob = log_prob.gather(1, action)

            entropies.append(entropy)
            log_probs.append(log_prob)
            values.append(value)

            self.env.step(action.item())

            if not self.env.is_done():
                if self.cfg.spv >= random.random():
                    with torch.no_grad():
                        model_in_list, targets, _ = self.env.qa_construct(self.gpu_id)
                        outputs = self.model(*model_in_list)
                    success = outputs.max(0)[1].item() == targets.item()

                    # TD reward
                    if former_success and not success:
                        rewards.append(-1)
                    else:
                        rewards.append(0)
                    former_success = success
                else:
                    rewards.append(0)

            self.env.step_append()
            train_data, solvable = self.env.observe()

        num_sf = self.env.invest_memory()
        solvable = True
        if num_sf > 0:
            solv = 1
        else:
            solv = 0

        # Use all the waiting memory entries.. Then Do QA
        if solvable:
            model_in_list, targets, _ = self.env.qa_construct(self.gpu_id)
            outputs = self.model(*model_in_list)
            qa_loss = self.criterion(outputs.unsqueeze(0), targets)

            if outputs.max(0)[1].item() == targets.item():
                acc = 1
            else :
                acc = 0
        else:
            qa_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            acc = 0

        assert (len(values) == len(rewards) == len(log_probs) == len(entropies)), \
               "value : %d  rewards : %d  log_probs : %d  entropies : %d" % \
               (len(values), len(rewards), len(log_probs), len(entropies))

        if self.cfg.model in NAIVE:
            rl_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            entropy = torch.zeros(1, 1).cuda(self.gpu_id)

            l2_loss = self.get_l2_loss()
            qa_l2_loss = qa_loss + self.l2_loss_coef * l2_loss
            final_loss = qa_l2_loss

        elif self.cfg.model in EMR:
            episode_len = len(rewards)
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            policy_loss = 0
            for i in reversed(range(episode_len)):
                policy_loss = policy_loss - rewards[i] * log_probs[i]

            if episode_len == 0:
                rl_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                entropy = torch.zeros(1, 1).cuda(self.gpu_id)
            else:
                rl_loss = policy_loss

            l2_loss = self.get_l2_loss()
            qa_l2_loss = qa_loss + self.l2_loss_coef * l2_loss

            final_loss = qa_l2_loss + rl_loss

        # Backprop
        self.optim.zero_grad()
        final_loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.model.cpu()
        self.ensure_shared_grads()
        self.optim.step()

        self.queue.put_nowait(dict(exact=0,
                                   f1=0,
                                   acc=acc,
                                   solv=solv,
                                   loss=final_loss.item(),
                                   l2_loss=l2_loss.item(),
                                   qa_loss=qa_loss.item(),
                                   rl_loss=rl_loss.item(),
                                   policy_loss=policy_loss.item(),
                                   entropy=entropy.item(),
                                   value_loss=value_loss.item(),
                                   grad_norm=grad_norm,
                                   reward=sum(rewards)
                                   ))
