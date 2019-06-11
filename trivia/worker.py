import json
import numpy as np
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_

from util import get_num_gpus, set_seed, ReduceScale, warmup_linear, get_score_from_trivia, f1_score, exact_match_score
from model.util import create_a3c_model


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
        self.qa_loss = ReduceScale(reduce_factor)
        self.rl_loss = ReduceScale(reduce_factor)
        self.policy_loss = ReduceScale(reduce_factor)
        self.entropy = ReduceScale(reduce_factor)
        self.value_loss = ReduceScale(reduce_factor)
        self.grad_norm = ReduceScale(reduce_factor)
        self.reward = ReduceScale(reduce_factor)

    def run(self):
        log_dir = os.path.join(self.cfg.log_dir, 'train-tb')
        self.writer_tb = SummaryWriter(logdir=log_dir)
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
                    solv = self.solv.update(item['solv'])
                    loss = self.loss.update(item['loss'])
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
    def __init__(self, cfg, worker_id, done, shared_model, tokenizer):
        super().__init__(name='a3c-worker-%02d' % (worker_id))
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.entropy_coef = cfg.entropy_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.qa_loss_coef = cfg.qa_loss_coef
        self.rl_loss_coef = cfg.rl_loss_coef
        self.max_grad_norm = cfg.max_grad_norm

        self.worker_id = worker_id
        self.gpu_id = self.worker_id % get_num_gpus()
        self.seed = cfg.seed + worker_id
        self.done = done
        self.shared_model = shared_model
        self.tokenizer = tokenizer

        self.f1_score = f1_score
        self.exact_match_score = exact_match_score

    def sync_model(self):
        self.model.load_state_dict(self.shared_model.state_dict())

    def get_score(self, batch, p1_pred, p2_pred, solvable):
        pred_s_idx = p1_pred.item()
        pred_e_idx = p2_pred.item()

        if pred_s_idx > pred_e_idx or not solvable:
            f1 = exact = 0.0
            return f1, exact

        example_id = self.env.dataset.example_ids[self.env.data_idx]
        if self.worker_id == 0:
            example_id = os.path.join(self.cfg.prepro_dir, self.cfg.task, self.cfg.valid_set, example_id)
        else:
            example_id = os.path.join(self.cfg.prepro_dir, self.cfg.task, 'train', example_id)
        example_file = example_id + '.answ.words'
        with open(example_file, 'r') as f:
            lines = f.readlines()
        gts = []
        for line in lines:
            gts.append(line.strip())

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

    def _qa_forward(self):
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
            qa_loss = torch.tensor([0.0], requires_grad=True).cuda(self.gpu_id)
        return dict(p1_pred=p1_pred, p2_pred=p2_pred, acc=acc, qa_loss=qa_loss,
                    solvable=solvable, f1=f1, exact=exact, score=score, answer=answer)


class ValidWorker(WorkerBase):
    def __init__(self, cfg, worker_id, done, shared_model, optim, tokenizer, env, gstep):
        super().__init__(cfg, worker_id, done, shared_model, tokenizer)
        self.optim = optim
        self.env = env
        self.gstep = gstep
        self.best_exact = 0.0
        self.best_f1 = 0.0
        self.num_episodes = len(self.env.dataset)

    def init(self):
        log_dir = os.path.join(self.cfg.log_dir, 'valid')
        self.writer = SummaryWriter(logdir=log_dir)
        self.writer.add_text('cfg', json.dumps(self.cfg.__dict__))

    def run(self):
        self.init()
        set_seed(self.seed)
        self.model = create_a3c_model(self.cfg)
        self.model.cuda(self.gpu_id)
        self.model.eval()
        self.env.set_model(self.model)
        self.env.set_gpu_id(self.gpu_id)

        while not self.done.value:
            self.sync_model()
            step = self.gstep.value
            model_state = self.model.state_dict()
            optim_state = self.optim.state_dict()
            self.id_list = self.env.dataset.example_ids
            self.qa_list = list(set(['_'.join(doc_id.split('_')[:-1]) for doc_id in self.id_list]))
            self.answers = dict()
            for qa_id in self.qa_list:
                self.answers[qa_id] = ('', -100000000)
            status = dict(exact=0.0,
                          f1=0.0,
                          acc=0.0,
                          solv=0.0,
                          loss=0.0,
                          qa_loss=0.0,
                          rl_loss=0.0,
                          policy_loss=0.0,
                          entropy=0.0,
                          value_loss=0.0,
                          reward=0.0,
                          score=[],
                          answer=[])
            for i in tqdm(range(self.num_episodes), desc=self.name,
                          position=self.worker_id):
                self.env.reset(i)
                result = self.run_episode()
                for k, v in result.items():
                    if k == 'answer':
                        status[k].append(v)
                    elif k == 'score':
                        status[k].append(v)
                    else:
                        status[k] += v
            for k, v in status.items():
                if k == 'answer' or k == 'score':
                    continue
                else:
                    status[k] /= self.num_episodes

            for i in range(self.num_episodes):
                qa_id = '_'.join(self.id_list[i].split('_')[:-1])
                score = status['score'][i]
                answer = status['answer'][i]
                if self.answers[qa_id][1] < score:
                    self.answers[qa_id] = (answer, score)

            for qa_id in self.answers.keys():
                self.answers[qa_id] = self.answers[qa_id][0]

            with open(self.cfg.prediction_file, 'w', encoding='utf-8') as f:
                print(json.dumps(self.answers), file=f)
            results = get_score_from_trivia(self.cfg, self.cfg.valid_set)
            exact = results['exact_match']
            f1 = results['f1']
            status['exact'] = exact
            status['f1'] = f1

            self._update_tensorboard(step, **status)

            self._save_checkpoint(step=step,
                                  exact=status['exact'],
                                  f1=status['f1'],
                                  model_state=model_state,
                                  optim_state=optim_state)

            self.answers = dict()
            for qa_id in self.qa_list:
                self.answers[qa_id] = ('', -100000000)

    def _update_tensorboard(self, step, exact, f1, acc, solv, loss, qa_loss, rl_loss,
                            policy_loss, entropy, value_loss, reward, answer, score):
        self.writer.add_scalar('exactmatch', exact, step)
        self.writer.add_scalar('f1score', f1, step)
        self.writer.add_scalar('accuracy', acc, step)
        self.writer.add_scalar('solvable', solv, step)
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('qa_loss', qa_loss, step)
        self.writer.add_scalar('rl_loss', rl_loss, step)
        self.writer.add_scalar('policy_loss', policy_loss, step)
        self.writer.add_scalar('entropy', entropy, step)
        self.writer.add_scalar('value_loss', value_loss, step)
        self.writer.add_scalar('reward', reward, step)

    def _save_checkpoint(self, step, exact, f1, model_state, optim_state):
        if exact > self.best_exact and f1 > self.best_f1:
            self.best_exact = exact
            self.best_f1 = f1
            best_ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-best.ckpt')
            torch.save(dict(
                cfg=self.cfg,
                step=step,
                model=model_state,
                optim=optim_state,
            ), best_ckpt_path)
        last_ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-last.ckpt')
        torch.save(dict(
            cfg=self.cfg,
            step=step,
            model=model_state,
            optim=optim_state,
        ), last_ckpt_path)

    def run_episode(self):
        with torch.no_grad():
            log_probs = []
            values = []
            entropies = []
            rewards = []
            while not self.env.is_done():
                if len(self.env.memory) < self.cfg.memory_num-1:
                    self.env._append_current()
                    self.env.sent_ptr += 1
                else:
                    if self.cfg.model == 'LIFO':
                        break
                    self.env._append_current()
                    self.env.sent_ptr += 1

                    batch, solvable, _ = self.env.observe()
                    batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}

                    result = self.model.mem_forward(**batch)
                    logit, value = result['logit'], result['value']

                    prob = F.softmax(logit, 1)
                    log_prob = F.log_softmax(logit, 1)
                    entropy = -(log_prob * prob).sum(1, keepdim=True)
                    entropies.append(entropy)
                    _, action = prob.max(1, keepdim=True)

                    log_prob = log_prob.gather(1, action)
                    log_probs.append(log_prob)
                    values.append(value)

                    self.env.step(action=action.item(), **result)

                    result = self._qa_forward()
                    acc, qa_loss = result['acc'], result['qa_loss']
                    solv = 1.0 if result['solvable'] else 0.0
                    exact, f1 = result['exact'], result['f1']
                    if self.cfg.rl_method == 'discrete':
                        rewards.append(solv)
                    else:
                        rewards.append(f1)

            assert(len(self.env.memory) <= self.cfg.memory_num)
            result = self._qa_forward()
            acc, qa_loss = result['acc'], result['qa_loss']
            exact, f1 = result['exact'], result['f1']
            solv = 1.0 if result['solvable'] else 0.0

            assert(len(values) == len(rewards) == len(log_probs) == len(entropies))

            episode_len = len(rewards)
            if episode_len == 0:
                rl_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                entropy = torch.zeros(1, 1).cuda(self.gpu_id)
            else:
                R = torch.zeros(1, 1)
                values.append(R.cuda(self.gpu_id))
                R = R.cuda(self.gpu_id)
                if self.cfg.rl_method == 'discrete':
                    policy_loss = 0.0
                    value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                    for i in reversed(range(episode_len)):
                        policy_loss = policy_loss - log_probs[i] * rewards[i]
                elif self.cfg.rl_method == 'policy':
                    standardized_rewards = []
                    for r in rewards[::-1]:
                        R = self.gamma * R + r
                        standardized_rewards.insert(0, R.item())
                    standardized_rewards = np.array(standardized_rewards)
                    standardized_rewards = (standardized_rewards - standardized_rewards.mean()) / (standardized_rewards.std() + np.finfo(np.float32).eps)
                    policy_loss = 0.0
                    value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                    for i in reversed(range(episode_len)):
                        policy_loss = policy_loss - log_probs[i] * standardized_rewards[i]
                elif self.cfg.rl_method == 'a3c':
                    sigma_reward = sum(rewards)
                    if sigma_reward > 0:
                        normalized_rewards = [reward / sigma_reward for reward in rewards]
                    else:
                        normalized_rewards = rewards
                    policy_loss = 0.0
                    value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                    gae = torch.zeros(1, 1).cuda(self.gpu_id)
                    for i in reversed(range(episode_len)):
                        R = self.gamma * R + normalized_rewards[i]
                        advantage = R - values[i]
                        value_loss = value_loss + 0.5 * advantage.pow(2)

                        # Generalized Advantage Estimataion
                        delta_t = normalized_rewards[i] + self.gamma * values[i+1] - values[i]
                        gae = gae * self.gamma * self.tau + delta_t

                        policy_loss = policy_loss - gae.detach() * log_probs[i] - self.entropy_coef * entropies[i]
                else:
                    print('Need to set reinforcement learning method')
                    exit(1)
            rl_loss = policy_loss + self.value_loss_coef * value_loss
            if self.cfg.model in ['FIFO', 'LIFO']:
                loss = qa_loss
            else:
                loss = self.qa_loss_coef * qa_loss + self.rl_loss_coef * rl_loss
        return dict(exact=exact,
                    f1=f1,
                    acc=acc,
                    solv=solv,
                    loss=loss.item(),
                    qa_loss=qa_loss.item(),
                    rl_loss=rl_loss.item(),
                    policy_loss=policy_loss.item(),
                    entropy=entropy.item(),
                    value_loss=value_loss.item(),
                    reward=sum(rewards),
                    score=result['score'],
                    answer=result['answer'])


class TrainWorker(WorkerBase):
    def __init__(self, cfg, worker_id, done, shared_model, optim, tokenizer, env, queue, gstep):
        super().__init__(cfg, worker_id, done, shared_model, tokenizer)
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
        set_seed(self.seed)
        self.model = create_a3c_model(self.cfg)
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

    def run_episode(self):
        tb_exact = []
        tb_f1 = []
        tb_acc = []
        tb_solv = []
        tb_loss = []
        tb_qa_loss = []
        tb_rl_loss = []
        tb_policy_loss = []
        tb_entropy = []
        tb_value_loss = []
        tb_grad_norm = []
        tb_rewards = []

        log_probs = []
        values = []
        entropies = []
        rewards = []
        episode_step = 0
        while not self.env.is_done():
            if len(self.env.memory) < self.cfg.memory_num-1:
                self.env._append_current()
                self.env.sent_ptr += 1
            else:
                if self.cfg.model == 'LIFO':
                    break
                self.env._append_current()
                self.env.sent_ptr += 1

                batch, solvable, _ = self.env.observe()
                batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}

                result = self.model.mem_forward(**batch)
                logit, value = result['logit'], result['value']

                prob = F.softmax(logit, 1)
                log_prob = F.log_softmax(logit, 1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)
                action = prob.multinomial(num_samples=1).detach()

                log_prob = log_prob.gather(1, action)
                log_probs.append(log_prob)
                values.append(value)

                self.env.step(action=action.item(), **result)

                # FIFO
                if self.cfg.model in ['FIFO']:
                    with torch.no_grad():
                        result = self._qa_forward()
                        acc, qa_loss = result['acc'], result['qa_loss']
                    exact, f1 = result['exact'], result['f1']
                    rewards.append(f1)
                    continue

                episode_step += 1
                if episode_step == self.cfg.num_steps:
                    assert(len(self.env.memory) <= self.cfg.memory_num)
                    with torch.no_grad():
                        result = self._qa_forward()
                        acc, qa_loss = result['acc'], result['qa_loss']
                    solv = 1.0 if result['solvable'] else 0.0
                    exact, f1 = result['exact'], result['f1']
                    if self.cfg.rl_method == 'discrete':
                        rewards.append(solv)
                    else:
                        rewards.append(f1)

                    assert(len(values) == len(rewards) == len(log_probs) == len(entropies))
                    episode_len = len(rewards)
                    R = torch.zeros(1, 1)
                    values.append(R.cuda(self.gpu_id))
                    R = R.cuda(self.gpu_id)
                    if self.cfg.rl_method == 'discrete':
                        policy_loss = 0.0
                        value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                        for i in reversed(range(episode_len)):
                            policy_loss = policy_loss - log_probs[i] * rewards[i]
                    elif self.cfg.rl_method == 'policy':
                        standardized_rewards = []
                        for r in rewards[::-1]:
                            R = self.gamma * R + r
                            standardized_rewards.insert(0, R.item())
                        standardized_rewards = np.array(standardized_rewards)
                        standardized_rewards = (standardized_rewards - standardized_rewards.mean()) / (standardized_rewards.std() + np.finfo(np.float32).eps)
                        policy_loss = 0.0
                        value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                        for i in reversed(range(episode_len)):
                            policy_loss = policy_loss - log_probs[i] * standardized_rewards[i]
                    elif self.cfg.rl_method == 'a3c':
                        sigma_reward = sum(rewards)
                        if sigma_reward > 0:
                            normalized_rewards = [reward / sigma_reward for reward in rewards]
                        else:
                            normalized_rewards = rewards
                        policy_loss = 0.0
                        value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                        gae = torch.zeros(1, 1).cuda(self.gpu_id)
                        for i in reversed(range(episode_len)):
                            R = self.gamma * R + normalized_rewards[i]
                            advantage = R - values[i]
                            value_loss = value_loss + 0.5 * advantage.pow(2)

                            # Generalized Advantage Estimataion
                            delta_t = normalized_rewards[i] + self.gamma * values[i+1] - values[i]
                            gae = gae * self.gamma * self.tau + delta_t

                            policy_loss = policy_loss - gae.detach() * log_probs[i] - self.entropy_coef * entropies[i]
                    else:
                        print('Need to set reinforcement learning method')
                        exit(1)
                    rl_loss = policy_loss + self.value_loss_coef * value_loss
                    # loss = self.qa_loss_coef * qa_loss + self.rl_loss_coef * rl_loss
                    loss = self.rl_loss_coef * rl_loss

                    # Backprop
                    self.optim.zero_grad()
                    loss.backward()
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.model.cpu()
                    self.ensure_shared_grads()
                    self.optim.step()

                    tb_exact.append(exact)
                    tb_f1.append(f1)
                    tb_acc.append(acc)
                    tb_solv.append(solv)
                    tb_loss.append(loss.item())
                    tb_qa_loss.append(qa_loss.item())
                    tb_rl_loss.append(rl_loss.item())
                    tb_policy_loss.append(policy_loss.item())
                    tb_entropy.append(entropy.item())
                    tb_value_loss.append(value_loss.item())
                    tb_grad_norm.append(grad_norm)
                    tb_rewards.append(sum(rewards))

                    log_probs = []
                    values = []
                    entropies = []
                    rewards = []
                    episode_step = 0
                    self.sync_model()
                    self.model.cuda(self.gpu_id)
                    self.env._memory_reset()
                else:
                    with torch.no_grad():
                        result = self._qa_forward()
                        acc, qa_loss = result['acc'], result['qa_loss']
                    solv = 1.0 if result['solvable'] else 0.0
                    exact, f1 = result['exact'], result['f1']
                    if self.cfg.rl_method == 'discrete':
                        rewards.append(solv)
                    else:
                        rewards.append(f1)

        assert(len(self.env.memory) <= self.cfg.memory_num)
        result = self._qa_forward()
        acc, qa_loss = result['acc'], result['qa_loss']
        exact, f1 = result['exact'], result['f1']
        solv = 1.0 if result['solvable'] else 0.0

        assert(len(values) == len(rewards) == len(log_probs) == len(entropies))

        episode_len = len(rewards)
        if episode_len == 0:
            rl_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            entropy = torch.zeros(1, 1).cuda(self.gpu_id)
            tb_rewards.append(f1)
        else:
            R = torch.zeros(1, 1)
            values.append(R.cuda(self.gpu_id))
            R = R.cuda(self.gpu_id)
            if self.cfg.rl_method == 'discrete':
                policy_loss = 0.0
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                for i in reversed(range(episode_len)):
                    policy_loss = policy_loss - log_probs[i] * rewards[i]
            elif self.cfg.rl_method == 'policy':
                standardized_rewards = []
                for r in rewards[::-1]:
                    R = self.gamma * R + r
                    standardized_rewards.insert(0, R.item())
                standardized_rewards = np.array(standardized_rewards)
                standardized_rewards = (standardized_rewards - standardized_rewards.mean()) / (standardized_rewards.std() + np.finfo(np.float32).eps)
                policy_loss = 0.0
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                for i in reversed(range(episode_len)):
                    policy_loss = policy_loss - log_probs[i] * standardized_rewards[i]
            elif self.cfg.rl_method == 'a3c':
                sigma_reward = sum(rewards)
                if sigma_reward > 0:
                    normalized_rewards = [reward / sigma_reward for reward in rewards]
                else:
                    normalized_rewards = rewards
                policy_loss = 0.0
                value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
                gae = torch.zeros(1, 1).cuda(self.gpu_id)
                for i in reversed(range(episode_len)):
                    R = self.gamma * R + normalized_rewards[i]
                    advantage = R - values[i]
                    value_loss = value_loss + 0.5 * advantage.pow(2)

                    # Generalized Advantage Estimataion
                    delta_t = normalized_rewards[i] + self.gamma * values[i+1] - values[i]
                    gae = gae * self.gamma * self.tau + delta_t

                    policy_loss = policy_loss - gae.detach() * log_probs[i] - self.entropy_coef * entropies[i]
            else:
                print('Need to set reinforcement learning method')
                exit(1)
        rl_loss = policy_loss + self.value_loss_coef * value_loss
        if self.cfg.model in ['FIFO', 'LIFO']:
            loss = qa_loss
        else:
            loss = self.qa_loss_coef * qa_loss + self.rl_loss_coef * rl_loss

        # Backprop
        self.optim.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.model.cpu()
        self.ensure_shared_grads()
        self.optim.step()

        tb_exact.append(exact)
        tb_f1.append(f1)
        tb_acc.append(acc)
        tb_solv.append(solv)
        tb_loss.append(loss.item())
        tb_qa_loss.append(qa_loss.item())
        tb_rl_loss.append(rl_loss.item())
        tb_policy_loss.append(policy_loss.item())
        tb_entropy.append(entropy.item())
        tb_value_loss.append(value_loss.item())
        tb_grad_norm.append(grad_norm)
        tb_rewards.append(sum(rewards))

        self.queue.put_nowait(dict(exact=sum(tb_exact)/len(tb_exact),
                                   f1=sum(tb_f1)/len(tb_f1),
                                   acc=sum(tb_acc)/len(tb_acc),
                                   solv=sum(tb_solv)/len(tb_solv),
                                   loss=sum(tb_loss)/len(tb_loss),
                                   qa_loss=sum(tb_qa_loss)/len(tb_qa_loss),
                                   rl_loss=sum(tb_rl_loss)/len(tb_rl_loss),
                                   policy_loss=sum(tb_policy_loss)/len(tb_policy_loss),
                                   entropy=sum(tb_entropy)/len(tb_entropy),
                                   value_loss=sum(tb_value_loss)/len(tb_value_loss),
                                   grad_norm=sum(tb_grad_norm)/len(tb_grad_norm),
                                   reward=sum(tb_rewards)/len(tb_rewards),
                                   ))
