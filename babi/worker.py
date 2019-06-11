import json
import os
import shutil
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from model.util import create_a3c_model
from util import EMA, get_num_gpus, set_seed


class TensorboardWorker(mp.Process):
    def __init__(self, cfg, worker_id, queue, done, gstep):
        super().__init__(name='a3c-worker-tb')
        self.cfg = cfg
        self.worker_id = worker_id
        self.queue = queue
        self.done = done
        self.gstep = gstep

        ema_factor = 0.999
        self.acc = EMA(ema_factor)
        self.solv = EMA(ema_factor)
        self.loss = EMA(ema_factor)
        self.l2_loss = EMA(ema_factor)
        self.qa_loss = EMA(ema_factor)
        self.rl_loss = EMA(ema_factor)
        self.policy_loss = EMA(ema_factor)
        self.entropy = EMA(ema_factor)
        self.value_loss = EMA(ema_factor)
        self.grad_norm = EMA(ema_factor)

    def run(self):
        log_dir = os.path.join(self.cfg.log_dir, 'train-ema')
        self.writer_ema = SummaryWriter(logdir=log_dir)
        self.writer_ema.add_text('cfg', json.dumps(self.cfg.__dict__))

        with tqdm(desc=self.cfg.log_dir,
                  initial=self.gstep.value, total=self.cfg.num_episodes,
                  position=self.worker_id) as pbar:
            while self.gstep.value < self.cfg.num_episodes:
                while not self.queue.empty() and self.gstep.value < self.cfg.num_episodes:
                    item = self.queue.get()
                    step = self.gstep.value

                    acc = self.acc.update(item['acc'])
                    solv = self.solv.update(item['solv'])
                    loss = self.loss.update(item['loss'])
                    qa_loss = self.qa_loss.update(item['qa_loss'])
                    rl_loss = self.rl_loss.update(item['rl_loss'])
                    policy_loss = self.policy_loss.update(item['policy_loss'])
                    entropy = self.entropy.update(item['entropy'])
                    value_loss = self.value_loss.update(item['value_loss'])
                    grad_norm = self.grad_norm.update(item['grad_norm'])

                    self.writer_ema.add_scalar('accuracy', acc, step)
                    self.writer_ema.add_scalar('solvable', solv, step)
                    self.writer_ema.add_scalar('loss', loss, step)
                    self.writer_ema.add_scalar('qa_loss', qa_loss, step)
                    self.writer_ema.add_scalar('rl_loss', rl_loss, step)
                    self.writer_ema.add_scalar('policy_loss', policy_loss, step)
                    self.writer_ema.add_scalar('entropy', entropy, step)
                    self.writer_ema.add_scalar('value_loss', value_loss, step)
                    self.writer_ema.add_scalar('grad_norm', grad_norm, step)

                    self.gstep.value += 1
                    pbar.update()
        self.done.value = True


class WorkerBase(mp.Process):
    def __init__(self, cfg, worker_id, done, shared_model, vocab, stats):
        super().__init__(name='a3c-worker-%02d' % (worker_id))
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.entropy_coef = cfg.entropy_coef
        self.value_loss_coef = cfg.value_loss_coef
        self.qa_loss_coef = cfg.qa_loss_coef
        self.max_grad_norm = cfg.max_grad_norm

        self.worker_id = worker_id
        self.gpu_id = self.worker_id % get_num_gpus()
        self.seed = cfg.seed + worker_id
        self.done = done
        self.shared_model = shared_model
        self.vocab = vocab
        self.stats = stats

    def sync_model(self):
        self.model.load_state_dict(self.shared_model.state_dict())

    def _qa_forward(self, train):
        batch = self.env.observe()
        batch = {k: v.cuda(self.gpu_id) for k, v in batch.items()}

        if train:
            read_output = self.model.qa_forward(**batch)
        else:
            with torch.no_grad():
                read_output = self.model.qa_forward(**batch)
        qa_logit = read_output['logit']

        target = batch['answ_idx']
        _, pred = qa_logit.max(1)

        nll_loss = F.nll_loss(F.log_softmax(qa_logit, 1), target.squeeze(0))
        return dict(pred=pred, loss=nll_loss, target=target, **read_output)


class ValidWorker(WorkerBase):
    def __init__(self, cfg, worker_id, done, shared_model, optim, vocab, stats, env, gstep):
        super().__init__(cfg, worker_id, done, shared_model, vocab, stats)
        self.optim = optim
        self.env = env
        self.gstep = gstep
        self.best_acc = 0.0
        if cfg.num_valid_episodes == 0:
            self.num_episodes = len(self.env.dataset)
        else:
            self.num_episodes = cfg.num_valid_episodes

    def init(self):
        log_dir = os.path.join(self.cfg.log_dir, 'valid')
        self.writer = SummaryWriter(logdir=log_dir)
        self.writer.add_text('cfg', json.dumps(self.cfg.__dict__))

    def run(self):
        self.init()
        set_seed(self.seed)
        self.model = create_a3c_model(self.cfg, self.vocab, self.stats)
        self.model.cuda(self.gpu_id)
        self.model.eval()
        self.env.set_model(self.model)
        self.env.set_gpu_id(self.gpu_id)

        while not self.done.value:
            self.sync_model()
            step = self.gstep.value
            model_state = self.model.state_dict()
            optim_state = self.optim.state_dict()
            status = dict(acc=0.0,
                          solv=0.0,
                          loss=0.0,
                          qa_loss=0.0,
                          rl_loss=0.0,
                          policy_loss=0.0,
                          entropy=0.0,
                          value_loss=0.0)
            for idx in tqdm(range(self.num_episodes), desc=self.name,
                          position=self.worker_id):
                self.env.reset(idx)
                result = self.run_episode()
                for k, v in result.items():
                    status[k] += v
            for k, v in status.items():
                status[k] /= self.num_episodes

            self._update_tensorboard(step, **status)

            self._save_checkpoint(step=step,
                                  acc=status['acc'],
                                  model_state=model_state,
                                  optim_state=optim_state)

    def _update_tensorboard(self, step, acc, solv, loss, qa_loss, rl_loss,
                            policy_loss, entropy, value_loss):
        self.writer.add_scalar('accuracy', acc, step)
        self.writer.add_scalar('solvable', solv, step)
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('qa_loss', qa_loss, step)
        self.writer.add_scalar('rl_loss', rl_loss, step)
        self.writer.add_scalar('policy_loss', policy_loss, step)
        self.writer.add_scalar('entropy', entropy, step)
        self.writer.add_scalar('value_loss', value_loss, step)

    def _save_checkpoint(self, step, acc, model_state, optim_state):
        ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-last.ckpt')
        torch.save(dict(
            cfg=self.cfg,
            step=step,
            model=model_state,
            optim=optim_state,
        ), ckpt_path)

        if acc >= self.best_acc:
            self.best_acc = acc
            best_ckpt_path = os.path.join(self.cfg.log_dir, 'ckpt', 'model-best.ckpt')
            shutil.copyfile(ckpt_path, best_ckpt_path)

    def run_episode(self):
        log_probs = []
        values = []
        entropies = []
        rewards = []
        accs = []
        solvs = []
        qa_loss = 0.0

        while not self.env.is_done():
            if len(self.env.memory) < self.cfg.memory_size-1:
                self.env._append_current()
                self.env.sent_ptr += 1

                if self.env.is_qa_step():
                    read_output = self._qa_forward(train=False)

                    qa_loss = qa_loss + read_output['loss']
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
                log_prob = F.log_softmax(act_logit, 1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)
                _, action = prob.max(1, keepdim=True)

                log_prob = log_prob.gather(1, action).cuda(self.gpu_id)
                log_probs.append(log_prob)
                values.append(value)
                if not self.env.is_qa_step():
                    rewards.append(1.0 if self.env.check_solvable() else 0.0)

                self.env.step(action=action.item(), **write_output)

                if self.env.is_qa_step():
                    with torch.no_grad():
                        read_output = self._qa_forward(train=False)

                    qa_loss = qa_loss + read_output['loss']
                    acc = (read_output['pred'] == read_output['target']).item()
                    accs.append(acc)
                    solvs.append(1.0 if self.env.check_solvable() else 0.0)
                    rewards.append(1.0 if self.env.check_solvable() else 0.0)
                    self.env.qa_ptr += 1

        assert(len(accs) == len(self.env.data.qas))
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

            policy_loss = 0.0
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            gae = torch.zeros(1, 1).cuda(self.gpu_id)
            for i in reversed(range(episode_len)):
                R = self.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                delta_t = rewards[i] + self.gamma * values[i+1] - values[i]
                gae = gae * self.gamma * self.tau + delta_t

                policy_loss = policy_loss - gae.detach() * log_probs[i] - self.entropy_coef * entropies[i]

        if self.cfg.model in ['FIFO', 'LIFO', 'UNIFORM']:
            policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            entropy = torch.zeros(1, 1).cuda(self.gpu_id)

        rl_loss = policy_loss + self.value_loss_coef * value_loss
        loss = self.qa_loss_coef * qa_loss + rl_loss

        return dict(acc=sum(accs) / len(accs),
                    solv=sum(solvs) / len(solvs),
                    loss=loss.item(),
                    qa_loss=qa_loss.item(),
                    rl_loss=rl_loss.item(),
                    policy_loss=policy_loss.item(),
                    entropy=entropy.item(),
                    value_loss=value_loss.item())


class TrainWorker(WorkerBase):
    def __init__(self, cfg, worker_id, done, shared_model, optim, vocab, stats, env, queue, gstep):
        super().__init__(cfg, worker_id, done, shared_model, vocab, stats)
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
        self.model = create_a3c_model(self.cfg, self.vocab, self.stats)
        self.model.train()
        self.env.set_gpu_id(self.gpu_id)
        self.env.set_model(self.model)

        init_approx = self.gstep.value // (self.cfg.num_workers - 1)
        total_approx = self.cfg.num_episodes // (self.cfg.num_workers - 1)
        with tqdm(desc=self.name, initial=init_approx, total=total_approx,
                  position=self.worker_id) as pbar:
            while not self.done.value:
                self.sync_model()
                self.model.cuda(self.gpu_id)
                self.env.reset()
                self.run_episode()
                pbar.update()

    def run_episode(self):
        log_probs = []
        values = []
        entropies = []
        rewards = []
        accs = []
        solvs = []
        qa_loss = 0.0

        while not self.env.is_done():
            if len(self.env.memory) < self.cfg.memory_size-1:
                self.env._append_current()
                self.env.sent_ptr += 1

                if self.env.is_qa_step():
                    read_output = self._qa_forward(train=True)

                    qa_loss = qa_loss + read_output['loss']
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

                write_output = self.model.mem_forward(**batch)
                act_logit, value = write_output['logit'], write_output['value']

                prob = F.softmax(act_logit, 1)
                log_prob = F.log_softmax(act_logit, 1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)
                action = prob.multinomial(num_samples=1).detach()

                log_prob = log_prob.gather(1, action).cuda(self.gpu_id)
                log_probs.append(log_prob)
                values.append(value)
                if not self.env.is_qa_step():
                    rewards.append(1.0 if self.env.check_solvable() else 0.0)

                self.env.step(action=action.item(), **write_output)

                if self.env.is_qa_step():
                    read_output = self._qa_forward(train=True)

                    qa_loss = qa_loss + read_output['loss']
                    acc = (read_output['pred'] == read_output['target']).item()
                    accs.append(acc)
                    solvs.append(1.0 if self.env.check_solvable() else 0.0)
                    rewards.append(1.0 if self.env.check_solvable() else 0.0)
                    self.env.qa_ptr += 1

        assert(len(accs) == len(self.env.data.qas))
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

            policy_loss = 0.0
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            gae = torch.zeros(1, 1).cuda(self.gpu_id)
            for i in reversed(range(episode_len)):
                R = self.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                delta_t = rewards[i] + self.gamma * values[i+1] - values[i]
                gae = gae * self.gamma * self.tau + delta_t

                policy_loss = policy_loss - gae.detach() * log_probs[i] - self.entropy_coef * entropies[i]

        if self.cfg.model in ['FIFO', 'LIFO', 'UNIFORM']:
            policy_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            value_loss = torch.zeros(1, 1).cuda(self.gpu_id)
            entropy = torch.zeros(1, 1).cuda(self.gpu_id)

        rl_loss = policy_loss + self.value_loss_coef * value_loss
        loss = self.qa_loss_coef * qa_loss + rl_loss

        # Backprop
        self.optim.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.model.cpu()
        self.ensure_shared_grads()
        self.optim.step()

        self.queue.put_nowait(dict(acc=sum(accs) / len(accs),
                                   solv=sum(solvs) / len(solvs),
                                   loss=loss.item(),
                                   qa_loss=qa_loss.item(),
                                   rl_loss=rl_loss.item(),
                                   policy_loss=policy_loss.item(),
                                   entropy=entropy.item(),
                                   value_loss=value_loss.item(),
                                   grad_norm=grad_norm,
                                   ))
