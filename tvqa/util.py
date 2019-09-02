from collections import Counter
import numpy as np
import os
import random
import re
import string
import subprocess

import torch
from torch import nn
import torch.nn.functional as F

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_line_num(filepath):
    return int(subprocess.check_output("wc -l %s | awk '{print $1}'"
                                       % (filepath), shell=True))

def get_num_gpus():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        num_gpus = len(visible.split(','))
    else:
        num_gpus = torch.cuda.device_count()
    return num_gpus

def mask(x, m, value=1e-30):
    out = (1 - m).float() * value
    in_ = x * m.float()
    return in_ + out


def normalize_answer(s):
    '''Reference: https://rajpurkar.github.io/SQuAD-explorer'''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class F1score():
    '''Reference: https://rajpurkar.github.io/SQuAD-explorer'''
    # Input:
    #   - prediction: [word1, word2, word3, word4, ...]
    #   - ground_truth: [word1, word2, word3, word4, ...]
    # Output:
    #   F1 score
    def calc_score(self, prediction, ground_truth):
        ground_truth = ' '.join(ground_truth)
        prediction = ' '.join(prediction)

        ground_truth = normalize_answer(ground_truth).split()
        prediction = normalize_answer(prediction).split()
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


class ExactMatch():
    '''Reference: https://rajpurkar.github.io/SQuAD-explorer'''
    # Input:
    #   - prediction: [word1, word2, word3, word4, ...]
    #   - ground_truth: [word1, word2, word3, word4, ...]
    # Output:
    #   1(prediction == ground_truth)
    #                or
    #   0(prediction != ground_truth)
    def calc_score(self, prediction, ground_truth):
        ground_truth = ' '.join(ground_truth)
        prediction = ' '.join(prediction)

        ground_truth = normalize_answer(ground_truth)
        prediction = normalize_answer(prediction)
        if prediction == ground_truth:
            return 1
        else:
            return 0


class ReduceScale(object):
    def __init__(self, factor, initial_value=None):
        self.factor = factor
        self.avg = initial_value

    def update(self, value):
        if self.avg is None:
            self.avg = value
        else:
            self.avg = self.factor * self.avg + (1-self.factor) * value
        return self.avg

    def get(self):
        return self.avg


class EMA(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = average.clone()


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_features, n_layers=2, act=F.relu):
        super(Highway, self).__init__()
        self.n_layer = n_layers
        self.linear =  nn.ModuleList([Linear(in_features, in_features) for _ in range(n_layers)])
        self.non_linear =  nn.ModuleList([Linear(in_features, in_features) for _ in range(n_layers)])
        self.gate =  nn.ModuleList([Linear(in_features, in_features) for _ in range(n_layers)])
        self.act = act

    def forward(self, x):
        for i in range(self.n_layer):
            gate_x = torch.sigmoid(self.gate[i](x))
            non_linear_x = self.act(self.non_linear[i](x))
            linear_x = self.linear[i](x)

            x = gate_x * non_linear_x + (1 - gate_x) * linear_x
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

        if self.rnn.bidirectional:
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x, x_len):
        x = self.dropout(x)
        x_len = x_len.sum(-1)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        self.rnn.flatten_parameters()
        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        return x
