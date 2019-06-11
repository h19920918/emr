from collections import Counter
import json
import math
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
        return x/warmup
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


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


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


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def has_exact_match(ground_truths, candidates):
    for ground_truth in ground_truths:
        if ground_truth in candidates:
            return True
    return False


def get_ground_truths(answer):
    return answer['NormalizedAliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])]


def evaluate_triviaqa(ground_truth, predicted_answers, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Missed question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predicted_answers[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0 and not mute:
            print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = exact_match / len(qid_list)
    f1 = f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}


def get_score_from_trivia(cfg, set_name):
    if set_name == 'verified-dev':
        dataset_file = './data/trivia_qa/qa/verified-wikipedia-dev.json'
    elif set_name == 'dev':
        dataset_file = './data/trivia_qa/qa/wikipedia-dev.json'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)

    with open(cfg.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    key_to_ground_truth = get_key_to_ground_truth(dataset_json)
    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions, mute=True)
    return eval_dict


def get_key_to_ground_truth(data):
    if data['Domain'] == 'Wikipedia':
        return {datum['QuestionId']: datum['Answer'] for datum in data['Data']}
    else:
        return get_qd_to_answer(data)


def get_question_doc_string(qid, doc_name):
    return '{}--{}'.format(qid, doc_name)


def get_qd_to_answer(data):
    key_to_answer = {}
    for datum in data['Data']:
        for page in datum.get('EntityPages', []) + datum.get('SearchResults', []):
            qd_tuple = get_question_doc_string(datum['QuestionId'], page['Filename'])
            key_to_answer[qd_tuple] = datum['Answer']
    return key_to_answer


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


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        weight_shape = list(self.linear.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        self.linear.weight.data.uniform_(-w_bound, w_bound)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.0):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.rnn.num_layers):
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)

            if self.rnn.bidirectional:
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)

    def forward(self, x, x_len=None):
        if x_len is None:
            if hasattr(self, 'dropout'):
                x = self.dropout(x)
            x, _ = self.rnn(x)
            return x
        else:
            if hasattr(self, 'dropout'):
                x = self.dropout(x)
            x_len = x_len.sum(-1)

            x_len_sorted, x_idx = torch.sort(x_len, descending=True)
            x_sorted = x.index_select(dim=0, index=x_idx)
            _, x_ori_idx = torch.sort(x_idx)

            self.rnn.flatten_parameters()
            x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
            x_packed, _ = self.rnn(x_packed)

            x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
            x = x.index_select(dim=0, index=x_ori_idx)
            return x


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.rnn_cell = nn.GRUCell(input_size=input_size,
                                   hidden_size=hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(getattr(self.rnn_cell, f'bias_hh'), val=0)
        nn.init.constant_(getattr(self.rnn_cell, f'bias_ih'), val=0)

    def forward(self, x, h):
        x = self.rnn_cell(x, h)
        return x


class Transformer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, dropout=0.0):
        super(Transformer, self).__init__()
        self.attention = Attention(hidden_size=hidden_size,
                                   num_attention_heads=num_attention_heads,
                                   dropout=dropout)
        self.intermediate = Intermediate(hidden_size=hidden_size,
                                         intermediate_size=intermediate_size)
        self.output = Output(hidden_size=hidden_size,
                             intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size=hidden_size,
                                  num_attention_heads=num_attention_heads,
                                  dropout=dropout)
        self.output = SelfOutput(hidden_size=hidden_size,
                                 dropout=dropout)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(SelfOutput, self).__init__()
        self.dense = Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if hasattr(self, 'dropout'):
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        if hasattr(self, 'dropout'):
            attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        weight_shape = list(self.weight.data.size())
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_out))
        self.weight.data.uniform_(-w_bound, w_bound)
        self.bias.data.uniform_(-w_bound, w_bound)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = Linear(hidden_size, intermediate_size)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0):
        super(Output, self).__init__()
        self.dense = Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if hasattr(self, 'dropout'):
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)
