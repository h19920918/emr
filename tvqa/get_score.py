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
import json


def get_line_num(filepath):
    return int(subprocess.check_output("wc -l %s | awk '{print $1}'"
                                       % (filepath), shell=True))


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


def main():
    answer_file = '/st1/mkkang/ICLR2019/trivia_qa/qa/wikipedia-dev.json'
    with open(answer_file, "r", encoding='utf-8') as reader:
        answer_data = json.load(reader)["Data"]

    solve_file = '/st1/mkkang/LTMN-BERT/output/predictions.json'
    with open(solve_file, "r", encoding='utf-8') as reader:
        solve_data = json.load(reader)

    f1score = F1score()
    exactscore = ExactMatch()

    f1_list = []
    exact_list = []

    for answer_sheet in answer_data:
        # Check only value?
        truth = answer_sheet["Answer"]["Value"].lower()
        prediction = solve_data[answer_sheet["QuestionId"]]

        f1 = f1score.calc_score(prediction, truth)
        exact = exactscore.calc_score(prediction, truth)
        f1_list.append(f1)
        exact_list.append(exact)
        print("*" * 20)
        print("QuestionId : %s" % answer_sheet["QuestionId"])
        print("Question : %s" % answer_sheet["Question"])
        for i, entity in enumerate(answer_sheet["EntityPages"]):
            if i == 0:
                print("Context : %s" % (entity["Filename"]))
            else:
                print("          %s" % (entity["Filename"]))
        print("Truth : %s" % truth)
        print("Prediction : %s" % prediction)
        #input()

    print("F1 score : {}".format(sum(f1_list) / len(f1_list)))
    print("Exact score : {}".format(sum(exact_list) / len(exact_list)))

    return

if __name__ == '__main__':
    main()
