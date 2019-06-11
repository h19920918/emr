import argparse
import collections
import json
from math import ceil
import multiprocessing as mp
import os
import pickle
import re
import string
from tqdm import tqdm

from tokenization import BertTokenizer


def normalize_tokens(s):
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


def extract_indices(story_token, answ_token):
    answ_len = len(answ_token)
    story_len = len(story_token)
    s_idx, e_idx = None, None
    if answ_len == 1:
        for i in range(story_len):
            if answ_token[0] == story_token[i]:
                s_idx, e_idx = i, i
                break
    else:
        for i in range(story_len-answ_len+1):
            if story_token[i:i+answ_len] == answ_token:
                s_idx, e_idx = i, i+answ_len-1
                break
    return s_idx, e_idx


class ExampleLoader(mp.Process):
    def __init__(self, loader_id, cfg, domain, split, examples, queue):
        super().__init__(name='loader-%02d' % loader_id)
        self.loader_id = loader_id
        self.cfg = cfg
        self.domain = domain
        self.split = split
        self.examples = examples
        self.queue = queue

        self.story_path = os.path.join(self.cfg.data_dir, 'evidence', self.domain)
        self.prepro_path = os.path.join(self.cfg.prepro_dir, self.domain, self.split)

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def run(self):
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)
        for example in tqdm(self.examples, desc=self.name, position=self.loader_id):
            if self.domain == 'wikipedia':
                story_list = example['EntityPages']
                story_list = [os.path.join(self.story_path, story['Filename']) for story in story_list]
            elif self.domain == 'web':
                story_list = example['SearchResult']
                story_list = [os.path.join(self.story_path, story['Filename']) for story in story_list]
            else:
                print('There is a empty instance')
                exit(1)

            example_id = example['QuestionId'].lower()
            ques = example['Question']
            ques = normalize_tokens(ques)
            ques = ques.split()
            tmp_ques = []
            for q in ques:
                q = tokenizer.tokenize(q)
                tmp_ques += q
            ques_token = tmp_ques[:self.cfg.max_ques_token]
            if 'test' not in self.split:
                answers = example['Answer']['NormalizedAliases']
                if 'HumanAnswers' in example['Answer'].keys():
                    answers += [ans for ans in example['Answer']['HumanAnswers']]

            story_idx = 1
            story_id = '%s_%02d' % (example_id, story_idx)
            null_flag = True
            for s_file in story_list:
                s_filename = s_file.split('/')[-1]
                with open(s_file, 'r') as f:
                    s_context = f.readlines()
                s_context = ' '.join(s_context)
                s_context = normalize_tokens(s_context)
                doc_tokens = s_context.split()

                if self.split == 'train':
                    # Train
                    none_flag = True
                    story_token = doc_tokens[:self.cfg.train_story_token]
                    s_lengths = []
                    e_lengths = []
                    answ_tokens = []
                    for answ_cand in answers:
                        s_idx, e_idx = None, None
                        answ_cand = normalize_tokens(answ_cand)
                        answ_token = answ_cand.split()
                        s_idx, e_idx = extract_indices(story_token, answ_token)
                        s_lengths.append(s_idx)
                        e_lengths.append(e_idx)
                        answ_tokens.append(answ_token)
                    min_s_idx, min_e_idx = 100000000, 100000000
                    for i in range(len(s_lengths)):
                        if s_lengths[i] is None or e_lengths[i] is None:
                            continue
                        else:
                            if min_s_idx > s_lengths[i] and min_e_idx > e_lengths[i]:
                                min_s_idx = s_lengths[i]
                                min_e_idx = e_lengths[i]
                                answ_token = answ_tokens[i]
                            none_flag = False
                    if none_flag:
                        continue
                    s_idx, e_idx = min_s_idx, min_e_idx

                    all_tokens = []
                    tok_to_orig_index = []
                    orig_to_tok_index = []
                    for (i, token) in enumerate(story_token):
                        orig_to_tok_index.append(len(all_tokens))
                        sub_tokens = tokenizer.tokenize(token)
                        for sub_token in sub_tokens:
                            tok_to_orig_index.append(i)
                            all_tokens.append(sub_token)

                    tok_start_position = orig_to_tok_index[s_idx]
                    if e_idx < len(story_token) - 1:
                        tok_end_position = orig_to_tok_index[e_idx+1]-1
                    else:
                        tok_end_position = len(all_tokens) - 1
                    if tok_start_position > self.cfg.train_story_token or tok_end_position > self.cfg.train_story_token:
                        continue
                    null_flag = False

                    self.write_txtfile(self.prepro_path, story_id, '.story.words', all_tokens)
                    self.write_txtfile(self.prepro_path, story_id, '.ques.words', ques_token)
                    self.write_txtfile(self.prepro_path, story_id, '.answ.words', [answ_token])
                    self.write_txtfile(self.prepro_path, story_id, '.index', [[tok_start_position], [tok_end_position]])
                    self.write_txtfile(self.prepro_path, story_id, '.filename', s_filename)
                elif self.split == 'dev' or self.split == 'verified-dev':
                    # Dev
                    none_flag = True
                    story_token = doc_tokens[:self.cfg.test_story_token]
                    s_lengths = []
                    e_lengths = []
                    answ_tokens = []
                    answers = list(set(answers))
                    for answ_cand in answers:
                        s_idx, e_idx = None, None
                        answ_cand = normalize_tokens(answ_cand)
                        answ_token = answ_cand.split()
                        s_idx, e_idx = extract_indices(story_token, answ_token)
                        if s_idx is not None and e_idx is not None:
                            s_lengths.append(s_idx)
                            e_lengths.append(e_idx)
                            answ_tokens.append(answ_token)
                            none_flag = False
                    if none_flag:
                        continue

                    all_tokens = []
                    tok_to_orig_index = []
                    orig_to_tok_index = []
                    for (i, token) in enumerate(story_token):
                        orig_to_tok_index.append(len(all_tokens))
                        sub_tokens = tokenizer.tokenize(token)
                        for sub_token in sub_tokens:
                            tok_to_orig_index.append(i)
                            all_tokens.append(sub_token)

                    tok_start_positions = []
                    tok_end_positions = []
                    for s_idx, e_idx in zip(s_lengths, e_lengths):
                        tok_start_positions.append(orig_to_tok_index[s_idx])
                        if e_idx < len(story_token) - 1:
                            tok_end_positions.append(orig_to_tok_index[e_idx+1]-1)
                        else:
                            tok_end_positions.append(len(all_tokens) - 1)

                    self.write_txtfile(self.prepro_path, story_id, '.story.words', all_tokens)
                    self.write_txtfile(self.prepro_path, story_id, '.ques.words', ques_token)
                    self.write_txtfile(self.prepro_path, story_id, '.answ.words', answ_tokens)
                    self.write_txtfile(self.prepro_path, story_id, '.index', [tok_start_positions, tok_end_positions])
                    self.write_txtfile(self.prepro_path, story_id, '.filename', s_filename)
                else:
                    # Test
                    story_token = doc_tokens[:self.cfg.test_story_token]
                    all_tokens = []
                    tok_to_orig_index = []
                    orig_to_tok_index = []
                    for (i, token) in enumerate(story_token):
                        orig_to_tok_index.append(len(all_tokens))
                        sub_tokens = tokenizer.tokenize(token)
                        for sub_token in sub_tokens:
                            tok_to_orig_index.append(i)
                            all_tokens.append(sub_token)

                    self.write_txtfile(self.prepro_path, story_id, '.story.words', all_tokens)
                    self.write_txtfile(self.prepro_path, story_id, '.ques.words', ques_token)
                    self.write_txtfile(self.prepro_path, story_id, '.filename', s_filename)

                story_idx += 1
                story_id = '%s_%02d' % (example_id, story_idx)

    def write_txtfile(self, prepro_path, story_id, name, tokens):
        filename = os.path.join(prepro_path, story_id + name)
        with open(filename, 'w+') as f:
            if name == '.answ.words':
                for token in tokens:
                    f.write(' '.join(token))
                    f.write('\n')
            elif name == '.index':
                s_lengths = tokens[0]
                e_lengths = tokens[1]
                for s_idx, e_idx in zip(s_lengths, e_lengths):
                    f.write(str(s_idx) + ' ' + str(e_idx))
                    f.write('\n')
            elif name == '.filename':
                    f.write(tokens)
            else:
                for token in tokens:
                    f.write(str(token))
                    f.write('\n')


def config():
    parser = argparse.ArgumentParser(description='Preprocessing for TriviaQA')
    parser.add_argument('--data-dir', type=str, default='./data/trivia_qa')
    parser.add_argument('--prepro-dir', type=str, default='./prepro/trivia_qa')
    parser.add_argument('--task', type=str, default='wikipedia')
    parser.add_argument('--bert-model', type=str, default='bert-base-uncased')

    parser.add_argument('--train-story-token', default=1200)
    parser.add_argument('--test-story-token', default='all')
    parser.add_argument('--max-ques-token', type=int, default=39)

    parser.add_argument('--num-dataloaders', type=int, default=mp.cpu_count())

    parser.add_argument('--debug', action='store_true')
    cfg = parser.parse_args()

    print('='*50)
    print('Preprocessing for TriviaQA')
    print('='*50)
    print()
    print('='*50)
    print('Configuration')
    print('='*50)
    print('-'*50)
    for c in vars(cfg):
        print(c, '=', getattr(cfg, c))
    print('-'*50)

    if cfg.train_story_token == 'all' and cfg.test_story_token == 'all':
        cfg.prepro_dir = os.path.join(cfg.prepro_dir, 'train-all-test-all')
        cfg.train_story_token = 100000000
        cfg.test_story_token = 100000000
    elif cfg.test_story_token == 'all':
        cfg.prepro_dir = os.path.join(cfg.prepro_dir, 'train-%d-test-all' % (cfg.train_story_token))
        cfg.test_story_token = 100000000
    else:
        cfg.prepro_dir = os.path.join(cfg.prepro_dir, 'train-%d-test-%d' % (cfg.train_story_token, cfg.test_story_token))
    return cfg


def prepro_data(cfg):
    qa_folder = os.path.join(cfg.data_dir, 'qa')
    qa_files = [os.path.join(qa_folder, qa_file) for qa_file in os.listdir(qa_folder)]
    usage_qa_files = []
    print('='*50)
    print('Usage Question-Answering files')
    print('='*50)
    print('-'*50)
    for qa_file in qa_files:
        if cfg.task not in qa_file:
            continue
        usage_qa_files.append(qa_file)
        print(qa_file)
    print('-'*50)

    for qa_file in usage_qa_files:
        print('='*50)
        print('Preprocess for %s' % qa_file)
        print('='*50)
        data = json.load(open(qa_file))
        examples = data['Data']
        domain = data['Domain'].lower()
        split = data['Split'].lower()
        verified = data['VerifiedEval']
        if verified:
            split = 'verified-' + split

        prepro_path = os.path.join(cfg.prepro_dir, domain, split)
        if cfg.debug == False:
            if not os.path.exists(prepro_path):
                os.makedirs(prepro_path)

        loaders = []
        queue = mp.Queue()
        chunk_size = int(ceil(len(examples) / cfg.num_dataloaders))
        for i in range(cfg.num_dataloaders):
            example_chunk = examples[i*chunk_size:(i+1)*chunk_size]
            loader = ExampleLoader(i, cfg, domain, split, example_chunk, queue)
            if cfg.debug:
                loader.run()
            else:
                loader.start()
            loaders.append(loader)

        for loader in loaders:
            loader.join()


def main(cfg):
    if cfg.debug:
        print('='*50)
        print('Enter debugging mode')
        print('='*50)
    else:
        if os.path.isdir(cfg.prepro_dir):
            print('Already exist preprocessing folder, please remove existing folder')
            print('Folder name = %s' % cfg.prepro_dir)
            exit(1)
        else:
            os.makedirs(cfg.prepro_dir)
    prepro_data(cfg)
    print('='*50)
    print('Done')
    print('='*50)


if __name__ == '__main__':
    cfg = config()
    if cfg.debug:
        cfg.num_dataloaders = 1
    main(cfg)
