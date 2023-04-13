import errno
import json
import argparse

import logging
import os
import random
import sys

import numpy as np
import torch


task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}

class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cache_dir', type=str, default='/scratch/bbkc/danielz/.cache/',
                        help='from_pretrained cache dir')
    parser.add_argument('--task', type=str, default='sst2',
                        choices=['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte', 'wnli'],
                        help='GLUE tasks')
    parser.add_argument("--model", type=str, default='chavinlo/alpaca-native')
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default='adv_results/dataset',
                        help="pre-processed dataset")
    parser.add_argument('--embedding-data', type=str, default='../synonym_data/s.pt',
                        help='location of the embedding data, should be a json file')
    parser.add_argument('--word-list', type=str, default='../synonym_data/word_list.pkl',
                        help='location of the word list data, should be a json file')
    parser.add_argument('--const', type=float, default=1e4,
                        help='initial const for cw attack')
    parser.add_argument('--confidence', type=float, default=0,
                        help='initial const for cw attack')
    parser.add_argument('--lr', type=float, default=.1,
                        help='initial learning rate')

    parser.add_argument('--max_steps', type=int, default=100,
                        help='cw max steps')
    parser.add_argument('--l1', action='store_true',
                        help='use l1 norm')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')

    return parser.parse_args()


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def init_logger(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir('./'+root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='a')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



# PAD = 0
# UNK = 1
# # not part of the qa vocab, assigned with minus index
# EOS = -1
# SOS = -2
#
# PAD_WORD = '<pad>'
# UNK_WORD = '<unk>'
# EOS_WORD = '<eos>'
# SOS_WORD = '<sos>'
