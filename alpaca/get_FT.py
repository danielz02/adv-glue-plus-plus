import json
from numpy import random
import joblib
import numpy as np
import torch
import string

from datasets import load_dataset
from transformers import AutoTokenizer

from tokenization_alpaca import GLUE_TASK_TO_KEYS
from util import get_args


def bug_delete(word):
    res = word
    point = random.randint(1, len(word) - 2 + 1)
    res = res[0:point] + res[point + 1:]
    return res


def bug_swap(word):
    if len(word) <= 4:
        return word
    res = word
    points = random.choice(range(1, len(word) - 1), 2, replace=False)
    a = points[0]
    b = points[1]

    res = list(res)
    w = res[a]
    res[a] = res[b]
    res[b] = w
    res = ''.join(res)
    return res


def bug_sub_C(word):
    res = word
    key_neighbors = get_key_neighbors()
    point = random.randint(0, len(word) - 1 + 1)

    if word[point] not in key_neighbors:
        return word
    choices = key_neighbors[word[point]]
    subbed_choice = choices[random.randint(0, len(choices) - 1 + 1)]
    res = list(res)
    res[point] = subbed_choice
    res = ''.join(res)

    return res


def bug_insert(word):
    if len(word) >= 6:
        return word
    res = word
    point = random.randint(1, len(word) - 1 + 1)
    res = res[0:point] + random.choice(string.ascii_lowercase, size=1) + res[point:]
    return res


def get_key_neighbors():
    # By keyboard proximity
    neighbors = {
        "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
        "i": "uojkl", "o": "ipkl", "p": "ol",
        "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
        "j": "yuihknm", "k": "uiojlm", "l": "opk",
        "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
    }
    # By visual proximity
    neighbors['i'] += '1'
    neighbors['l'] += '1'
    neighbors['z'] += '2'
    neighbors['e'] += '3'
    neighbors['a'] += '4'
    neighbors['s'] += '5'
    neighbors['g'] += '6'
    neighbors['b'] += '8'
    neighbors['g'] += '9'
    neighbors['q'] += '9'
    neighbors['o'] += '0'

    return neighbors


def get_bug(word):
    bugs = [word]
    if len(word) <= 2:
        return bugs
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    bugs.append(bug_delete(word))
    bugs.append(bug_swap(word))
    bugs.append(bug_sub_C(word))
    return list(set(bugs))


def get_bug_dict(data):
    bug_dict = {}

    indexed_tokens = data["input_ids"]
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(data["input_start_idx"], data["input_end_idx"]):
        if tokenized_words[i].strip("▁") in word_list:
            words = get_bug(tokenized_words[i])
        else:
            words = []
        if len(words) >= 1:
            bug_dict[tokenized_words[i]] = words
        else:
            bug_dict[tokenized_words[i]] = [tokenized_words[i]]

    data["bug_dict"] = json.dumps(bug_dict)  # Have to do this to work with Apache Arrow...
    return data


if __name__ == '__main__':
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir=args.cache_dir)
    word_list = set(np.load(args.word_list))
    torch.manual_seed(args.seed)
    if args.task == 'mnli':
        split = 'validation_matched'
    elif args.task == 'mnli-mm':
        split = 'validation_mismatched'
    else:
        split = "validation"
    test_data = load_dataset("glue", args.task.replace("-mm", ""), cache_dir=args.cache_dir, split=split)
    test_data = test_data.load_from_disk(f"./adv-glue/{args.task}/FC")
    if "input_embeddings" in test_data.column_names:
        test_data = test_data.remove_columns(["input_embeddings"])
    test_data.map(get_bug_dict, num_proc=64).save_to_disk(f"./adv-glue/{args.task}/FC_FT")
