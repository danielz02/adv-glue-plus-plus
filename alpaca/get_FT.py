import json
import random
import joblib
import torch
import string

from util import args
from datasets import load_dataset
from transformers import AutoTokenizer


task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}


def bug_delete(word):
    res = word
    point = random.randint(1, len(word) - 2)
    res = res[0:point] + res[point + 1:]
    return res


def bug_swap(word):
    if len(word) <= 4:
        return word
    res = word
    points = random.sample(range(1, len(word) - 1), 2)
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
    point = random.randint(0, len(word) - 1)

    if word[point] not in key_neighbors:
        return word
    choices = key_neighbors[word[point]]
    subbed_choice = choices[random.randint(0, len(choices) - 1)]
    res = list(res)
    res[point] = subbed_choice
    res = ''.join(res)

    return res


def bug_insert(word):
    if len(word) >= 6:
        return word
    res = word
    point = random.randint(1, len(word) - 1)
    res = res[0:point] + random.choice(string.ascii_lowercase) + res[point:]
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
    for key in task_to_keys[args.test_data]:
        if not key:
            continue
        if "seq" not in data:
            data["seq"] = tokenizer.encode(data[key])
            data["seq_len"] = len(data["seq"])

        indexed_tokens = data["seq"]
        tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
        for i in range(1, len(indexed_tokens)):
            if tokenized_words[i] in word_list:
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
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="/scratch/bbkc/danielz/.cache/")
    word_list = joblib.load(args.word_list)
    torch.manual_seed(args.seed)
    test_data = load_dataset("glue", args.test_data, cache_dir="/scratch/bbkc/danielz/.cache/", split="validation")
    test_data.map(get_bug_dict, num_proc=16).save_to_disk(f"./adv-glue/{args.test_data}/FT")
