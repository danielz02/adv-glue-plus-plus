import json
import os
from copy import deepcopy

import nltk
import numpy as np
import torch
import joblib
from collections import Counter
from datasets import load_from_disk

from tokenization_alpaca import GLUE_TASK_TO_KEYS
from util import get_args
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer


def get_knowledge(word):
    original_word = deepcopy(word)
    word = word.strip("▁")

    knowledge = [original_word]
    if not os.path.exists("./corpora/wordnet"):
        nltk.download("wordnet", download_dir="./corpora/wordnet")
    else:
        nltk.data.path = "./corpora/wordnet"
    synset = wn.synsets(word)
    hyposet = []
    hyposet += synset
    for item in synset:
        hyposet += item.hyponyms()
        hyposet += item.hypernyms()
    if len(synset) == 0:  # no synonym
        return knowledge
    else:
        posset = [syn.name().split('.')[1] for syn in synset]  # pos set
        pos = Counter(posset).most_common(1)[0][0]  # most common pos in synset
        new_synset = []
        for syn in synset:
            if syn.name().split('.')[1] == pos:  # only choose synonyms with the most common pos
                new_synset.append(syn.name().split('.')[0])
        synset = new_synset
        if word not in synset:
            synset.append(word)
        return [
            tokenizer.tokenize(x, add_special_tokens=False) for x in set(synset)
        ]


def get_knowledge_dict(data):
    knowledge_dict = {}

    indexed_tokens = data["input_ids"]
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(data["input_start_idx"], data["input_end_idx"]):
        if tokenized_words[i].strip("▁") in word_list:
            words = get_knowledge(tokenized_words[i])
        else:
            words = []
        if len(words) >= 1:
            knowledge_dict[tokenized_words[i]] = words
        else:
            knowledge_dict[tokenized_words[i]] = [tokenized_words[i]]

    data["knowledge_dict"] = json.dumps(knowledge_dict)  # Have to do this to work with Apache Arrow...
    return data


if __name__ == '__main__':
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir=args.cache_dir)
    word_list = set(np.load(args.word_list))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    test_data = load_from_disk(f"./adv-glue/{args.task}/FC_FT")
    if "input_embeddings" in test_data.column_names:
        test_data = test_data.remove_columns(["input_embeddings"])
    test_data.map(get_knowledge_dict, num_proc=64).save_to_disk(f"./adv-glue/{args.task}/FC_FT_FK")
