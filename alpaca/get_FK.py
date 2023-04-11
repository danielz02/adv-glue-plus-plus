import json
import nltk
import torch
import joblib
from collections import Counter
from datasets import load_dataset
from util import args, task_to_keys
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer


def get_knowledge(word):
    knowledge = [word]
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
        return list(set(synset))


def get_knowledge_dict(data):
    knowledge_dict = {}

    for key in task_to_keys[args.test_data]:
        if not key:
            continue
        if "seq" not in data:  # TODO: Deal with multiple keys
            data["seq"] = tokenizer.encode(data[key])
            data["seq_len"] = len(data["seq"])

        indexed_tokens = data["seq"]
        tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
        for i in range(1, len(indexed_tokens)):
            if tokenized_words[i] in word_list:
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
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="/scratch/bbkc/danielz/.cache/")
    word_list = joblib.load(args.word_list)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    test_data = load_dataset("glue", args.test_data, cache_dir="/scratch/bbkc/danielz/.cache/", split="validation")
    test_data.map(get_knowledge_dict, num_proc=16).save_to_disk(f"./adv-glue/{args.test_data}/FK")
