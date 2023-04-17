import json
import os

import torch
import numpy as np
from util import get_args
from collections import Counter
from datasets import load_dataset
from multiprocess import set_start_method
from tokenization_alpaca import GLUE_TASK_TO_KEYS
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


try:
    set_start_method("spawn")
except RuntimeError:
    pass


args = get_args()
# torch.multiprocessing.set_start_method('spawn', force=True)
# multiprocessing.set_start_method("spawn", force=True)
embedding_space = torch.from_numpy(np.load(args.embedding_space))
word_list = np.load(args.word_list)


def get_knn(t, k):
    dist = torch.norm(embedding_space - t, dim=1, p=None)
    knn = dist.topk(k, largest=False)
    words = []
    for index in knn.indices:
        words.append(word_list[index])
    count = Counter(words)
    sorted_words = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


def filter_words(words, neighbors):
    words = [item[0] for item in words if item[1] >= neighbors]
    return words


def get_similar_dict(data, rank):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

    similar_char_dict = {}

    for key in GLUE_TASK_TO_KEYS[args.task]:
        if not key:
            continue
        if "seq" not in data:  # TODO: Deal with multiple keys
            data["seq"] = tokenizer.encode(data[key])
            data["seq_len"] = len(data["seq"])

        indexed_tokens = data["seq"]

        token_tensor = torch.tensor([indexed_tokens], device=device)
        mask_tensor = torch.tensor([[1] * len(indexed_tokens)], device=device)
        with torch.no_grad():
            encoded_layers = cluster_model.model(token_tensor, mask_tensor).last_hidden_state
        tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
        for i in range(1, len(indexed_tokens)):
            if tokenized_words[i] in word_list:
                words = get_knn(encoded_layers[0][i].cpu(), 700)
                words = filter_words(words, 8)
            else:
                words = []
            if len(words) >= 1:
                similar_char_dict[tokenized_words[i]] = words
            else:
                similar_char_dict[tokenized_words[i]] = [tokenized_words[i]]

    data["similar_dict"] = json.dumps(similar_char_dict)
    return data


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.

    device = torch.device("cuda")

    cluster_model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")
    cluster_model.eval()
    cluster_model = cluster_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")

    torch.manual_seed(args.seed)
    test_data = load_dataset("glue", args.task, cache_dir="./.cache/", split="validation")
    test_data.map(get_similar_dict, num_proc=1, with_rank=True).save_to_disk(f"./adv-glue/{args.task}/FC")
