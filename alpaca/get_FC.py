import json
import os

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import faiss
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


def get_topk_words(indices):
    count = Counter(word_list[indices])
    sorted_words = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


def filter_words(words, neighbors):
    words = [item[0] for item in words if item[1] >= neighbors]
    return words


def get_similar_dict(data):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

    similar_char_dict = {}

    for key in GLUE_TASK_TO_KEYS[args.task]:
        if not key:
            continue

        indexed_tokens = data["input_ids"]
        input_embeddings = data["input_embeddings"][data["input_start_idx"]:data["input_end_idx"]]
        # TODO: How to deal with special character ▁?
        tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
        knn_dist, knn_indices = gpu_index_flat.search(input_embeddings, 700)

        for i in range(data["input_start_idx"], data["input_end_idx"]):
            if tokenized_words[i].strip("▁") in word_list_set:
                words = get_topk_words(knn_indices[i - data["input_start_idx"]])
                words = filter_words(words, 8)
            else:
                words = []
            if len(words) >= 1:
                similar_char_dict[tokenized_words[i]] = [
                    tokenizer.tokenize(word, add_special_tokens=False) for word in words
                ]
            else:
                similar_char_dict[tokenized_words[i]] = [tokenized_words[i]]

    data["similar_dict"] = json.dumps(similar_char_dict)
    return data


def get_input_embedding(batch):
    for key in GLUE_TASK_TO_KEYS[args.task]:
        if not key:
            continue

        tokens_tensor = pad_sequence(batch["input_ids"], batch_first=True).to(device)
        masks_tensor = pad_sequence([torch.ones_like(x) for x in batch["input_ids"]], batch_first=True).to(device)
        with torch.no_grad():
            encoded_layers = model.model(input_ids=tokens_tensor, attention_mask=masks_tensor).last_hidden_state
        batch["input_embeddings"] = encoded_layers.cpu().numpy()

    return batch


if __name__ == '__main__':
    # Set the random seed manually for reproducibility.

    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")

    embedding_space = torch.from_numpy(np.load(args.embedding_space))
    word_list = np.load(args.word_list).reshape(-1)
    word_list_set = set(word_list)

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(4096)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(embedding_space)

    test_data = load_dataset("glue", args.task, cache_dir="./.cache/", split="validation")
    test_data = test_data.load_from_disk(f"./.cache/glue-preprocessed-benign/sst2/")
    test_data.set_format(type="pt")

    if os.path.exists(f"./.cache/glue-embedding-benign/sst2/"):
        test_data = test_data.load_from_disk(f"./.cache/glue-embedding-benign/sst2/")
    else:
        model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")
        model.eval()
        model = model.to(device)

        test_data = test_data.map(get_input_embedding, num_proc=1, with_rank=False, batched=True, batch_size=64)
        test_data.save_to_disk(f"./.cache/glue-embedding-benign/sst2/")

    test_data.map(get_similar_dict, num_proc=1, with_rank=False).save_to_disk(f"./adv-glue/{args.task}/FC")
