import os.path
from copy import deepcopy
from typing import List, Union, Optional, Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, TensorType, PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding
from transformers.utils import PaddingStrategy

from alpaca.tokenization_alpaca import ALPACA_LABEL_CANDIDATE, ALPACA_TASK_DESCRIPTION, GLUE_TASK_TO_KEYS


VICUNA_PROMPT_TEMPLATE = """\
### Human: {instruction}\n<i>{input}</i>
### Assistant: <l>{label}
"""

IGNORE_INDEX = -100


def get_preprocess_function(task_name: str, tokenizer: PreTrainedTokenizer, ):
    assert task_name in ALPACA_LABEL_CANDIDATE
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    # FIXME: New special tokens assigned id 0?
    tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>", "<j>", "<k>"]})

    def preprocess_function(example):
        for i, label in enumerate(ALPACA_LABEL_CANDIDATE[task_name]):
            sentence1 = example[sentence1_key]
            message = f"{sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = example[sentence2_key]
                message = f"{message}<j>\n{sentence2_key}: <k>{sentence2}"
            message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')
            prompt = VICUNA_PROMPT_TEMPLATE.format(
                instruction=ALPACA_TASK_DESCRIPTION[task_name], input=message, label=f"{label}"
            )
            tokens = tokenizer.tokenize(prompt)
            input_start_idx = tokens.index("<i>")
            tokens.remove("<i>")
            if sentence2_key:
                input1_end_idx = tokens.index("<j>")
                tokens.remove("<j>")
                input2_start_idx = tokens.index("<k>")
                tokens.remove("<k>")
            else:
                input1_end_idx = None
                input2_start_idx = torch.nan
            input_end_idx = tokens.index("</i>")
            tokens.remove("</i>")
            label_start_idx = tokens.index("<l>")
            tokens.remove("<l>")
            token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)

            instruction_token_ids = token_ids[:input_start_idx].clone()
            input_token_ids = token_ids[input_start_idx:input_end_idx].clone()
            response_header_token_ids = token_ids[input_end_idx:label_start_idx].clone()
            response_token_ids = token_ids[label_start_idx:].clone()

            if i == example["label"]:
                example["input_ids"] = token_ids
            example[f"instruction_token_ids"] = instruction_token_ids
            example[f"input_token_ids"] = input_token_ids
            example[f"response_header_token_ids"] = response_header_token_ids
            example[f"{label}_response_token_ids"] = response_token_ids
            example["label_names"] = ALPACA_LABEL_CANDIDATE[task_name]

            example["input_start_idx"] = input_start_idx
            example["input1_end_idx"] = input1_end_idx if input1_end_idx else input_end_idx
            example["input2_start_idx"] = input2_start_idx
            example["input_end_idx"] = input_end_idx
            example["label_start_idx"] = label_start_idx

        example["target"] = get_attack_target(example, task_name)

        return example

    return preprocess_function


def get_attack_target(x, task):
    labels = ALPACA_LABEL_CANDIDATE[task]

    if len(labels) == 3:
        if x["label"] == 2:
            target = 0
        elif x["label"] == 0:
            target = 2
        else:
            if np.random.uniform() < 0.5:
                target = 0
            else:
                target = 2
    elif len(labels) == 2:
        if x["label"] == 0:
            target = 1
        else:
            target = 0
    else:
        raise Exception('Unknown number of labels.')

    return target


def main():
    for task in ALPACA_TASK_DESCRIPTION.keys():
        tokenizer = LlamaTokenizer.from_pretrained("TheBloke/stable-vicuna-13B-HF", cache_dir="./.cache/")

        if task == 'mnli':
            split = 'validation_matched'
        elif task == 'mnli-mm':
            split = 'validation_mismatched'
        else:
            split = "validation"
        test_data = load_dataset("glue", task.replace("-mm", ""), cache_dir="./.cache/", split=split)
        save_dir = f"./.cache/glue-preprocessed-benign/TheBloke/stable-vicuna-13B-HF/{task}/"
        if os.path.exists(save_dir):
            print("Loading preprocessed results")
            test_data = test_data.load_from_disk(save_dir)
        else:
            print("Preprocessing results")
            test_data = test_data.map(
                get_preprocess_function(task, tokenizer), num_proc=16
            )
            test_data.save_to_disk(save_dir)

        print(test_data[0])


if __name__ == "__main__":
    main()
