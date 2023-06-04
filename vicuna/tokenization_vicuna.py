import os.path
from copy import deepcopy
from typing import List, Union, Optional, Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from attacks.tokenization import LABEL_CANDIDATE, GLUE_TASK_TO_KEYS, TASK_DESCRIPTION

# Updated based on https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md More specific
# instructions here: https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat
# /conversation.py#L115-L124 And here:
# https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat/conversation.py#L36


IGNORE_INDEX = -100


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])


def get_preprocess_function(task_name: str, tokenizer: PreTrainedTokenizer, ):
    assert task_name in LABEL_CANDIDATE
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    # FIXME: New special tokens assigned id 0?
    tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>", "<j>", "<k>"]})

    def preprocess_function(example):
        for i, label in enumerate(LABEL_CANDIDATE[task_name]):
            sentence1 = example[sentence1_key]
            message = f"{TASK_DESCRIPTION[task_name]} {sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = example[sentence2_key]
                message = f"{message}<j> {sentence2_key}: <k>{sentence2}"
            message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')
            conversation = Conversation(
                system="<s> A chat between a curious user and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the user's questions.",
                roles=["USER", "ASSISTANT"],
                messages=[["USER", f"<i>{message}</i>"], ["ASSISTANT", f"<l>{label}"]],
                offset=0,
                sep_style=SeparatorStyle.TWO,
                sep=" ",
                sep2="</s>",
            )
            tokens = tokenizer.tokenize(conversation.get_prompt())
            # TODO: Double check tokenizer. Currently using a temporary fix
            if "<s>" in tokens[0]:
                tokens[0] = tokens[0].strip("▁")
            if "</s>" in tokens[-1]:
                tokens[-1] = tokens[-1].strip("▁")

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
            example["label_names"] = LABEL_CANDIDATE[task_name]

            example["input_start_idx"] = input_start_idx
            example["input1_end_idx"] = input1_end_idx if input1_end_idx else input_end_idx
            example["input2_start_idx"] = input2_start_idx
            example["input_end_idx"] = input_end_idx
            example["label_start_idx"] = label_start_idx

        example["target"] = get_attack_target(example, task_name)

        return example

    return preprocess_function


def get_attack_target(x, task):
    labels = LABEL_CANDIDATE[task]

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
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-13B-1.1-HF", cache_dir="./.cache")

    for task in TASK_DESCRIPTION.keys():
        if task == 'mnli':
            split = 'validation_matched'
        elif task == 'mnli-mm':
            split = 'validation_mismatched'
        else:
            split = "validation"
        test_data = load_dataset("glue", task.replace("-mm", ""), cache_dir="./.cache/", split=split)
        save_dir = f"./.cache/glue-preprocessed-benign/TheBloke/vicuna-13B-1.1-HF/{task}/"
        if os.path.exists(save_dir):
            print("Loading preprocessed results")
            test_data = test_data.load_from_disk(save_dir)
        else:
            print("Preprocessing results")
            test_data = test_data.map(get_preprocess_function(task, tokenizer), num_proc=16)
            test_data.save_to_disk(save_dir)

        print(test_data[0])


if __name__ == "__main__":
    main()
