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

from alpaca.tokenization_alpaca import ALPACA_LABEL_CANDIDATE, GLUE_TASK_TO_KEYS, ALPACA_TASK_DESCRIPTION

# Updated based on https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md More specific
# instructions here: https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat
# /conversation.py#L115-L124 And here:
# https://github.com/lm-sys/FastChat/blob/00d9e6675bdff60be6603ffff9313b1d797d2e3e/fastchat/conversation.py#L36


IGNORE_INDEX = -100


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    NEW_LINE = auto()
    BILLA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # The name of this template
    name: str
    # System prompts
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    # Used for the state in the gradio servers.
    conv_id: Any = None
    skip_next: bool = False
    model_name: str = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.BAIZE:
            ret = self.system + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + message + "\n"
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                            role
                            + ": "
                            + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.NEW_LINE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.BILLA:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])


def get_preprocess_function(task_name: str, tokenizer: PreTrainedTokenizer, ):
    assert task_name in ALPACA_LABEL_CANDIDATE
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    # FIXME: New special tokens assigned id 0?
    tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>", "<j>", "<k>"]})

    def preprocess_function(example):
        for i, label in enumerate(ALPACA_LABEL_CANDIDATE[task_name]):
            sentence1 = example[sentence1_key]
            message = f"{ALPACA_TASK_DESCRIPTION[task_name]}\n{sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = example[sentence2_key]
                message = f"{message}<j>\n{sentence2_key}: <k>{sentence2}"
            message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')
            conversation = Conversation(
                name="vicuna_v1.1",
                system="A chat between a curious user and an artificial intelligence assistant. "
                       "The assistant gives helpful, detailed, and polite answers to the user's questions.",
                roles=["USER", "ASSISTANT"],
                messages=[["USER", f"<i>{message}</i>"], ["ASSISTANT", f"<l>{label}"]],
                offset=0,
                sep_style=SeparatorStyle.ADD_COLON_TWO,
                sep=" ",
                sep2="</s>",
            )
            tokens = tokenizer.tokenize(conversation.get_prompt())
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
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-13B-1.1-HF", cache_dir="./.cache")

    for task in ALPACA_TASK_DESCRIPTION.keys():
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
