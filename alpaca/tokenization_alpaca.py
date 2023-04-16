from copy import deepcopy
from typing import List, Union, Optional, Dict

import torch
from transformers import LlamaTokenizer, TensorType, PreTrainedTokenizer
from transformers.tokenization_utils_base import TextInput, TruncationStrategy, BatchEncoding
from transformers.utils import PaddingStrategy

ALPACA_TASK_DESCRIPTION = {
    "sst2": "For the given input text, label the sentiment of the text as positive or negative. The answer should be "
            "exact 'positive' or 'negative'."
}

ALPACA_LABEL_CANDIDATE = {
    "sst2": ["negative", "positive"]
}

ALPACA_PROMPT_TEMPLATE = "Below is an instruction that describes a task, paired with an input that provides further " \
                         "context. Write a response that appropriately completes the request.\n\n" \
                         "### Instruction:\n{instruction}\n\n### Input:\n<i>{input}</i>\n\n### Response:<l>{label}"

GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

IGNORE_INDEX = -100


def get_preprocess_function(task_name: int, tokenizer: PreTrainedTokenizer,):
    assert task_name in ALPACA_LABEL_CANDIDATE
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
    # FIXME: New special tokens assigned id 0?
    tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>"]})

    def preprocess_function(example):
        for label in ALPACA_LABEL_CANDIDATE[task_name]:
            sentence1 = example[sentence1_key]
            message = f"{sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = example[sentence2_key]
                message = f"{message}\n{sentence2_key}: {sentence2}"
            message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')
            prompt = ALPACA_PROMPT_TEMPLATE.format(
                instruction=ALPACA_TASK_DESCRIPTION[task_name], input=message, label=f"{label} </s>"
            )
            tokens = tokenizer.tokenize(prompt)
            input_start_idx = tokens.index("<i>")
            tokens.remove("<i>")
            input_end_idx = tokens.index("</i>")
            tokens.remove("</i>")
            label_start_idx = tokens.index("<l>")
            tokens.remove("<l>")
            token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)
            label_token_ids = token_ids.clone()
            label_token_ids[:label_start_idx] = IGNORE_INDEX

            token_type_ids = torch.zeros_like(token_ids)
            token_type_ids[input_start_idx:input_end_idx] = 1
            token_type_ids[input_end_idx:label_start_idx] = 0
            token_type_ids[label_start_idx:-1] = 2

            example[f"{label}_input_ids"] = token_ids
            example[f"{label}_input_ids_type"] = token_type_ids
            example[f"{label}_label_input_ids"] = label_token_ids

        return example

    return preprocess_function


class AlpacaZeroShotTokenizer(LlamaTokenizer):
    """
        Should be a more elegant implementation. Save for later.
        1. Add special tokens for instruction, input, label
        2. Implement create_token_type_ids_from_sequences like BERTTokenizer to create mask for different segment
    """

    def __init__(self, vocab_file, **kwargs):
        kwargs["add_eos_token"] = True  # need to add eos token for language modeling
        super().__init__(vocab_file, **kwargs)

    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different from `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:  # FIXME: Concatenate as Llama zero-shot prompts
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs
