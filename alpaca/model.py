import json
import os.path
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from torch import nn
import torch

from tokenization_alpaca import ALPACA_TASK_DESCRIPTION
from util import get_args


class LlamaForZeroShotSequenceClassification(nn.Module):
    def __init__(self, config, cache_dir, torch_compile=False):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(config, cache_dir=cache_dir)
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        if torch_compile:
            self.model = torch.compile(self.model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # (batch_size, n_labels, seq_len)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # (batch_size, n_labels, seq_len)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        if inputs_embeds is not None:
            batch_size, n_labels, seq_len, _ = inputs_embeds.shape
        elif input_ids is not None:
            batch_size, n_labels, seq_len = input_ids.shape
        else:
            raise ValueError("Need at least one input_embedding!")

        if input_ids is not None:
            input_ids = input_ids.view(-1, seq_len)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.view(-1, seq_len, self.hidden_size)
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, seq_len)
        if labels is not None:
            labels = labels.view(-1, seq_len)

        assert "classification_labels" in kwargs

        decoder_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # FIXME: Remove hard code
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        shift_logits = decoder_outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels).view(batch_size * n_labels, -1)
        log_probs = torch.sum(-loss, dim=-1).view(batch_size, n_labels)

        hidden_states = decoder_outputs[0]
        logits = log_probs
        if kwargs["classification_labels"]:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(input=logits, target=kwargs["classification_labels"])
        else:
            loss = None

        if not return_dict:
            output = (logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,  # TODO: Needs reshaping. Leave it for now...
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)


class ZeroShotLlamaForSemAttack(nn.Module):
    def __init__(self, model_name, cache_dir, torch_compile=False):
        super().__init__()
        self.torch_compile = torch_compile
        self.llama_classifier = LlamaForZeroShotSequenceClassification(
            model_name, cache_dir, torch_compile=torch_compile
        )

    def get_input_embedding_vector(self, input_ids: torch.LongTensor):
        return self.llama_classifier.model.get_input_embeddings()(input_ids)

    def forward(
        self, input_dict: Dict[str, Union[torch.Tensor, List]], gold=None, perturbed=None, **kwargs
    ) -> Dict[str, Any]:
        # Assuming single input_embedding for now...
        instruction_token_embedding = self.get_input_embedding_vector(input_dict["instruction_token_ids"])
        if perturbed is not None:
            input_token_embedding = perturbed
        else:
            input_token_embedding = self.get_input_embedding_vector(input_dict["input_token_ids"])
        response_header_token_embedding = self.get_input_embedding_vector(input_dict["response_header_token_ids"])
        inputs_embeds = torch.stack([
            torch.cat([
                instruction_token_embedding,
                input_token_embedding,
                response_header_token_embedding,
                self.get_input_embedding_vector(input_dict[f"{label}_response_token_ids"])
            ]) for label in input_dict["label_names"]
        ])  # (num_labels, seq_len, 4096)

        label_input_ids = torch.stack([
            torch.cat([
                torch.ones(
                    instruction_token_embedding.size(0), dtype=torch.long,
                    device=instruction_token_embedding.device
                ) * -100,
                torch.ones(
                    input_token_embedding.size(0), dtype=torch.long,
                    device=instruction_token_embedding.device
                ) * -100,
                torch.ones(
                    response_header_token_embedding.size(0), dtype=torch.long,
                    device=instruction_token_embedding.device
                ) * -100,
                input_dict[f"{label}_response_token_ids"]
            ]) for label in input_dict["label_names"]
        ])  # (num_labels, seq_len,)

        classifier_args = {
            "input_ids": None,
            "labels": label_input_ids.unsqueeze(0),
            "inputs_embeds": inputs_embeds.unsqueeze(0),
            "classification_labels": None,  # gold
            "return_dict": True
        }

        out = self.llama_classifier(**classifier_args, **kwargs)
        ret = {"pred": out.logits, "logits": out.logits}
        if gold:
            ret["loss"] = out.loss
        ret['embedding'] = out.hidden_states  # FIXME: Double check which hidden state we should get
        return ret


def test():
    from tqdm import tqdm

    args = get_args()
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = ZeroShotLlamaForSemAttack(args.model, cache_dir=args.cache_dir, torch_compile=False)
    model.to(device=device, dtype=torch.bfloat16).eval()

    for task in ALPACA_TASK_DESCRIPTION.keys():
        data_dir = os.path.join(args.cache_dir, f"glue-preprocessed-benign", args.model, task)
        test_data = load_from_disk(data_dir)
        test_data.set_format("pt", output_all_columns=True)

        generation_predictions = []
        dev_predictions = []
        dev_labels = []
        for data in tqdm(test_data):
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            pred = model(data)
            generate_ids = model.llama_classifier.generate(
                input_ids=data["input_ids"][:data["label_start_idx"]].unsqueeze(dim=0), max_new_tokens=5
            )
            generated_prediction = tokenizer.batch_decode(generate_ids)
            print(generated_prediction, tokenizer.batch_decode(data["input_ids"][:data["label_start_idx"]]))
            generation_predictions.append(generated_prediction)
            dev_predictions.append(pred["pred"].reshape(-1).argmax().item())
            dev_labels.append(data["label"].item())
        print(task, np.mean(np.array(dev_labels) == np.array(dev_predictions)))

        dest_path = os.path.join(f"./adv-glue/", args.model, task, "benign-zeroshot.json")
        if not os.path.exists(os.path.dirname(dest_path)):
            os.makedirs(os.path.dirname(dest_path))
        with open(dest_path, "w") as f:
            json.dump(
                {
                    "labels": dev_labels,
                    "lm_predictions": dev_predictions,
                    "generation_predictions": generation_predictions
                }, f, indent=4
            )


if __name__ == "__main__":
    test()
