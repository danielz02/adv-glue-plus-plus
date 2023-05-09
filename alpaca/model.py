from typing import Optional, Union, Tuple, List, Dict, Any

from torch.nn import CrossEntropyLoss
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from torch import nn
from torch.nn.functional import softmax
import torch
import util


class LlamaForZeroShotSequenceClassification(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

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
            inputs_embeds = inputs_embeds.view(-1, seq_len, 4096)
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, seq_len)
        if labels is not None:
            labels = labels.view(-1, seq_len)

        assert "classification_labels" in kwargs

        decoder_outputs = super().forward(
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
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
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


class ZeroShotLlamaForSemAttack(nn.Module):
    def __init__(self, model_name, cache_dir):
        super().__init__()
        self.llama_classifier = LlamaForZeroShotSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

    def get_input_embedding_vector(self, input_ids: torch.LongTensor):
        return self.llama_classifier.get_input_embeddings()(input_ids)

    def forward(
        self, input_dict: Dict[str, Union[torch.Tensor, List]], gold=None, perturbed=None, **kwargs
    ) -> Dict[str, Any]:
        # Assuming single input_embedding for now...
        instruction_token_embedding = self.llama_classifier.get_input_embeddings()(input_dict["instruction_token_ids"])
        if perturbed is not None:
            input_token_embedding = perturbed
        else:
            input_token_embedding = self.llama_classifier.get_input_embeddings()(input_dict["input_token_ids"])
        response_header_token_embedding = self.llama_classifier.get_input_embeddings()(
            input_dict["response_header_token_ids"]
        )
        inputs_embeds = torch.stack([
            torch.cat([
                instruction_token_embedding,
                input_token_embedding,
                response_header_token_embedding,
                self.llama_classifier.get_input_embeddings()(input_dict[f"{label}_response_token_ids"])
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
    from datasets import load_dataset

    device = torch.device("cuda:0")

    test_data = load_dataset("glue", "sst2", cache_dir="./.cache/", split="validation")
    test_data = test_data.load_from_disk(f"./.cache/sst2/")
    test_data.set_format("pt", output_all_columns=True)

    model = ZeroShotLlamaForSemAttack("chavinlo/alpaca-native", cache_dir="./.cache/")
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    sst2_dev_predictions = []
    for data in tqdm(test_data):
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        pred = model(data, perturbed=torch.randn(50, 4096, device=device))
        print(pred["pred"].reshape(-1).argmax().item())
        sst2_dev_predictions.append(pred["pred"].reshape(-1).argmax().item())

    with open("./adv-glue/sst-2-dev-alpaca-zeroshot.json", "w") as f:
        import json
        json.dump(sst2_dev_predictions, f, indent=4)


if __name__ == "__main__":
    test()
