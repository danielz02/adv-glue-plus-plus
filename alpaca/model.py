from typing import Optional, Union, Tuple, List, Dict, Any

from torch.nn import CrossEntropyLoss
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel
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
        batch_size, n_labels, seq_len = input_ids.shape
        assert "classification_labels" in kwargs

        transformer_outputs = super().forward(
            input_ids.view(-1, seq_len),  # Benign inputs
            attention_mask=attention_mask.view(-1, seq_len),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.view(-1, seq_len),
            labels=labels.view(-1, seq_len),
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = transformer_outputs.loss.view(batch_size, n_labels)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(input=logits, target=kwargs["classification_labels"])

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,  # TODO: Needs reshaping. Leave it for now...
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class LlamaForSemAttackAdvGeneration(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.llama_classifier = LlamaForZeroShotSequenceClassification.from_pretrained(model_name)

    def forward(
        self, src: Dict[str, Union[torch.Tensor, List]], gold=None, perturbed=None, **kwargs
    ) -> Dict[str, Any]:
        # Assuming single input for now...
        instruction_token_embedding = self.llama_classifier.get_input_embeddings(src["instruction_token_ids"])
        if perturbed:
            input_token_embedding = perturbed
        else:
            input_token_embedding = self.llama_classifier.get_input_embeddings(src["input_token_ids"])
        response_header_token_embedding = self.llama_classifier.get_input_embeddings(
            src["response_header_token_ids"]
        )
        inputs_embeds = torch.stack([
            torch.cat([
                instruction_token_embedding,
                input_token_embedding,
                response_header_token_embedding,
                self.llama_classifier.get_input_embeddings(src[f"{label}_response_token_ids"])
            ]) for label in src["label_names"]
        ])  # (num_labels, seq_len, 4096)

        label_input_ids = torch.stack([
            torch.cat([
                torch.ones(instruction_token_embedding.size(0), device=instruction_token_embedding.device) * -100,
                torch.ones(input_token_embedding.size(0), device=instruction_token_embedding.device) * -100,
                torch.ones(response_header_token_embedding.size(0), device=instruction_token_embedding.device) * -100,
                src[f"{label}_response_token_ids"]
            ]) for label in src["label_names"]
        ])  # (num_labels, seq_len,)

        classifier_args = {
            "input_ids": None,
            "labels": label_input_ids.unsqueeze(0),
            "inputs_embeds": inputs_embeds.unsqueeze(0),
            "classification_labels": gold
        }

        out = self.llama_classifier(**classifier_args, **kwargs)
        embed = out[1]
        logits = self.proj(self.drop(embed))
        ret = {"pred": logits}
        if gold is not None:
            ret["loss"] = self.loss_fct(logits, gold)
        ret['embedding'] = out[0]
        return ret

