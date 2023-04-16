from typing import Optional, Union, Tuple, List

from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithPast
from torch import nn
from torch.nn.functional import softmax
import torch


class RobertaEnsembleForSequenceClassification(nn.Module):
    def __init__(self, dir0, dir1, dir2):
        super().__init__()
        self.roberta0 = RobertaForSequenceClassification.from_pretrained(dir0)
        self.roberta1 = RobertaForSequenceClassification.from_pretrained(dir1)
        self.roberta2 = RobertaForSequenceClassification.from_pretrained(dir2)
        self.softmax = nn.Softmax(dim=1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        out0 = self.roberta0(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                             labels=labels, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)
        out1 = self.roberta1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                             labels=labels, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)
        out2 = self.roberta2(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                             labels=labels, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict)

        # Exception: stsb
        logits = (self.softmax(out0.logits) + self.softmax(out1.logits) + self.softmax(out2.logits)) / 3

        return SequenceClassifierOutput(logits=logits)


class Model(nn.Module):
    def __init__(self, task, model_name, pretrained_dir):
        super(Model, self).__init__()
        self.task = task
        if model_name == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir)
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
        elif model_name == 'roberta':
            self.model = RobertaForSequenceClassification.from_pretrained(pretrained_dir)
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_dir)
        else:
            dir0 = pretrained_dir
            dir1 = pretrained_dir.replace('roberta', 'roberta1')
            dir2 = pretrained_dir.replace('roberta', 'roberta2')
            self.model = RobertaEnsembleForSequenceClassification(dir0, dir1, dir2)
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_dir)
        self.model.eval()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                          labels=labels, output_attentions=output_attentions,
                          output_hidden_states=output_hidden_states, return_dict=return_dict)


class LlamaForZeroShotSequenceClassification(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        pass
