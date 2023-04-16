from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
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
