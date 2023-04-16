from typing import Optional, Union, Tuple, List

from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, CausalLMOutputWithPast
from torch import nn
from torch.nn.functional import softmax
import torch
import util


instruction = {
    "sst2": "For the given input text, label the sentiment of the text as positive or negative. The answer should be "
            "exact 'positive' or 'negative'."
}

candidate_labels = {
    "sst2": ["negative", "positive"]
}

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{user_input}

### Response:

"""


class Model(nn.Module):
    def __init__(self, task, model_name, cache_dir, device):
        super(Model, self).__init__()
        self.task = task
        self.device = device

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.to(self.device)
        self.eval()

        self.label_tokens = self.tokenize_labels()

    def tokenize_labels(self):
        self.tokenizer.add_eos_token = True
        self.tokenizer.add_bos_token = False
        return [self.tokenizer(x, return_tensors="pt").to(self.device) for x in candidate_labels[self.task]]

    def tokenize(self, data):
        key1, key2 = util.task_to_keys[self.task]
        message = f"{key1}: {data[key1]}"
        if key2:
            message = f"{message}\n{key2}: {data[key2]}"
        message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')

        prompt = prompt_template.format(instruction=instruction[self.task], user_input=message)
        self.tokenizer.add_eos_token = False
        self.tokenizer.add_bos_token = True
        prompt_token = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        return prompt_token

    def get_logits(self, prompt_token, label_token):
        labels = torch.hstack([torch.ones_like(prompt_token.input_ids) * -100, label_token.input_ids])
        label_out = self.model(input_ids=torch.hstack([prompt_token.input_ids, label_token.input_ids]), labels=labels)

        shift_logits = label_out.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        log_probs = torch.sum(-loss)

        return log_probs

    def forward(self, prompt_token, inputs_embeds=None):
        label_probs = torch.cat([get_logits(prompt_token, label_token).reshape(1) for label_token in self.label_tokens])
        return SequenceClassifierOutput(logits=label_probs)


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
