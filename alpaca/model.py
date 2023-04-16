from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput
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
