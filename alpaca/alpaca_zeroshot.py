import json

import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


tasks = ["sst2"]
task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}

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


def tokenize(tok, prompt, labels, dev):
    tok.add_eos_token = False
    tok.add_bos_token = True
    prompt_token = tok(prompt, return_tensors="pt").to(dev)

    tok.add_eos_token = True
    tok.add_bos_token = False
    label_tokens = [tok(x, return_tensors="pt").to(dev) for x in labels]

    return prompt_token, label_tokens


def get_logits(prompt_token, label_token, model):
    labels = torch.hstack([torch.ones_like(prompt_token.input_ids) * -100, label_token.input_ids])
    label_out = model(input_ids=torch.hstack([prompt_token.input_ids, label_token.input_ids]), labels=labels)

    shift_logits = label_out.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    log_probs = torch.sum(-loss)

    return log_probs


def eval_dataset(task, tokenizer, model, device):
    dataset = load_dataset("glue", task, cache_dir="/scratch/bbkc/danielz/.cache/", split="validation")
    key1, key2 = task_to_keys[task]

    log_probs = []
    predictions = []
    for data in tqdm(dataset):
        message = f"{key1}: {data[key1]}"
        if key2:
            message = f"{message}\n{key2}: {data[key2]}"
        message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')

        prompt = prompt_template.format(instruction=instruction[task], user_input=message)
        prompt_token, label_tokens = tokenize(tokenizer, prompt, candidate_labels[task], device)

        label_probs = torch.cat(
            [get_logits(prompt_token, label_token, model).reshape(1) for label_token in label_tokens])
        prediction = int(label_probs.detach().cpu().numpy().argmax())

        predictions.append(prediction)
        log_probs.append(label_probs.detach().cpu().numpy().astype(float))

    return predictions


def main():
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")
    model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")
    model.to(device)
    model.eval()

    sst2_dev_predictions = eval_dataset("sst2", tokenizer, model, device)

    with open("./adv-glue/sst-2-dev-alpaca-zeroshot.json", "w") as f:
        json.dump(sst2_dev_predictions, f, indent=4)


if __name__ == "__main__":
    main()
