import json
from copy import deepcopy

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from vicuna.tokenization_vicuna import Conversation, SeparatorStyle

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/vicuna-13B-1.1-HF",
    cache_dir="./.cache",
    format="pt",
    use_fast=False
)
model = LlamaForCausalLM.from_pretrained(
    "TheBloke/vicuna-13B-1.1-HF",
    cache_dir="./.cache",
    torch_dtype=torch.bfloat16
)
model = model.to(device=device)
dataset = load_dataset("glue", "sst2", split="validation")
dataset.set_format(type="pt", output_all_columns=True)

conversation = Conversation(
    system="",  # "A chat between a curious user and an artificial intelligence assistant. "
           # "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

ans = []
labels = []
for data in tqdm(dataset):
    conv = deepcopy(conversation)
    qs = f"For the given input text, label the sentiment of the text as positive or negative. The answer should be " \
         f"exactly 'positive' or 'negative'.\nsentence: {data['sentence']}"
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.01,
        max_new_tokens=1024,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    ans.append(outputs)
    labels.append(data["label"].item())

with open("./.cache/vicuna_sst2.json", "w") as f:
    json.dump({"ans": ans, "labels": labels}, f)
