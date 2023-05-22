import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/stable-vicuna-13B-HF", cache_dir="./.cache/")
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/stable-vicuna-13B-HF", cache_dir="./.cache/", torch_dtype=torch.bfloat16
)
model = model.to(device=device)
dataset = load_dataset("glue", "sst2", split="validation")
dataset.set_format(type="pt", output_all_columns=True)

prompt = """\
### Human: {qs}
### Assistant:\
"""

ans = []
labels = []
for data in tqdm(dataset):
    qs = f"For the given input text, label the sentiment of the text as positive or negative. The answer should be " \
         f"exactly 'positive' or 'negative'.\nsentence: {data['sentence']}"
    inputs = tokenizer(prompt.format(qs=qs), return_tensors='pt').to(device)
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=3,
        do_sample=True,
        temperature=0.01,
        top_p=1.0,
    )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    ans += outputs
    labels.append(data["label"].item())
    print(outputs)

with open("./.cache/stable-vicuna_sst2.json", "w") as f:
    json.dump({"ans": ans, "labels": labels}, f)
