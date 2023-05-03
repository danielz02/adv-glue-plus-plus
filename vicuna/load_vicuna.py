import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-13B-1.1-HF", cache_dir="./cache")
model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-13B-1.1-HF", cache_dir="./cache")
model = model.half().to(device)
model = torch.compile(model)
