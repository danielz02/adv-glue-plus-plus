'''
This example code shows how to use the PWWS attack model to attack a customized sentiment analysis model.
'''
import OpenAttack
import numpy as np
import datasets
import util
import torch
from transformers import AutoTokenizer
from tokenization_alpaca import ALPACA_LABEL_CANDIDATE, ALPACA_TASK_DESCRIPTION, ALPACA_PROMPT_TEMPLATE, GLUE_TASK_TO_KEYS
from model import ZeroShotLlamaForSemAttack
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# configure access interface of the customized victim model by extending OpenAttack.Classifier.
class ZeroShotLlamaClassifier(OpenAttack.Classifier):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = ZeroShotLlamaForSemAttack(args.model, args.cache_dir).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<l>", "<i>", "</i>"]})

    def preprocess_function(self, sent):
        task_name = self.args.task
        assert task_name in ALPACA_LABEL_CANDIDATE
        sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
        example = {}
        for i, label in enumerate(ALPACA_LABEL_CANDIDATE[task_name]):
            sentence1 = sent
            message = f"{sentence1_key}: {sentence1}"
            message = f"{message}".replace('sentence1', 'premise').replace('sentence2', 'hypothesis')
            prompt = ALPACA_PROMPT_TEMPLATE.format(
                instruction=ALPACA_TASK_DESCRIPTION[task_name], input=message, label=f"{label}"
            )
            tokens = self.tokenizer.tokenize(prompt)
            input_start_idx = tokens.index("<i>")
            tokens.remove("<i>")
            input_end_idx = tokens.index("</i>")
            tokens.remove("</i>")
            label_start_idx = tokens.index("<l>")
            tokens.remove("<l>")
            token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long).to(self.device)

            instruction_token_ids = token_ids[:input_start_idx].clone()
            input_token_ids = token_ids[input_start_idx:input_end_idx].clone()
            response_header_token_ids = token_ids[input_end_idx:label_start_idx].clone()
            response_token_ids = token_ids[label_start_idx:].clone()

            # token_type_ids = torch.zeros_like(token_ids)
            # token_type_ids[input_start_idx:input_end_idx] = 1
            # token_type_ids[input_end_idx:label_start_idx] = 0
            # token_type_ids[label_start_idx:-1] = 2

            # if i == example["label"]:
            #     example["input_ids"] = token_ids
            example[f"instruction_token_ids"] = instruction_token_ids
            example[f"input_token_ids"] = input_token_ids
            example[f"response_header_token_ids"] = response_header_token_ids
            example[f"{label}_response_token_ids"] = response_token_ids
            example["label_names"] = ALPACA_LABEL_CANDIDATE[task_name]

            example["input_start_idx"] = input_start_idx
            example["input_end_idx"] = input_end_idx
            example["label_start_idx"] = label_start_idx

        return example

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = []
        for sent in input_:
            data = self.preprocess_function(sent)
            res = self.model(data)
            prob = torch.nn.functional.softmax(res["pred"].reshape(-1), dim=0)
            ret.append(prob.detach().cpu().numpy())

        # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
        return np.array(ret)


def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }


def main():
    args = util.get_args()
    dataset = datasets.load_dataset("glue", args.task, cache_dir=args.cache_dir, split="validation").map(function=dataset_mapping)

    device = torch.device("cuda:0")
    victim = ZeroShotLlamaClassifier(args, device)

    attacker = OpenAttack.attackers.TextFoolerAttacker()
    # attacker = OpenAttack.attackers.TextBuggerAttacker()

    attack_eval = OpenAttack.AttackEval(attacker, victim)

    attack_eval.eval(dataset, visualize=True)
    # attack_eval.eval(dataset, visualize=True, num_workers=16)  # TypeError: cannot pickle '_io.BufferedReader' object


def test():
    args = util.get_args()
    dataset = datasets.load_dataset("glue", args.task, cache_dir=args.cache_dir, split="validation").map(function=dataset_mapping)

    device = torch.device("cuda:1")
    victim = ZeroShotLlamaClassifier(args, device)

    for i in dataset:
        print(victim.get_prob([i['x']]))
        print(i['y'])


if __name__ == "__main__":
    main()
    # test()
