'''
This example code shows how to use the PWWS attack model to attack a customized sentiment analysis model.
'''
import os
import OpenAttack
import numpy as np
import datasets
from datasets import load_dataset

import util
import json
import torch
from transformers import AutoTokenizer
from tokenization_alpaca import ALPACA_LABEL_CANDIDATE, ALPACA_TASK_DESCRIPTION, ALPACA_PROMPT_TEMPLATE, \
    GLUE_TASK_TO_KEYS, get_attack_target
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
        self.fixed_sentence = None
        self.fixed_idx = self.args.fix_sentence

    def fix_sentence(self, sent):
        self.fixed_sentence = sent

    def preprocess_function(self, sent):
        task_name = self.args.task
        assert task_name in ALPACA_LABEL_CANDIDATE
        sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task_name]
        example = {}
        for i, label in enumerate(ALPACA_LABEL_CANDIDATE[task_name]):
            sentence1 = self.fixed_sentence if self.fixed_idx == 0 else sent
            message = f"{sentence1_key}: {sentence1}"
            if sentence2_key:
                sentence2 = sent if self.fixed_idx == 0 else self.fixed_sentence
                message = f"{message}\n{sentence2_key}: {sentence2}"
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


def get_dataset_mapping(model, task, fix_id):
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[task]
    if task == 'sst2':
        assert fix_id == 1

    def dataset_mapping(x):
        target = get_attack_target(x, task)
        sentence1 = x[sentence1_key]
        if sentence2_key:
            sentence2 = x[sentence2_key]
        else:
            assert fix_id == 1
            sentence2 = None
        input_x = sentence2 if fix_id == 0 else sentence1
        fixed_x = sentence1 if fix_id == 0 else sentence2

        return_dict = {
            "x": input_x,
            "fixed_x": fixed_x,
            "y": 1 if x["label"] > 0.5 else 0,
            "target": target,
        }
        if model:
            model.fix_sentence(fixed_x)
            model["pred"] = model.get_pred([input_x])[0]

        return return_dict

    return dataset_mapping


def get_dataset(args, model):
    if args.task == 'mnli':
        split = 'validation_matched'
    elif args.task == 'mnli-mm':
        split = 'validation_mismatched'
    else:
        split = 'validation'
    if args.task in ['qqp', 'qnli', 'mnli', 'mnli-mm']:
        split += '[:1000]'
    dataset = load_dataset("glue", args.task.replace('-mm', ''), cache_dir=args.cache_dir, split=split)
    dataset = dataset.map(function=get_dataset_mapping(model, args.task, args.fix_sentence))
    return dataset


def main():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except RuntimeError:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    args = util.get_args()
    args.output_dir = os.path.join(args.output_dir, 'openattack', args.attack, args.task, str(args.fix_sentence))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0")
    victim = ZeroShotLlamaClassifier(args, device)

    dataset = get_dataset(args, model=victim)

    algorithm_to_attacker = {
        'textbugger': OpenAttack.attackers.TextBuggerAttacker,
        'textfooler': OpenAttack.attackers.TextFoolerAttacker,
        'sememepso': OpenAttack.attackers.PSOAttacker,
        'bertattack': OpenAttack.attackers.BERTAttacker,
        'bae': OpenAttack.attackers.BAEAttacker,
        'genetic': OpenAttack.attackers.GeneticAttacker,
        'pwws': OpenAttack.attackers.PWWSAttacker,
        'deepwordbug': OpenAttack.attackers.DeepWordBugAttacker,
    }
    print('Attack using', args.attack)
    attacker = algorithm_to_attacker[args.attack]()

    attack_eval = OpenAttack.AttackEval(attacker, victim)

    summary, results = attack_eval.eval(dataset, visualize=False, progress_bar=True)
    # attack_eval.eval(dataset, visualize=True, num_workers=16)  # TypeError: cannot pickle '_io.BufferedReader' object

    print('Saving results to {}'.format(args.output_dir))
    with open(os.path.join(args.output_dir, f"{args.attack}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    with open(os.path.join(args.output_dir, f"{args.attack}_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)


def test():
    args = util.get_args()
    dataset_mapping = get_dataset_mapping(args.model, args.task, args.fix_sentence)
    dataset = datasets.load_dataset("glue", args.task, cache_dir=args.cache_dir, split="validation").map(function=dataset_mapping)

    device = torch.device("cuda:1")
    victim = ZeroShotLlamaClassifier(args, device)

    for i in dataset:
        print(victim.get_prob([i['x']]))
        print(i['y'])


if __name__ == "__main__":
    main()
    # test()
