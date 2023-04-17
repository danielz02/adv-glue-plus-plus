import os
import util
import copy
import torch
import joblib
import numpy as np
from tqdm import tqdm
from CW_attack import CarliniL2
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import ZeroShotLlamaForSemAttack


def transform(seq, unk_words_dict=None):
    if unk_words_dict is None:
        unk_words_dict = {}
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    unk_count = 0
    for x in seq:
        if x == util.unk_id:
            unk_count += 1
    if unk_count == 0:
        return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq if x not in [1, 2]])
    else:
        tokens_lists = [[]]
        for idx, x in enumerate(seq):
            if x in [util.bos_id, util.eos_id]:
                continue
            if x == util.unk_id:
                unk_words = unk_words_dict[idx]
                cur_size = len(tokens_lists)
                size = len(unk_words)
                new_tokens_lists = []
                for copy_time in range(size):
                    if len(new_tokens_lists) > 100:
                        continue
                    new_tokens_lists += copy.deepcopy(tokens_lists)
                tokens_lists = new_tokens_lists
                for unk_idx in range(size):
                    for i in range(cur_size):
                        full_idx = unk_idx * cur_size + i
                        if full_idx < len(tokens_lists):
                            tokens_lists[full_idx].append(unk_words[unk_idx])
            else:
                for tokens_idx in range(len(tokens_lists)):
                    tokens_lists[tokens_idx].append(tokenizer._convert_id_to_token(x))
        return [tokenizer.convert_tokens_to_string(tokens) for tokens in tokens_lists]


def init_dict():
    return {util.bos_id: [util.bos_id], util.eos_id: [util.eos_id]}


def get_word_from_token(token):
    token.lower()


def transform_token(orig_token, new_word_list):
    if orig_token.lower().capitalize() == orig_token:
        return [new_word.capitalize() for new_word in new_word_list]
    else:
        return new_word_list


def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1

    return tot


def get_cluster_dict(input_cluster_dict, input_ids):
    cluster_dict = init_dict()
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    for i in range(len(token_list)):
        if input_ids[i] in cluster_dict:
            continue
        word = get_word_from_token(token_list[i])
        if word not in input_cluster_dict:
            cluster_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_cluster_dict[word]
        candidates = [x[0] for x in candidates]
        candidates = transform_token(token_list[i], candidates)
        candidates = [tokenizer._convert_token_to_id(x) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        while util.unk_id in candidates:
            candidates.remove(util.unk_id)
        cluster_dict[input_ids[i]] = candidates

    return cluster_dict


def get_knowledge_dict(input_knowledge_dict, input_ids):
    knowledge_dict = init_dict()
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    for i in range(len(token_list)):
        if input_ids[i] in knowledge_dict:
            continue
        word = get_word_from_token(token_list[i])
        if word not in input_knowledge_dict:
            knowledge_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_knowledge_dict[word]
        candidates = [x[0] for x in candidates]
        candidates = transform_token(token_list[i], candidates)
        candidates = [tokenizer._convert_token_to_id(x) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        while util.unk_id in candidates:
            candidates.remove(util.unk_id)
        knowledge_dict[input_ids[i]] = candidates

    return knowledge_dict


def get_typo_dict(input_typo_dict, input_ids):
    typo_dict = init_dict()
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    unk_words_dict = {}
    for i in range(len(token_list)):
        if input_ids[i] in typo_dict:
            for j in range(len(token_list)):
                if input_ids[i] == input_ids[j]:
                    if j in unk_words_dict:
                        unk_words_dict[i] = unk_words_dict[j]
                    break
            continue
        word = get_word_from_token(token_list[i])
        if word not in input_typo_dict:
            typo_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_typo_dict[word]
        candidates = [x[0] for x in candidates]
        candidates = transform_token(token_list[i], candidates)
        unk_words_dict[i] = [x for x in candidates if tokenizer._convert_token_to_id(x) == util.unk_id]
        candidates = [tokenizer._convert_token_to_id(x) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        typo_dict[input_ids[i]] = candidates

    return typo_dict, unk_words_dict


def cw_word_attack(data_val):
    logger.info("Begin Attack")
    logger.info(("const confidence lr:", args.const, args.confidence, args.lr))

    orig_failures = 0
    adv_correct = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    adv_pickle = []
    changed_rates = []
    nums_changed = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    text_len = []

    ori_labels = []
    ori_preds = []
    preds = []

    test_batch = DataLoader(data_val, batch_size=1, shuffle=False)
    cw = CarliniL2(args, logger, debug=False, targeted=True, cuda=True)
    for batch_index, batch in enumerate(tqdm(test_batch)):
        inputs = tokenizer(batch['sentence'][0], return_tensors="pt", add_special_tokens=False)
        batch_add_start = batch['add_start'] = []
        batch_add_end = batch['add_end'] = []
        batch['seq_len'] = []
        for i, sentence in enumerate(batch['sentence']):
            batch['add_start'].append(1)
            batch['add_end'].append(inputs['input_ids'][i].cpu().numpy().tolist().index(util.eos_id) - 1)
            batch['seq_len'].append(batch['add_end'][i])
        tot += len(batch['label'])
        inputs = {k: v.to(device) for (k, v) in inputs.items()}

        if batch['label'][0] == 1:
            attack_targets = torch.full_like(batch['label'], 0)
        else:
            attack_targets = torch.full_like(batch['label'], 1)
        label = batch['label'] = batch['label'].to(device)
        attack_targets = attack_targets.to(device)

        # test original acc
        out = model(**inputs).logits
        prediction = torch.max(out, 1)[1]
        ori_prediction = prediction
        batch['orig_correct'] = torch.sum((prediction == label).float())
        if prediction.item() != label.item():
            orig_failures += 1
            continue

        # prepare attack
        input_embedding = model.get_input_embedding_vector(inputs['input_ids'])
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for i, sentence in enumerate(batch['sentence']):
            cw_mask[i][batch['add_start'][i]:batch['add_end'][i]] = 1

        cluster_char_dict = get_cluster_dict(batch['cluster_dict'], inputs['input_ids'])
        typo_dict, unk_words_dict = get_typo_dict(batch['typo_dict'], inputs['input_ids'])
        knowledge_dict = get_knowledge_dict(batch['knowledge_dict'], inputs['input_ids'])

        for k, v in cluster_char_dict.items():
            synset = list(set(v + knowledge_dict[k]))
            knowledge_dict[k] = synset

        for k, v in typo_dict.items():
            synset = list(set(v + knowledge_dict[k]))
            knowledge_dict[k] = synset

        cw.wv = knowledge_dict
        cw.mask = cw_mask
        cw.seq = inputs['input_ids']
        cw.batch_info = batch
        cw.num_classes = len(model.model.config.label2id)

        # attack
        adv_data = cw.run(model, input_embedding, attack_targets, inputs)
        # retest
        adv_seq = torch.tensor(inputs['input_ids']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for i in range(add_start, add_end):
                    adv_seq.data[bi, i] = knowledge_dict[adv_seq.data[bi, i].item()][cw.o_best_sent[bi][i - add_start]]
        adv_inputs = copy.deepcopy(inputs)
        adv_inputs['input_ids'] = adv_seq

        out = model(**adv_inputs).logits
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()

        for i in range(len(batch['label'])):
            diff = difference(adv_seq[i], inputs['input_ids'][i])
            tot_diff += diff
            tot_len += batch['seq_len'][i]
            changed_rate = 1.0 * diff / batch['seq_len'][i]
            if ori_prediction[i].item() == label[i].item() and prediction[i].item() == attack_targets[i].item():
                changed_rates.append(changed_rate)
                nums_changed.append(diff)
                orig_texts.append(transform(inputs['input_ids'][i]))
                adv_texts.append(transform(adv_seq[i], unk_words_dict))
                true_labels.append(label[i].item())
                new_labels.append(prediction[i].item())
                text_len.append(batch['seq_len'][i])
            ori_labels.append(label[i].item())
            ori_preds.append(ori_prediction[i].item())
            preds.append(prediction[i].item())

    message = 'For target model {}:\noriginal accuracy: {:.2f}%,\nadv accuracy: {:.2f}%,\n' \
              'attack success rates: {:.2f},\navg changed rate: {:.02f}%\n'.format(args.model,
                                                                                   (1 - orig_failures / len(test_batch)) * 100,
                                                                                   (adv_correct / len(test_batch)) * 100,
                                                                                   len(adv_texts) / (len(test_batch) - orig_failures) * 100,
                                                                                   np.mean(changed_rates) * 100)
    logger.info(message)

    joblib.dump({'ori_labels': ori_labels, 'ori_preds': ori_preds, 'preds': preds},
                os.path.join(args.output_dir, 'labels.pkl'))

    results = []
    for i in range(len(changed_rates)):
        save_dict = {}
        save_dict['orig_text'] = orig_texts[i]
        save_dict['orig_y'] = true_labels[i]
        save_dict['pred_y'] = new_labels[i]
        save_dict['diff'] = nums_changed[i]
        save_dict['diff_ratio'] = changed_rates[i]
        save_dict['seq_len'] = text_len[i]
        if isinstance(adv_texts[i], str):
            save_dict['adv_text'] = adv_texts[i]
            results.append(save_dict)
        else:
            for t in adv_texts[i]:
                new_save_dict = copy.deepcopy(save_dict)
                new_save_dict['adv_text'] = t
                results.append(new_save_dict)
    joblib.dump(results, os.path.join(args.output_dir, 'attack_results.pkl'))


if __name__ == '__main__':
    args = util.get_args()
    args.output_dir = os.path.join(args.output_dir, args.model, args.task)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    logger = util.init_logger(args.output_dir)

    device = torch.device("cuda:0")

    # FIXME: Hard code for now
    model = ZeroShotLlamaForSemAttack(args.model, args.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model.to(device)
    model.eval()

    # Set the random seed manually for reproducibility.
    util.set_seed(args.seed)
    # TODO: load pre-precessed dataset with candidate perturbation set
    test_data = load_dataset("glue", args.task, cache_dir=args.cache_dir, split="validation")

    cw_word_attack(test_data)
