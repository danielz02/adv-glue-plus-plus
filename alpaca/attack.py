import json
import random
import codecs
import joblib
import os
import numpy as np
import torch
from datasets import load_dataset
import copy
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from CW_attack import CarliniL2
import util
# from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import LlamaTokenizer, LlamaForCausalLM


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
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    adv_pickle = []

    test_batch = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    cw = CarliniL2(debug=args.debugging, targeted=not args.untargeted, cuda=True)
    for batch_index, batch in enumerate(tqdm(test_batch)):
        batch_add_start = batch['add_start'] = []
        batch_add_end = batch['add_end'] = []
        for i, seq in enumerate(batch['seq_len']):
            batch['add_start'].append(1)
            batch['add_end'].append(seq)

        data = batch['seq'] = torch.stack(batch['seq']).t().to(device)
        orig_sent = transform(batch['seq'][0])

        seq_len = batch['seq_len'] = batch['seq_len'].to(device)
        if args.untargeted:
            attack_targets = batch['label']
        else:
            if args.strategy == 0:
                if batch['label'][0] == 1:
                    attack_targets = torch.full_like(batch['label'], 0)
                else:
                    attack_targets = torch.full_like(batch['label'], 1)
            elif args.strategy == 1:
                if batch['label'][0] < 2:
                    attack_targets = torch.full_like(batch['label'], 4)
                else:
                    attack_targets = torch.full_like(batch['label'], 0)
        label = batch['label'] = batch['label'].to(device)
        attack_targets = attack_targets.to(device)

        # test original acc
        out = model(batch['seq'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        ori_prediction = prediction
        if ori_prediction[0].item() != label[0].item():
            continue
        batch['orig_correct'] = torch.sum((prediction == label).float())

        # prepare attack
        input_embedding = model.bert.embeddings.word_embeddings(data)
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for i, seq in enumerate(batch['seq_len']):
            cw_mask[i][1:seq] = 1

        if args.function == 'all':
            cluster_char_dict = get_similar_dict(batch['similar_dict'])
            bug_char_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['seq'][0])
            similar_char_dict = get_knowledge_dict(batch['knowledge_dict'])

            for k, v in cluster_char_dict.items():
                synset = list(set(v + similar_char_dict[k]))
                while 100 in synset:
                    synset.remove(100)
                if len(synset) >= 1:
                    similar_char_dict[k] = synset
                else:
                    similar_char_dict[k] = [k]

            for k, v in bug_char_dict.items():
                synset = list(set(v + similar_char_dict[k]))
                # while 100 in synset:
                #     synset.remove(100)
                if len(synset) >= 1:
                    similar_char_dict[k] = synset
                else:
                    similar_char_dict[k] = [k]

            all_dict = similar_char_dict
        elif args.function == 'typo':
            all_dict, unk_words_dict = get_bug_dict(json.loads(batch['bug_dict']), batch['seq'][0])
        elif args.function == 'knowledge':
            all_dict = get_knowledge_dict(json.loads(batch['knowledge_dict']))
            unk_words_dict = None
        elif args.function == 'cluster':
            all_dict = get_similar_dict(batch['similar_dict'])
            unk_words_dict = None
        else:
            raise Exception('Unknown perturbation function.')

        cw.wv = all_dict
        cw.mask = cw_mask
        cw.seq = data
        cw.batch_info = batch
        cw.seq_len = seq_len

        # attack
        adv_data = cw.run(model, input_embedding, attack_targets)
        # retest
        adv_seq = torch.tensor(batch['seq']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for i in range(add_start, add_end):
                    adv_seq.data[bi, i] = all_dict[adv_seq.data[bi, i].item()][cw.o_best_sent[bi][i - add_start]]

        out = model(adv_seq, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += torch.sum((prediction != label).float()).item()
        tot += len(batch['label'])

        for i in range(len(batch['label'])):
            diff = difference(adv_seq[i], data[i])
            adv_pickle.append({
                'index': batch_index,
                'adv_text': transform(adv_seq[i], unk_words_dict),
                'orig_text': transform(batch['seq'][i]),
                'raw_text': batch['raw_text'][i],
                'label': label[i].item(),
                'target': attack_targets[i].item(),
                'ori_pred': ori_prediction[i].item(),
                'pred': prediction[i].item(),
                'diff': diff,
                'orig_seq': batch['seq'][i].cpu().numpy().tolist(),
                'adv_seq': adv_seq[i].cpu().numpy().tolist(),
                'seq_len': batch['seq_len'][i].item()
            })
            assert ori_prediction[i].item() == label[i].item()
            if (args.untargeted and prediction[i].item() != label[i].item()) or (not args.untargeted and prediction[i].item() == attack_targets[i].item()):
                tot_diff += diff
                tot_len += batch['seq_len'][i].item()
                if batch_index % 100 == 0:
                    try:
                        logger.info(("tot:", tot))
                        logger.info(("avg_seq_len: {:.1f}".format(tot_len / tot)))
                        logger.info(("avg_diff: {:.1f}".format(tot_diff / tot)))
                        logger.info(("avg_diff_rate: {:.1f}%".format(tot_diff / tot_len * 100)))
                        logger.info(("orig_correct: {:.1f}%".format(orig_correct / tot * 100)))
                        logger.info(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
                        if args.untargeted:
                            logger.info(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                            logger.info(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                        else:
                            logger.info(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                            logger.info(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                    except:
                        continue
    joblib.dump(adv_pickle, os.path.join(root_dir, 'adv_text.pkl'))
    logger.info(("tot:", tot))
    logger.info(("avg_seq_len: {:.1f}".format(tot_len / tot)))
    logger.info(("avg_diff: {:.1f}".format(tot_diff / tot)))
    logger.info(("avg_diff_rate: {:.1f}%".format(tot_diff / tot_len * 100)))
    logger.info(("orig_correct: {:.1f}%".format(orig_correct / tot * 100)))
    logger.info(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
    if args.untargeted:
        logger.info(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
        logger.info(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
    else:
        logger.info(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
        logger.info(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
    logger.info(("const confidence:", args.const, args.confidence))


def check_consistency():
    logger.info("Start checking consistency")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        i['adv_text'] = i['adv_text'].replace('[CLS] ', '')

    adv_text = YelpDataset(adv_text, raw=True)
    test_batch = DataLoader(adv_text, batch_size=args.batch_size, shuffle=False)

    inconsistent = []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_batch)):
            batch['seq'] = torch.stack(batch['seq']).t().to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            out = model(batch['seq'], batch['seq_len'])
            logits = out['pred'].detach().cpu()
            pred = logits.argmax(dim=-1)
            if pred[0].item() != batch['pred'][0]:
                inconsistent.append((bi, batch))

    logger.info("Num of inconsistent: {}".format(len(inconsistent)))
    if len(inconsistent) != 0:
        joblib.dump(inconsistent, os.path.join(root_dir, 'inconsistent_adv.pkl'))

    return adv_text


if __name__ == '__main__':
    args = util.get_args()
    args.output_dir = os.path.join(args.output_dir, args.model, args.task)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)
    logger = util.init_logger(args.output_dir)

    device = torch.device("cuda:0")
    tokenizer = LlamaTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = LlamaForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
    model.to(device)
    model.eval()

    # Set the random seed manually for reproducibility.
    util.set_seed(args.seed)
    # TODO: load pre-precessed dataset with candidate perturbation set
    test_data = load_dataset("glue", args.task, cache_dir=args.cache_dir, split="validation")

    cw_word_attack(test_data)
