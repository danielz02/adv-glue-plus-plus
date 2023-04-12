# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Preprocessing the data."""

import os
import pickle

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.bert.modeling_bert import BertModel
import sqlite3 as sql
import re
import numpy as np
import umap
import json
from tqdm import tqdm
import nltk

DB_PATH = './corpora/enwiki-20170820.db'
nltk.download('averaged_perceptron_tagger', download_dir="./corpora")
nltk.download('punkt', download_dir="./corpora/")
nltk.data.path = "./corpora/wordnet"


def neighbors(word, sentences, model, tokenizer, device):
    """Get the info and (umap-projected) embeddings about a word."""
    # Get part of speech of this word.
    # sent_data = get_poses(word, sentences)

    # Get embeddings.
    points = get_embeddings(word.lower(), sentences, model, tokenizer, device)

    # Use UMAP to project down to 3 dimensions.
    # points_transformed = project_umap(points)

    # {'labels': sent_data, 'data': points_transformed, 'points': points}
    return {'word': word, 'points': points}


def project_umap(points):
    """Project the words (by layer) into 3 dimensions using umap."""
    points_transformed = []
    for layer in points:
        transformed = umap.UMAP().fit_transform(layer).tolist()
        points_transformed.append(transformed)
    return points_transformed


def get_embeddings(word, sentences, model, tokenizer, device):
    """Get the embedding for a word in each sentence."""
    # Tokenized input
    layers = range(-model.config.num_hidden_layers, 0)
    points = [[] for layer in layers]
    print('Getting embeddings for %d sentences ' % len(sentences))
    for sentence in sentences:
        sentence = '</s> ' + sentence + ' </s>'  # Changed to LlaMA bos and eos tokens
        tokenized_text = tokenizer.tokenize(sentence)  # </s> is not added automatically when calling tokenizer.tokenize

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        # should give you something like [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        # bos token is the same as the eos token in LlaMA
        sep_idxs = [-1] + [i for i, v in enumerate(tokenized_text) if v == '</s>' and i != 0]
        segments_ids = []
        for i in range(len(sep_idxs) - 1):
            segments_ids += [i] * (sep_idxs[i + 1] - sep_idxs[i])

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens], device=device)
        segments_tensors = torch.tensor([segments_ids], device=device)

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers = model(
                input_ids=tokens_tensor, attention_mask=segments_tensors, output_hidden_states=True
            ).hidden_states
            encoded_layers = [l.cpu() for l in encoded_layers]

        # We have a hidden states for each of the 12 layers in model bert-base-uncased
        encoded_layers = [l.numpy() for l in encoded_layers]
        try:
            word_idx = tokenized_text.index(word)
        # If the word is made up of multiple tokens, just use the first one of the tokens that make it up.
        except ValueError as e:
            word_idx = None
            for i, token in enumerate(tokenized_text):
                if token == word[:len(token)]:
                    word_idx = i
            print(e, word, tokenized_text, tokenized_text[word_idx])

        # Reconfigure to have an array of layer: embeddings
        for l in layers:
            sentence_embedding = encoded_layers[l][0][word_idx]
            points[l].append(sentence_embedding)

    points = np.asarray(points)
    return points


def tokenize_sentences(text):
    """Simple tokenizer."""
    print('starting tokenization')

    text = re.sub('\n', ' ', text)
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Filter out too long sentences.
    sentences = [t for t in sentences if len(t) < 150]

    return sentences


def get_query(select, db=DB_PATH):
    """Executes a select statement and returns results and column/field names."""
    with sql.connect(db) as conn:
        c = conn.cursor()
        c.execute(select)
        col_names = [str(name[0]).lower() for name in c.description]
    return c.fetchall(), col_names


def get_sentences():
    """Returns a bunch of sentences from wikipedia"""
    print('Selecting sentences from wikipedia...')

    select = 'select * from articles limit 5000000'
    docs, _ = get_query(select)
    docs = [doc[3] for doc in docs]
    doc = ' '.join(docs)
    print('Number of articles selected: %d' % len(docs))

    sentences = tokenize_sentences(doc)
    print('Total number of sentences: %d' % len(sentences))
    np.random.shuffle(sentences)
    return sentences


def get_poses(word, sentences):
    """Get the part of speech tag for the given word in a list of sentences."""
    sent_data = []
    for sent in sentences:
        text = nltk.word_tokenize(sent)
        pos = nltk.pos_tag(text)
        try:
            word_idx = text.index(word)
            pos_tag = pos[word_idx][1]
        except:
            pos_tag = 'X'
        sent_data.append({
            'sentence': sent,
            'pos': pos_tag
        })

    return sent_data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device : ", device)

    word_list = []
    len_list = [0, ]
    pointer = 0
    s = torch.zeros((13928506, 768), device=torch.device("cpu"))

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="/scratch/bbkc/danielz/.cache/")
    # Load pre-trained model (weights)
    model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", cache_dir="/scratch/bbkc/danielz/.cache/")
    model.eval()
    model = model.to(device)

    # Get selection of sentences from wikipedia.
    with open('static/words.json', "r") as f:
        words = json.load(f)

    if os.path.exists("./static/sentences.pkl"):
        sentences = joblib.load("./static/sentences.pkl")
    else:
        sentences = get_sentences()
        joblib.dump(sentences, "./static/sentences.pkl")

    for word in tqdm(words):
        # Filter out sentences that don't have the word.
        sentences_w_word = [t for t in sentences if ' ' + word + ' ' in t]

        # Take at most 200 sentences.
        sentences_w_word = sentences_w_word[:100]  # Changed from default

        # And don't show anything if there are less than 100 sentences.
        if len(sentences_w_word) > 50:  # Changed from default
            print('starting process for word : %s' % word)
            try:
                locs_and_data = neighbors(word, sentences_w_word, model, tokenizer, device)
            except TypeError as e:
                print(e)
                continue
            with open('static/jsons/%s.json' % word, 'w') as outfile:
                json.dump(locs_and_data, outfile)

            cur_len = locs_and_data['points'].shape[0]
            word_list += [word] * cur_len
            v = torch.from_numpy(locs_and_data['points'])
            s[pointer:pointer + cur_len] = v
            pointer = pointer + cur_len
            len_list.append(pointer)

    torch.save(s[:pointer], 's.pt')
    joblib.dump(word_list, 'word_list.pkl')
    joblib.dump(len_list, 'len_list.pkl')

    # Store an updated json with the filtered words.
    filtered_words = []
    for word in os.listdir('static/jsons'):
        word = word.split('.')[0]
        filtered_words.append(word)

    with open('static/filtered_words.json', 'w') as outfile:
        json.dump(filtered_words, outfile)
    print(filtered_words)


if __name__ == '__main__':
    main()
