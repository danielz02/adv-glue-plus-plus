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

import joblib
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.bert.modeling_bert import BertModel
import sqlite3 as sql
import re
import numpy as np
import umap
import json
from tqdm import tqdm
import nltk

from mpi4py import MPI


DB_PATH = './corpora/enwiki-20170820.db'
nltk.download('averaged_perceptron_tagger', download_dir="./corpora")
nltk.download('punkt', download_dir="./corpora/")
nltk.data.path = "./corpora/wordnet"


def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')


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
    points = []
    batch = []
    mask = []
    word_indices = []

    max_len = 512
    print('Getting embeddings for %d sentences ' % len(sentences))
    for sentence in tqdm(sentences):
        sentence = '</s> ' + sentence + ' </s>'  # Changed to LlaMA bos and eos tokens
        tokenized_text = tokenizer.tokenize(sentence)  # </s> is not added automatically when calling tokenizer.tokenize

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        if len(indexed_tokens) > max_len:
            indexed_tokens = indexed_tokens[:max_len]

        try:
            word_idx = tokenized_text.index(f"▁{word}")
        # If the word is made up of multiple tokens, just use the first one of the tokens that make it up.
        except ValueError as e:
            word_idx = None
            for i, token in enumerate(tokenized_text):
                # Match start of the word
                if token.startswith("▁"):
                    j = min(i + 1, len(tokenized_text))
                    while not tokenized_text[j].startswith("▁"):
                        j = min(j + 1, len(tokenized_text))
                        if j == len(tokenized_text):
                            break
                    if "".join(tokenized_text[i:j]) == f"▁{word}":
                        word_idx = i
                        break
            if not word_idx:
                print(e, word, tokenized_text)

        batch.append(indexed_tokens)
        mask.append([1] * len(indexed_tokens))
        word_indices.append(word_idx)

    tokens_tensor = pad_sequence([torch.from_numpy(np.array(x)) for x in batch], batch_first=True).to(device)
    masks_tensor = pad_sequence([torch.from_numpy(np.array(x)) for x in mask], batch_first=True).to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
        # (batch_size, sequence_length, hidden_size) * (1 + num_hidden_layers)
        batch_size = tokens_tensor.size(0)

        encoded_layers = torch.concat([
            model(
                input_ids=tokens_tensor[:(batch_size // 2)], attention_mask=masks_tensor[:(batch_size // 2)],
                output_hidden_states=True
            ).hidden_states[0],
            model(
                input_ids=tokens_tensor[(batch_size // 2):], attention_mask=masks_tensor[(batch_size // 2):],
                output_hidden_states=True
            ).hidden_states[0]
        ], dim=0)

    for i, idx in enumerate(word_indices):
        points.append(encoded_layers[i][idx].cpu().numpy())

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


def init_models():
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"rank {rank} device : {device}")

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="/scratch/bbkc/danielz/.cache/")
    # Load pre-trained model (weights)
    model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", cache_dir="/scratch/bbkc/danielz/.cache/")
    model.eval()
    model = model.to(device)

    return device, tokenizer, model


def main():
    s = np.memmap('./static/s.dat', mode="w+", shape=(13928506, 4096), dtype=np.float32)

    # Get selection of sentences from wikipedia.
    with open('./static/words.json', "r") as f:
        words = json.load(f)

    if os.path.exists("./static/sentences.pkl"):
        sentences = joblib.load("./static/sentences.pkl")
    else:
        sentences = get_sentences()
        joblib.dump(sentences, "./static/sentences.pkl")

    len_list = [0, ]
    word_list = []
    pointer = 0

    task_index = 0
    closed_workers = 0
    num_workers = size - 1

    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == tags.READY:
            if task_index < len(words):
                word = words[task_index]
                sentences_w_word = [t for t in sentences if ' ' + word + ' ' in t]

                # Take at most 200 sentences.
                sentences_w_word = sentences_w_word[:100]  # Changed from default

                # And don't show anything if there are less than 100 sentences.
                if len(sentences_w_word) > 50:  # Changed from default
                    print('starting process for word : %s' % word)
                    comm.send(
                        (word, sentences_w_word),
                        dest=source, tag=tags.START
                    )
            elif tag == tags.DONE:
                print(f"Got data from worker {source}")
                word, locs_and_data = data
                np.savez_compressed(f'./static/pickles/{word}.npz', **locs_and_data)
                cur_len = locs_and_data['points'].shape[0]
                s[pointer:(pointer + cur_len)] = locs_and_data['points']
                pointer = pointer + cur_len
                len_list.append(pointer)

                if (task_index % 1000) == 0:
                    s.flush()
            elif tag == tags.EXIT:
                print("Worker %d exited." % source)
                closed_workers += 1

    joblib.dump(word_list, './static/word_list.pkl')
    joblib.dump(len_list, './static/len_list.pkl')
    s.flush()
    del s

    # Store an updated json with the filtered words.
    filtered_words = []
    for word in os.listdir('static/jsons'):
        word = word.split('.')[0]
        filtered_words.append(word)

    with open('static/filtered_words.json', 'w') as outfile:
        json.dump(filtered_words, outfile)
    print(filtered_words)


def worker():
    name = MPI.Get_processor_name()
    print("I am a worker with rank %d on %s." % (rank, name))
    device, tokenizer, model = init_models()

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            word, sentences_w_word = data
            locs_and_data = neighbors(word, sentences_w_word, model, tokenizer, device)
            comm.send((word, locs_and_data), dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD  # get MPI communicator object
    size = comm.size  # total number of processes
    rank = comm.rank  # rank of this process
    local_rank = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]  # Only works for OpenMPI
    status = MPI.Status()  # get MPI status object

    if rank == 0:
        print(f"Running on {comm.size} cores")
        main()
    else:
        worker()
