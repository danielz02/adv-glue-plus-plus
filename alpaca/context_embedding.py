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
    # Tokenized input_embedding
    points = []
    batch = []
    mask = []
    word_indices = []

    max_len = 512
    for sentence in sentences:
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
                    while j < len(tokenized_text) and not tokenized_text[j].startswith("▁"):
                        j = min(j + 1, len(tokenized_text))
                        if j == len(tokenized_text):
                            break
                    if "".join(tokenized_text[i:j]) == f"▁{word}":
                        word_idx = i
                        break
            if not word_idx:
                print(e, word, tokenized_text)
                continue

        batch.append(indexed_tokens)
        mask.append([1] * len(indexed_tokens))
        word_indices.append(word_idx)

    if len(batch) == 0 or len(mask) == 0:
        return None

    tokens_tensor = pad_sequence([torch.from_numpy(np.array(x)) for x in batch], batch_first=True).to(device)
    masks_tensor = pad_sequence([torch.from_numpy(np.array(x)) for x in mask], batch_first=True).to(device)

    # Predict hidden states features for each layer
    with torch.no_grad():
        # (batch_size, sequence_length, hidden_size) * (1 + num_hidden_layers)
        num_batches = 8
        batch_size = tokens_tensor.size(0) // num_batches
        encoded_layers = []
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size if batch_num < num_batches - 1 else tokens_tensor.size(0)
            encoded_layers.append(
                model.model(
                    input_ids=tokens_tensor[batch_start:batch_end], attention_mask=masks_tensor[batch_start:batch_end],
                    output_hidden_states=False
                ).last_hidden_state,
            )
        encoded_layers = torch.cat(encoded_layers, dim=0)

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
        except ValueError:
            pos_tag = 'X'
        sent_data.append({
            'sentence': sent,
            'pos': pos_tag
        })

    return sent_data


def init_models():
    # FIXME: This only works with the current rankfile
    gpu_dev = int(local_rank) - 1 if rank <= 4 else int(local_rank)
    device = torch.device(f"cuda:{gpu_dev}" if torch.cuda.is_available() else "cpu")
    print(f"rank {rank} device : {device}")

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")
    # Load pre-trained model (weights)
    model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-native", cache_dir="./.cache/")
    model.eval()
    model = model.to(device=device, dtype=torch.bfloat16)

    return device, tokenizer, model


def load_sentences():
    with open('./static/words.json', "r") as f:
        words = json.load(f)

    if os.path.exists("./static/sentences.pkl"):
        sentences = joblib.load("./static/sentences.pkl")
    elif rank == 0:
        sentences = get_sentences()
        joblib.dump(sentences, "./static/sentences.pkl")
        comm.barrier()
    else:
        comm.barrier()  # waiting for rank 0 to get sentences
        sentences = joblib.load("./static/sentences.pkl")

    return words, sentences


def main():
    # Get selection of sentences from wikipedia.
    words, sentences = load_sentences()

    closed_workers = 0
    num_workers = size - 1

    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        print(f"[{rank}] Received data from worker {source} with tag {tag}...")

        if tag == tags.READY:
            print(f"[{rank}] Sending tasks to worker {source}...")
            num_tasks_per_worker = len(words) // num_workers
            tasks = np.arange(
                (source - 1) * num_tasks_per_worker,
                source * num_tasks_per_worker if source < num_workers else len(words)
            )
            comm.send(tasks, dest=source, tag=tags.START)
            print(f"[{rank}] Sent {len(tasks)} tasks to worker {source}")
        elif tag == tags.DONE:
            print(f"[{rank}] Worker {source} is done")
        elif tag == tags.EXIT:
            print(f"[{rank}] Worker {source} exited.")
            closed_workers += 1

    # s = np.memmap('./static/s.dat', mode="w+", shape=(13928506, 4096), dtype=np.float32)
    s = np.zeros(shape=(13928506, 4096), dtype=np.float32)
    len_list = [0, ]
    word_list = []
    pointer = 0
    for word in words:
        f = f"./static/pickles/{word}.npz"
        if not os.path.exists(f):
            continue
        locs_and_data = np.load(f)
        cur_len = locs_and_data['points'].shape[0]
        word_list += [word] * cur_len
        s[pointer:(pointer + cur_len)] = locs_and_data["points"]
        pointer = pointer + cur_len
        len_list.append(pointer)
    s = s[:pointer]
    len_list = np.array(len_list)
    word_list = np.array(word_list)
    print(len(s), pointer)
    np.save("./static/s.npy", s)
    np.save('./static/word_list.npy', word_list)
    np.save('./static/len_list.npy', len_list)

    # Store an updated json with the filtered words.
    filtered_words = []
    for word in os.listdir('./static/pickles'):
        word = word.split('.')[0]
        filtered_words.append(word)

    with open('./static/filtered_words.json', 'w') as outfile:
        json.dump(filtered_words, outfile)
    print(filtered_words)
    print("Master finishing")


def worker():
    name = MPI.Get_processor_name()
    print(f"[{rank}] I am a worker with rank {rank} on {name}.")
    device, tokenizer, model = init_models()
    words, sentences = load_sentences()

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        print(f"[{rank}] Worker {rank} on {name} is ready to receive tasks.")
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            print(f'[{rank}] Received {len(data)} tasks')
            for task_index in tqdm(data, desc=f"Rank {rank}"):
                word = words[task_index]

                if os.path.exists(f'./static/pickles/{word}.npz'):
                    print(f'[{rank}] Skipping {word}')
                    continue

                # Take at most 200 sentences.
                sentences_w_word = [t for t in sentences if ' ' + word + ' ' in t]
                sentences_w_word = sentences_w_word[:1000]  # Changed from default

                # And don't show anything if there are less than 100 sentences.
                if len(sentences_w_word) < 100:  # Changed from default
                    continue
                try:
                    locs_and_data = neighbors(word, sentences_w_word, model, tokenizer, device)
                    np.savez(f'./static/pickles/{word}.npz', **locs_and_data)
                except IndexError as e:
                    print(e)
                except ValueError as e:
                    print(e)
                print(f'[{rank}] Finished processing for word : {word}')
            comm.send(None, dest=0, tag=tags.DONE)
            break
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
    MPI.Finalize()
