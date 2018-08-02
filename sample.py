import argparse
import json
import os

import numpy as np
from keras.layers import LSTM, Dropout, Dense, Activation, \
    Embedding
from keras.models import Sequential

from .model_definitions import load_weights

DATA_DIR = './data'


def build_sample_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model


def sample(file, seed="", num_chars=100, random=True, multiple=True,
           probs=False):
    char_to_idx = os.path.join(os.path.dirname(os.path.dirname(file)),
                               'data',
                               'char_to_idx.json')
    with open(char_to_idx) as f:
        char_to_idx = json.load(f)
    idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    model = build_sample_model(vocab_size)
    load_weights(file, model)

    sampled = [char_to_idx[c] for c in seed]
    sampled_probs = ['-' for c in seed]

    for c in seed[:-1]:
        batch = np.zeros((1, 1))
        batch[0, 0] = char_to_idx[c]
        model.predict_on_batch(batch)

    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        if random:
            single_sample = np.random.choice(range(vocab_size), p=result)
        else:
            single_sample = np.argmax(result)

        sampled_probs.append(str(result[single_sample]))
        sampled.append(single_sample)
        if not multiple and (single_sample == 0):
            break

    sampled = ''.join(idx_to_char[c] for c in sampled)

    sampled_str = sampled \
        .replace("\n", "\n\n\n") \
        .replace("DCNL", "\n") \
        .replace("DCSP", "    ")

    if probs:
        sampled.replace("\n", "0")

        return \
            ", ".join([str(l)[:3].rjust(3) for l in sampled_probs]) + "\n" + \
            ", ".join([str(l).rjust(3) for l in sampled]) + "\n" + \
            sampled_str

    return sampled_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample some text from the trained model.')
    parser.add_argument('file', type=str, help='file checkpoint to sample from')
    parser.add_argument('--seed', default='',
                        help='initial seed for the generated text')
    parser.add_argument('--len', type=int, default=512,
                        help='number of characters to sample (default 512)')
    parser.add_argument('--not_random', action='store_false')
    args = parser.parse_args()

    print(sample(args.file, args.seed, args.len, args.not_random))
