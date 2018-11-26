import argparse
import json
import os

import numpy as np
from keras.layers import LSTM, Dropout, Dense, Activation, \
    Embedding
from keras.models import Sequential
from tqdm import tqdm_notebook

from model_definitions import load_weights

DATA_DIR = './data'


class SampleClass():
    def __init__(self, model_file):
        self.char_to_idx = os.path.join(
            os.path.dirname(os.path.dirname(model_file)),
            'char_to_idx.json')
        with open(self.char_to_idx) as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {i: ch for (ch, i) in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

        self.model_file = model_file
        self.model = self.build_sample_model(self.vocab_size)
        load_weights(self.model_file, self.model)

    def build_sample_model(self, vocab_size):
        model = Sequential()
        model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
        for i in range(2):
            model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
            model.add(Dropout(0.2))

        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))
        return model

    def sample(self, seed="", num_chars=100, random=True, multiple=True,
               probs=False, greedy=True, k=3, eos="DCNL"):

        sampled = [self.char_to_idx[c] for c in seed]

        self.init_model(sampled)

        if greedy:
            sampled, sampled_probs = self.greedy_sample(multiple, num_chars,
                                                        random,
                                                        sampled)
        else:
            sampled = sum(self.beamsearch(init_seed=sampled,
                                          k=k,
                                          eos=eos,
                                          num_chars=num_chars), [])

        sampled, sampled_str = self.idxs_to_chars(sampled)

        if not greedy:
            return sampled_str

        if probs:
            sampled.replace("\n", "0")

            return \
                ", ".join([str(l)[:3].rjust(3) for l in sampled_probs]) + "\n" + \
                ", ".join([str(l).rjust(3) for l in sampled]) + "\n" + \
                sampled_str

        return sampled_str

    def idxs_to_chars(self, sampled):
        sampled = ''.join(self.idx_to_char[c] for c in sampled)
        sampled_str = sampled \
            .replace("\n", "\n\n\n") \
            .replace("DCNL", "\n") \
            .replace("DCSP", "    ")
        return sampled, sampled_str

    def greedy_sample(self, multiple, num_chars, random, seed):

        sampled_probs = ['-' for c in seed]
        for i in range(num_chars):

            if seed:
                result = self.predict_char(seed[-1])
            else:
                result = self.predict_char(np.random.randint(self.vocab_size))
            if random:
                single_sample = np.random.choice(range(self.vocab_size),
                                                 p=result)
            else:
                single_sample = np.argmax(result)

            sampled_probs.append(str(result[single_sample]))
            seed.append(single_sample)
            if not multiple and (single_sample == 0):
                break

        return seed, sampled_probs

    def predict_char(self, char=None):
        batch = np.zeros((1, 1))
        if char:
            batch[0, 0] = char
        else:
            batch[0, 0] = np.random.randint(self.vocab_size)
        return self.model.predict_on_batch(batch).ravel()

    def init_model(self, sequence):
        self.model.reset_states()

        for c in sequence[:-1]:
            batch = np.zeros((1, 1))
            batch[0, 0] = c
            self.model.predict_on_batch(batch)

    def keras_rnn_predict(self, samples):
        """for every sample, calculate probability for every possible label
        """
        res = []
        for samp in samples:
            self.init_model(samp)

            if len(samp) == 0:
                res.append(self.predict_char())
            else:
                res.append(self.predict_char(samp[-1]))

        return np.array(res)

    def beamsearch(self, k=3, num_chars=100, init_seed=None, eos="DCNL"):
        """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
        all samples starts with init seed label and end with `eos` or truncated to length of `num_chars`.
        """

        if init_seed is None:
            init_seed = []
        dead_k = 0  # samples that reached eos
        dead_samples = []
        dead_scores = []
        live_k = 1  # samples that did not yet reached eos
        live_samples = [init_seed]
        live_scores = [0]
        eos = [self.char_to_idx[c] for c in eos]

        # pbar = tqdm_notebook(total=k)

        prev_dead_k = 0
        while live_k and dead_k < k:
            # for every possible live sample calc prob for every possible label
            probs = self.keras_rnn_predict(live_samples)

            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(live_scores)[:, None] - np.log(probs)
            cand_flat = cand_scores.flatten()

            # find the best (lowest) scores we have from all possible samples and new words
            ranks_flat = cand_flat.argsort()[:(k - dead_k)]
            live_scores = cand_flat[ranks_flat]

            # append the new words to their appropriate live sample
            voc_size = probs.shape[1]
            live_samples = [live_samples[r // voc_size] + [r % voc_size] for r
                            in
                            ranks_flat]

            # live samples that should be dead are...
            zombie = [s[-4:] == eos or len(s) >= num_chars for s in
                      live_samples]

            # add zombies to the dead
            dead_samples += [s for s, z in zip(live_samples, zombie) if
                             z]  # remove first label == empty
            dead_scores += [s for s, z in zip(live_scores, zombie) if z]
            dead_k = len(dead_samples)
            # remove zombies from the living
            live_samples = [s for s, z in zip(live_samples, zombie) if not z]
            live_scores = [s for s, z in zip(live_scores, zombie) if not z]
            live_k = len(live_samples)

            if dead_k > prev_dead_k:
                # pbar.update(dead_k - prev_dead_k)
                prev_dead_k = dead_k

        # pbar.close()
        return dead_samples + live_samples
