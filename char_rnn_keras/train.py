import os
import json
import argparse

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

from model_definitions import build_model, save_weights

DATA_DIR = './data'

BATCH_SIZE = 128
SEQ_LENGTH = 512


class BatchReader(Sequence):
    def __init__(self, tokenized_dataset, vocab_size):
        self.length = tokenized_dataset.shape[0]
        self.batch_chars = self.length // BATCH_SIZE
        self.vocab_size = vocab_size
        self.text = tokenized_dataset

    def __len__(self):
        return (self.batch_chars - SEQ_LENGTH) // SEQ_LENGTH

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        index *= SEQ_LENGTH
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, self.vocab_size))
        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = self.text[
                    self.batch_chars * batch_idx + index + i]
                Y[batch_idx, i, self.text[
                    self.batch_chars * batch_idx + index + i + 1]] = 1
        return X, Y


def train(text, epochs=100, load_w=None):
    char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    vocab_size = len(char_to_idx)

    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    tokenized_text = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)

    mc = ModelCheckpoint('model/weights.{epoch:02d}-{loss:.2f}.hdf5',
                         monitor='loss')

    batch_gen = BatchReader(tokenized_text, vocab_size)

    if load_w is not None:
        model.load_weights(load_w)

    model.fit_generator(batch_gen, steps_per_epoch=len(batch_gen),
                        epochs=epochs, callbacks=[mc], workers=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt',
                        help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--load', default=None, type=str,
                        help='Load checkpoint')
    args = parser.parse_args()

    train(open(os.path.join(DATA_DIR, args.input)).read(), args.epochs,
          args.load)
