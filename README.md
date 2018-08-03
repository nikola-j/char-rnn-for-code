# char-rnn-keras

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [karpathy/char-rnn](https://github.com/karpathy/char-rnn) and [ekzhang/char-rnn](https://github.com/ekzhang/char-rnn-keras).

### Requirements

This code is written in Python 3, and it requires the [Keras](https://keras.io) deep learning library.

### Usage

To train the model with default settings:
```bash
$ python train.py
```

To sample the model:
```bash
$ python sample.py 100
```

The dataset can be downloaded from [here](https://github.com/EdinburghNLP/code-docstring-corpus/blob/master/V2/mono/mono_methods_bodies.gz)
