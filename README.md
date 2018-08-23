# char-rnn-keras

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [karpathy/char-rnn](https://github.com/karpathy/char-rnn) and [ekzhang/char-rnn](https://github.com/ekzhang/char-rnn-keras).

Trained on a dataset of python code functions.

### Requirements

This code is written in Python 3, and it requires the [Keras](https://keras.io) deep learning library.

```bash
pip3 install requirements.txt
```

### Usage

To train the model with default settings use:
```bash
$ python train.py
```

To use the model use the notebook.

The dataset can be downloaded from [here](https://github.com/EdinburghNLP/code-docstring-corpus/blob/master/V2/mono/mono_methods_bodies.gz)
