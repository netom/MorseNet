# cwdecode

A neural network research project.

This repository contains experimental code. The goal of this project is to investigate
the possibility of using recurrent neural networks and connectionist temporal
classification (CTC) for decoding morse code messages from raw audio data.

Currently two of the Python scripts are useful:

## generate_wav_samples.py

It is both a command-line tool for debugging and a module for generating data
for training/validation.

For command line usage, issue: 

    $ ./generate_wav_samples.py --help"

## tensorflow_lstm_ctc.py

Builds and trains a simple neural network with LSTM unit(s) using CTC as the loss
function. The input is chunked raw audio.

Usage:

    $ ./tensorflow_lstm_ctc.py

Training progress can be observed with:

    $ tensorboard --log-dir=model_dir
