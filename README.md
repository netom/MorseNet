# cwdecode

A neural network research project.

This repository contains experimental code. The goal of this project is 
to investigate the possibility of using recurrent neural networks and 
connectionist temporal classification (CTC) for decoding morse code 
messages from raw audio data.

Currently two of the Python scripts are useful:

## generate_wav_samples.py

It is both a command-line tool for debugging and a module for
generating data for training/validation.

For command line usage, issue: 

    $ ./generate_wav_samples.py --help"

## tensorflow_lstm_ctc_train.py

Builds and trains a simple neural network with LSTM unit(s) using CTC
as the loss function. The input is chunked raw audio. The model
checkpoints and training / evaluation events are saved in a directory
called "model_train".

Usage:

    $ ./tensorflow_lstm_ctc_train.py

Training progress can be observed with:

    $ tensorboard --log-dir=model_train

If you're particularly happy with one of the checkpoints, copy it 
manually into a directory called "model_use". To copy a checkpoint,
you have to copy a meta, an index, and a data file. The checkpoint
number and the type can be read in the files' names. A file called
"checkpoint" is also needed. See the one in "model_training". The
structure should be self-evident.

## tensorflow_lstm_ctc_decode.py

Restores the latest checkpoint from the directory "model_use", and
decodes a wav file given it's name as it's sole argument.

    $ ./tensorflow_lstm_ctc_decode.py test.wav

The decoded text will appear under a pile of TF INFO messages.
