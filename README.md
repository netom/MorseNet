# cwdecode

Neural network research project for decoding morse code

This repository contains experimental code. The goal of this project is to investigate
the possibility of using recurrent neural networks a connectionist temporal
classification (CTC) for decoding morse code messages from raw audio data.

Currently two of the Python scripts are useful:

## generate_wav_samples.py

Generates training samples according to information in config.py

## tensorflow_lstm_ctc.py

Builds and trains a simple neural network with LSTM unit(s) using CTC as the loss
function. The input is chunked raw audio.

## Usage

    $ mkdir training_set
    $ mkdir model_dir
    $ ./generate_wav_samples.py
    ...wait several minutes wile the files are generated
    $ ./tensorflow_lstm_ctc.py
    ...loads some of the training data and starts training / validation cycles

Training progress can be observed with tensorboard --log-dir=model_dir
