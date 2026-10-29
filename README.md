# cwdecode

A neural network research project.

This repository contains experimental code. The goal of this project is 
to investigate the possibility of using recurrent neural networks and 
connectionist temporal classification (CTC) for decoding morse code 
messages from raw audio data.

## Setup

Prepare the Python virtual environment and activate it:

    $ . venv.sh

This form does not install CUDA-enabled TensorFlow and disables GPU devices.

To install CUDA libraries and enable the use of GPUs, source venv.sh WITH_CUDA
set to a non-empty string:

    $ WITH_CUDA=y . venv.sh

venv.sh also sets up LD_LIBRARY_PATH. Running the "deactivate" command will
restore LD_LIBRARY_PATH.

## generate_wav_samples.py

It is both a command-line tool for debugging and a module for
generating data for training/validation.

For command line usage, issue:

    $ ./generate_wav_samples.py --help

The tool writes `.wav` audio, `.npy` audio, and `.txt` transcripts to the
output directory. Use `--length SECONDS` to set the approximate sample length.
Training reads matching `.npy` audio and `.txt` transcript files from
`TRAINING_DATA_DIR` (default: `./samples`).

## tensorflow_lstm_ctc_train.py

Builds and trains a simple neural network with LSTM unit(s) using CTC
as the loss function. The input is chunked raw audio. The model
checkpoints and training / evaluation events are saved in a directory
called "model_train".

Usage:

    $ ./tensorflow_lstm_ctc_train.py

By default, training uses the saved samples in `./samples`. Generate a corpus
before training if needed:

    $ ./generate_wav_samples.py ./samples 1000

Set `TRAINING_DATA_DIR` in `config.py` to use another directory. Set
`TRAINING_DATA_SOURCE = 'generated'` to use the original on-the-fly synthetic
generator instead; the default `'saved'` mode does not start generator workers.
The loader requires each `.npy` audio file to have the configured sample
length and a `.txt` transcript with the same filename stem. The dataset is
shuffled and repeated to provide `NUM_BATCHES_PER_EPOCH`, so a small corpus is
reused within an epoch. `BATCH_SIZE` remains configurable, and the corpus does
not need to be an exact multiple of it.

For comparisons between input modes, set separate `CHECKPOINT_DIR` and
`LOG_DIR` values so training artifacts are not mixed.

Training progress can be observed with:

    $ tensorboard --logdir=logs

Checkpoints are saved in `model_train`.

## tensorflow_lstm_ctc_decode.py

Restores the latest checkpoint from the directory "model_train", and
decodes a wav file given it's name as it's sole argument.

    $ ./tensorflow_lstm_ctc_decode.py test.wav

The decoded text will appear under a pile of TF INFO messages.

The script uses beam search with width 100 by default. Its `--greedy` option
will reduce the beam width to 1, wich is equivalent to greedy decoding.

## tensorflow_lstm_ctc_live.py

Captures live audio from the device configured by `DEVICE` in `config.py` and
loads a checkpoint from `model_use`:

    $ ./tensorflow_lstm_ctc_live.py

Live decoding currently fails when it first tries to decode because the script
passes an unsupported argument to the decoder.

## Running with NVidia MPS

See: https://docs.nvidia.com/deploy/mps/latest/index.html

Start the MPS control process. To observe logs real-time, start it in the
foreground:

    $ nvidia-cuda-mps-control -f

Set up the environment in another terminal:

    $ export CUDA_VISIBLE_DEVICES=0
    $ export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=2G
    $ ./tensorflow_lstm_ctc_train.py

You can safely run inference in an other terminal with less resources:

    $ export CUDA_VISIBLE_DEVICES=0
    $ export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=512M
    $ ./tensorflow_lstm_ctc_decode.py test.wav
