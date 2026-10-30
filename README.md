# MorseNet

A neural network research project.

This repository contains experimental code. The goal of this project is to
investigate the possibility of using recurrent neural networks and
connectionist temporal classification (CTC) for decoding morse code messages
from raw audio data.

## Setup

Prepare the Python virtual environment and activate it:

    $ . venv.sh

This form does not install CUDA-enabled TensorFlow and disables GPU devices.

To install CUDA libraries and enable the use of GPUs, source venv.sh WITH_CUDA
set to a non-empty string:

    $ WITH_CUDA=y . venv.sh

`venv.sh` also sets up `LD_LIBRARY_PATH`. Running `deactivate` restores its
previous value.

## dataset.py

This command-line tool generates synthetic training samples.

For command line usage, issue:

    $ ./dataset.py --help

For example, to generate training samples numbered 0 to 999 in
`./dataset/train`:

    $ ./dataset.py 1000

To generate 200 test samples numbered 0 to 199 in `./dataset/test`:

    $ ./dataset.py 200 --split test

To add training samples numbered 1000 to 1999:

    $ ./dataset.py 2000 --from-index 1000

The tool writes `.wav` audio, `.npy` audio, and `.txt` transcripts to the
directory for `--split` (`train` or `test`; defaults to `train`). `--dirname`
overrides that directory. Existing files with the same indices are overwritten.
Use `--length SECONDS` to set the sample duration. Training expects the
configured `SEQ_LENGTH_SECONDS` (12 seconds by default).

## train.py

Builds and trains an LSTM neural network with CTC loss on chunked raw audio.
Model checkpoints are saved to `CHECKPOINT_DIR` (default: `./model`);
TensorBoard events are saved to `LOG_DIR` (default: `./logs`).

Usage:

    $ ./train.py

By default, training uses `./dataset/train` and evaluates on `./dataset/test`.
Generate data for both splits before training:

    $ ./dataset.py 1000 --split train
    $ ./dataset.py 200 --split test

Set `TRAINING_DATA_DIR` and `TEST_DATA_DIR` in `config.py` to use other
directories. Samples are parsed and batched by a `tf.data` pipeline with
parallel file reading. Each `.npy` audio file must contain exactly the
configured sample length as float32
(as written by `dataset.py`) and have a `.txt` transcript with the
same filename stem. Invalid sample sizes or transcript characters stop loading
with an error identifying the audio file; missing or unmatched transcripts are
reported before training starts. The training dataset is shuffled and repeated
to provide `NUM_BATCHES_PER_EPOCH`, so a small corpus is reused within an epoch.
Each test sample is evaluated once per epoch without updating model weights,
including any final partial batch. `BATCH_SIZE` remains configurable, and
neither split needs to be an exact multiple of it. Both split directories must
exist and contain matched `.npy`/`.txt` pairs.

Training progress can be observed with:

    $ tensorboard --logdir=logs

Checkpoints are saved in `model`. Restarting with the same
`CHECKPOINT_DIR` and `LOG_DIR` resumes the epoch counter and continues the
TensorBoard charts. An interrupted, incomplete epoch is rerun from its
beginning. For older checkpoints without an epoch counter, completed epochs are
estimated from saved batch steps using the current `NUM_BATCHES_PER_EPOCH`.
TensorBoard also records `test_loss` and `test_ler` for the held-out split.

## decode.py

Restores the latest checkpoint from `CHECKPOINT_DIR` (default: `./model`) and
decodes a WAV file.
Pass `--greedy` to use a beam width of 1 instead of the default 100.

    $ ./decode.py test.wav

The script may print TensorFlow informational messages before the decoded text.
`--greedy` sets beam width to 1; decoding still uses TensorFlow's beam-search
decoder.

## decode_live.py

The live decoder processes only new audio while retaining each LSTM's
hidden/cell state and the CTC previous-class state:

    $ python decode_live.py
    $ python decode_live.py --checkpoint-dir ./model_use
    $ python decode_live.py --list-devices
    $ python decode_live.py --device 3

It defaults to `CHECKPOINT_DIR` and `DEVICE` in `config.py`, captures at the
device's default rate, and uses streaming anti-aliasing resampling. It appends
greedy CTC output immediately, without a 12-second buffer or confirmation
delay. `LIVE_STATEFUL_BLOCK_SECONDS` controls capture block duration; it
defaults to 0.5 seconds to allow scheduling headroom for capture.
`LIVE_NORMALIZATION_SECONDS` controls causal normalization adaptation. PyAudio
must be installed.

Streaming normalization differs from training's per-sample normalization, and
long-lived LSTM state differs from training on independent samples, so this is
experimental and is not guaranteed to improve recognition. Restarting the
script resets its network and CTC state. Audio capture runs independently of
inference, with a bounded queue controlled by `LIVE_STATEFUL_QUEUE_SECONDS`. If
input overflows or that queue fills, decoding reports the loss and resets
network, CTC, and preprocessing state rather than carrying state across a gap.
Queued blocks are processed together to catch up after temporary delays.
`LIVE_STATEFUL_CPU_THREADS` defaults to 1 to reduce CPU scheduling contention
with audio capture. Loss warnings distinguish PortAudio input overflow from an
application queue overflow; an additional warning indicates if inference takes
longer than the audio it processes. Persistent PortAudio input overflow occurs
before the application queue; increasing the queue alone cannot fix it. On
Linux, try an available `pulse` or `default` input instead of a raw `hw:`
device using `--list-devices` and `--device`. Capture block size and actual
input latency are printed at startup to help diagnose device buffering.

## Running with NVIDIA MPS

See: https://docs.nvidia.com/deploy/mps/latest/index.html

Start the MPS control process. To observe logs real-time, start it in the
foreground:

    $ nvidia-cuda-mps-control -f

Set up the environment in another terminal:

    $ export CUDA_VISIBLE_DEVICES=0
    $ export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=2G
    $ ./train.py

Inference can also be run in another terminal with a smaller MPS memory limit:

    $ export CUDA_VISIBLE_DEVICES=0
    $ export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=512M
    $ ./decode.py test.wav
