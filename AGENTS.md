# Copilot Instructions

## Project Overview

MorseNet is a TensorFlow 2.x research project for decoding Morse code from raw
audio using LSTM neural networks and CTC (Connectionist Temporal
Classification) loss. Training data is generated synthetically on-the-fly.

## Setup

```bash
# CPU-only (default):
. venv.sh

# With CUDA/GPU support:
WITH_CUDA=y . venv.sh
```

`venv.sh` creates/reuses `.venv`, installs dependencies from
`requirements.txt.sh`, sets `LD_LIBRARY_PATH` for NVIDIA libs, and overrides
`deactivate` to restore the original path. Must be *sourced* (`. venv.sh`), not
executed.

## Running the Scripts

```bash
# Train (saves checkpoints to model_train/ and TensorBoard logs to logs/):
./tensorflow_lstm_ctc_train.py

# Monitor training:
tensorboard --logdir=logs

# Decode a WAV file (loads from model_train/):
./tensorflow_lstm_ctc_decode.py test.wav
./tensorflow_lstm_ctc_decode.py test.wav --greedy   # currently has no effect

# Live microphone decoding (loads from model_use/):
./tensorflow_lstm_ctc_live.py

# Generate synthetic WAV samples:
./generate_wav_samples.py <output_dir> <count> [--length SECONDS]
```

There are no automated tests.

## Architecture

Shared configuration values are defined in **`config.py`**. Scripts import the
values they use; most model and training scripts use `from config import *`.

### Data pipeline (`generate_wav_samples.py`)
- Generates synthetic CW audio on-the-fly: a sine tone (500–700 Hz) with timing
jitter, white noise, impulse noise, QSB fading, and bandpass filtering.
- Timing constants: `CHUNK=256` samples/step, `FRAMERATE=8000` Hz →
`TIMESTEPS=375` steps per 12-second sample.
- `seq_generator()` is a Python generator that feeds a shared work queue filled
by worker processes or threads (controlled by `SAMPLE_GENERATOR_WORKER_TYPE`
and `SAMPLE_GENERATOR_WORKERS` in `config.py`).
- Normalization `(x - mean) / std` is applied per-sample before the model.

### Model (`model.py`)
- `create_cw_model()` returns a `tf.keras.Sequential`: configurable dense input
layers → stacked LSTM+LayerNormalization pairs → dense output layers.
- `ctc_decode()` wraps `tf.nn.ctc_beam_search_decoder`. It transposes logits
to time-major input and rotates the class axis so the blank class is last, as
required by TensorFlow's beam-search decoder.
- The CTC **blank index is `0`** (`MORSE_CHR[0]` is the `\0` sentinel).

### Training (`tensorflow_lstm_ctc_train.py`)
- `CTCTrainer` uses a manual `tf.GradientTape` loop — not `model.fit()`.
- Dense and LSTM kernels have L2 regularizers configured with
`L2_LAMBDA`; `train_step` currently optimizes CTC loss only.
- `train_step` does not clip gradients.
- Checkpoints saved every 5 epochs to `model_train/` via
`tf.train.CheckpointManager`.
- LER (Label Error Rate) is computed with `tf.edit_distance` using
`ctc_decode()` with beam width 1 during training for speed.

### Inference
- **`tensorflow_lstm_ctc_decode.py`**: loads the latest checkpoint from
`model_train/`, preprocesses WAV audio (reshape to `(timesteps, CHUNK)`,
normalize), and runs beam search (width=100) by default. The `--greedy` flag
currently does not change the decoding algorithm.
- **`tensorflow_lstm_ctc_live.py`**: uses PyAudio to capture a rolling
12-second audio buffer from `DEVICE` (configurable in `config.py`), and loads
from `model_use/`. Its current decode call passes an unsupported argument to
`ctc_decode()`, so live decoding fails when it first attempts to decode.

## Key Conventions

- **All tunable parameters live in `config.py`**. Never hardcode values that
belong there (framerate, chunk size, character set, batch size, paths, etc.).
- **Model checkpoint directories**: `model_train/` is the active training
directory; `model_use/` is the deployment directory. Promote checkpoints
manually by copying the `checkpoint`, `ckpt-N.index`, and `ckpt-N.data-*`
files.
- **`MORSE_CHR` order is canonical**: character-to-index mapping (`MORSE_ORD`)
and the CTC blank index (`0`) depend on the list order in `config.py`. Do not
reorder it.
- Audio input shape into the model: `[batch, TIMESTEPS, CHUNK]` (batch-major,
not time-major).
- The `generate_wav_samples.py` module is dual-purpose: importable library
(`seq_generator`, `generate_seq`) and a standalone CLI tool.
