# Copilot Instructions

## Project Overview

MorseNet is a TensorFlow 2.x research project for decoding Morse code from raw
audio using LSTM neural networks and CTC (Connectionist Temporal
Classification) loss. Synthetic data is generated ahead of time into
`dataset/train/` and `dataset/test/`; training uses the first split and
evaluates on the second.

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
# Train (saves checkpoints to model/ and TensorBoard logs to logs/):
./train.py

# Monitor training:
tensorboard --logdir=logs

# Decode a WAV file (loads from model/):
./decode.py test.wav
./decode.py test.wav --greedy

# Live microphone decoding (loads from CHECKPOINT_DIR unless --checkpoint-dir is given):
./decode_live.py

# Generate synthetic WAV samples:
./dataset.py <to_index> [--split train|test] [--from-index N] [--dirname DIR] [--length SECONDS]
```

There are no automated tests.

## Architecture

Shared configuration values are defined in **`config.py`**. Scripts import the
values they use; most model and training scripts use `from config import *`.

### Data pipeline (`dataset.py`)
- Generates synthetic CW audio: a sine tone (500–700 Hz) with timing
jitter, white noise, impulse noise, QSB fading, and bandpass filtering.
- Timing constants: `CHUNK=256` samples/step, `FRAMERATE=8000` Hz →
`TIMESTEPS=375` steps per 12-second sample.
- `save_files()` writes each sample as `.npy` (float32 audio), `.wav` and
`.txt` (transcript) into the output directory. `train.py` loads the `.npy`/`.txt`
pairs from `TRAINING_DATA_DIR` and evaluates with `TEST_DATA_DIR`; there is no
on-the-fly generation. `--split test` selects the test directory when
generating samples.
- Normalization `(x - mean) / std` is applied per-sample before the model.

### Model (`model.py`)
- `create_cw_model()` returns a `tf.keras.Sequential`: configurable dense input
layers → stacked LSTM+LayerNormalization pairs → dense output layers.
- `ctc_decode()` wraps `tf.nn.ctc_beam_search_decoder`. It transposes logits
to time-major input and rotates the class axis so the blank class is last, as
required by TensorFlow's beam-search decoder.
- The CTC **blank index is `0`** (`MORSE_CHR[0]` is the `\0` sentinel).

### Training (`train.py`)
- `CTCTrainer` uses a manual `tf.GradientTape` loop — not `model.fit()`.
- Dense and LSTM kernels have L2 regularizers configured with
`L2_LAMBDA`; `train_step` currently optimizes CTC loss only.
- `train_step` does not clip gradients.
- Checkpoints saved every 5 epochs to `model/` via
`tf.train.CheckpointManager`.
- LER (Label Error Rate) is computed with `tf.edit_distance` using
`ctc_decode()` with beam width 1 during training for speed.
- Each epoch evaluates test loss and LER without applying gradients.
- Samples are stored in `dataset/train/` and `dataset/test/`; generation uses
`--split train` or `--split test`, and test samples are evaluated once per epoch.

### Inference
- **`decode.py`**: loads the latest checkpoint from
`model/`, preprocesses WAV audio (reshape to `(timesteps, CHUNK)`,
normalize), and runs beam search (width=100) by default. `--greedy` uses beam
width 1.
- **`decode_live.py`**: uses PyAudio to capture live audio from `DEVICE`
(configurable in `config.py`) and decodes it greedily in a streaming fashion,
retaining LSTM and CTC state between blocks. It loads from `CHECKPOINT_DIR`
by default (`--checkpoint-dir` overrides it).

## Key Conventions

- **All tunable parameters live in `config.py`**. Never hardcode values that
belong there (framerate, chunk size, character set, batch size, paths, etc.).
- **Model checkpoint directories**: `model/` is the active training
directory; `model_use/` is the deployment directory. Promote checkpoints
manually by copying the `checkpoint`, `ckpt-N.index`, and `ckpt-N.data-*`
files.
- **`MORSE_CHR` order is canonical**: character-to-index mapping (`MORSE_ORD`)
and the CTC blank index (`0`) depend on the list order in `config.py`. Do not
reorder it.
- Audio input shape into the model: `[batch, TIMESTEPS, CHUNK]` (batch-major,
not time-major).
- The `dataset.py` module is dual-purpose: importable library
(`generate_seq`, `save_files`) and a standalone CLI tool.
