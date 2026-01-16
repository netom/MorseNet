# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neural network research project focused on decoding morse code from raw audio data using recurrent neural networks (RNN) with LSTM units and Connectionist Temporal Classification (CTC) loss. The project uses **TensorFlow 2.x with Keras API** and generates synthetic morse code training data.

## Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

Requirements: TensorFlow 2.19.1, SciPy, NumPy, PyAudio (for live decoding)

## Common Commands

### Training the Model

```bash
# Activate virtual environment first
source venv/bin/activate

# Start training (saves checkpoints to model_train/, logs to logs/)
./tensorflow_lstm_ctc_train.py

# Monitor training progress with TensorBoard
tensorboard --logdir=logs
```

Training configuration:
- Batch size: 250
- Batches per epoch: 60
- Max epochs: 1000 (can be interrupted with Ctrl+C)
- Model directory: `./model_train`
- Checkpoints kept: up to 30 (automatic management)
- Checkpoints saved: every 5 epochs

### Generating Training Data

```bash
# Generate WAV samples for debugging/validation
./generate_wav_samples.py <DIRNAME> <BATCHSIZE> [--length LENGTH]

# Example: Generate 100 samples of 12 seconds each
./generate_wav_samples.py training_data 100 --length 12
```

### Decoding Audio

```bash
# Decode a morse code WAV file (requires model_use/ directory with checkpoint)
./tensorflow_lstm_ctc_decode.py test.wav

# Use greedy decoder (faster but less accurate)
./tensorflow_lstm_ctc_decode.py test.wav --greedy
```

To use a trained model for decoding, copy checkpoint files from `model_train/` to `model_use/`:
- `checkpoint` file
- `ckpt-N.index` file
- `ckpt-N.data-*` file(s)

### Live Decoding

```bash
# Decode live audio from microphone
./tensorflow_lstm_ctc_live.py

# Configure audio device in config.py (DEVICE parameter)
```

## Architecture

### Migration to TensorFlow 2.x

The codebase has been migrated from TensorFlow 1.x (Estimator API) to TensorFlow 2.x (Keras API):
- **Old**: tf.estimator, tf.contrib.rnn, session-based training
- **New**: Keras Functional API, custom training loops, eager execution
- **Model architecture**: Now standardized to 3 LSTM layers across all scripts (train/decode/live)
- **Key improvement**: Fixed layer depth inconsistency from TF 1.x version

### Model Architecture

The neural network is defined in `model.py` with three main components:

1. **Input Layer**: Accepts audio chunks [batch, timesteps, features]
   - timesteps = 375 (for 12-second audio at 8kHz with 256-sample chunks)
   - features = 256 (CHUNK size)

2. **Recurrent Layers**: 3-layer LSTM with layer normalization
   - Each LSTM has 128 units
   - ReLU activation
   - Orthogonal weight initialization
   - Dropout 0.5 during training
   - Layer normalization after each LSTM (replaces deprecated LayerNormBasicLSTMCell)

3. **Output Layer**: Dense projection to NUM_CLASSES (38 morse characters)
   - Linear activation (no softmax - CTC handles this)

Total parameters: ~466k

### Training Pipeline

1. **Data Generation** (`generate_wav_samples.py`):
   - Generates synthetic morse code audio with randomized parameters:
     - Words per minute (WPM): 10-40
     - Timing deviation: 0-20%
     - White noise, QSB fading, frequency variation
     - Bandpass filtering, impulse noise
   - Uses multiprocessing (2 workers) for parallel generation
   - Output: normalized audio chunks + sparse character labels

2. **Model Training** (`tensorflow_lstm_ctc_train.py`):
   - Custom training loop with `@tf.function` for performance
   - CTC loss via `tf.nn.ctc_loss()`
   - L2 regularization (lambda=0.005) on non-bias weights
   - Gradient clipping (max norm=1.0)
   - Adam optimizer (learning_rate=0.001)
   - Metric: Label Error Rate (LER) via edit distance
   - TensorBoard logging (loss, CTC loss, L2 loss, LER, epoch time)
   - Automatic checkpoint management with `tf.train.CheckpointManager`

3. **Model Inference** (`tensorflow_lstm_ctc_decode.py`):
   - Loads checkpoint with `tf.train.Checkpoint`
   - Supports variable-length audio sequences
   - CTC beam search decoder (width=100) or greedy decoder
   - Outputs decoded text with log probability

4. **Live Decoding** (`tensorflow_lstm_ctc_live.py`):
   - PyAudio-based real-time audio capture
   - Circular buffer (12 seconds)
   - Continuous decoding every 500ms
   - Uses greedy decoder for speed

### Configuration System

All key parameters are centralized in `config.py`:
- `FRAMERATE`: Audio sampling rate (8000 Hz)
- `CHUNK`: Buffer size per timestep (256 samples)
- `SEQ_LENGTH`: Total sequence length for training (12 seconds worth)
- `TIMESTEPS`: Number of chunks in a sequence (375)
- `MORSE_CHR`: Character set (A-Z, 0-9, space, null terminator)
- `NUM_CLASSES`: 38 (length of character set)
- `CHARS`: Morse code dictionary (dots/dashes for each character)
- `DEVICE`: Audio device index for live decoding (0 by default)

### Key Technical Details

1. **CTC Loss**: Uses `tf.nn.ctc_loss()` with time-major format (timesteps first)
2. **Sparse Tensors**: Labels are represented as sparse tensors for CTC
3. **Gradient Tape**: Manual gradient computation for full control over training
4. **@tf.function**: Graph compilation for training steps (10-50x speedup)
5. **Checkpoint Format**: TensorFlow 2.x native format (not compatible with TF 1.x)
6. **Eager Execution**: Default in TF 2.x, makes debugging easier

### Data Pipeline

Uses `tf.data.Dataset.from_generator()` with the audio generator:
- Outputs: (audio, sparse_labels)
- Batching: 250 samples per batch
- Parallelism: `tf.data.AUTOTUNE` for automatic optimization
- Prefetching: Automatic with `AUTOTUNE`
- Epochs: 60 batches per epoch (generated on-the-fly)

## File Structure

- `config.py`: Centralized configuration (audio params, character sets, morse dictionary)
- `model.py`: **NEW** - Keras model architecture, CTC utilities
- `tensorflow_lstm_ctc_train.py`: Training script with custom training loop
- `tensorflow_lstm_ctc_decode.py`: Inference script for WAV files
- `tensorflow_lstm_ctc_live.py`: Live decoding script with microphone input
- `generate_wav_samples.py`: Synthetic morse code audio generator
- `model_train/`: Training checkpoints (TF 2.x format)
- `model_use/`: Production model checkpoints (manually copied from model_train/)
- `logs/`: TensorBoard logs (timestamped directories)
- `examples/`: Sample generated WAV files with labels
- `test*.wav`: Test audio files for validation

## Development Notes

### Fixed Issues from TF 1.x

1. **Layer depth inconsistency**: Now standardized to 3 LSTM layers everywhere
2. **Deprecated APIs**: Migrated to Keras API
3. **Layer normalization**: Now uses `tf.keras.layers.LayerNormalization` instead of `LayerNormBasicLSTMCell`
4. **Checkpoint compatibility**: New checkpoints are TF 2.x native format

### Training Tips

1. Monitor TensorBoard for loss curves: `tensorboard --logdir=logs`
2. Training can be interrupted with Ctrl+C (checkpoint saved automatically)
3. Checkpoints are saved every 5 epochs to `model_train/`
4. For faster training: reduce BATCH_SIZE or NUM_BATCHES_PER_EPOCH
5. For better accuracy: train longer, increase recurrent_layer_width

### Inference Tips

1. Beam search decoder (default) is more accurate but slower
2. Use `--greedy` flag for faster decoding with slightly lower accuracy
3. Audio must be 8kHz sample rate for best results
4. Longer audio sequences generally decode better

### Common Issues

1. **"No checkpoint found"**: Copy checkpoint files from model_train/ to model_use/
2. **Audio device error**: Update DEVICE in config.py (run script to see available devices)
3. **Out of memory**: Reduce BATCH_SIZE in training script
4. **GPU warnings**: Normal on CPU-only systems, training will use CPU

## Performance

- Training speed: ~1-2 minutes per epoch (CPU, batch size 250)
- Decoding speed: ~0.5-1 seconds per 12-second audio file (beam search)
- Live decoding: Real-time with 12-second buffer
- Model size: ~1.8 MB (466k parameters)
