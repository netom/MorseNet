#!/usr/bin/env python3

"""
Inference script for morse code decoder using TensorFlow 2.x.

This script loads a trained model and decodes morse code from WAV audio files.
"""

from __future__ import generator_stop

import tensorflow as tf
import numpy as np
import scipy.io.wavfile
import sys
from pathlib import Path

# Suppress TensorFlow warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from config import *
from model import create_cw_model, ctc_decode, decoded_to_text


def load_model(checkpoint_dir='./model_use'):
    """
    Load model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing model checkpoints

    Returns:
        Loaded Keras model, or None if loading fails
    """
    # Create model with variable timesteps for inference
    model = create_cw_model(
        max_timesteps=None,  # Variable length for inference
        num_features=CHUNK,
        recurrent_layer_depth=3,  # STANDARDIZED to 3 layers
        recurrent_layer_width=128,
        num_classes=NUM_CLASSES
    )

    # Find latest checkpoint
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
        return None

    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        status = checkpoint.restore(latest_checkpoint)
        status.expect_partial()  # Ignore optimizer variables not needed for inference
        print(f"Loaded model from: {latest_checkpoint}")
    else:
        print(f"Warning: No checkpoint found in {checkpoint_dir}")
        print(f"Please copy trained model checkpoints to {checkpoint_dir}/")
        return None

    return model


def preprocess_audio(audio_data, framerate):
    """
    Preprocess audio data for inference.

    Args:
        audio_data: Raw audio samples from WAV file
        framerate: Sample rate of the audio

    Returns:
        Tuple of (preprocessed_audio, timesteps)
    """
    # Calculate number of complete chunks
    timesteps = len(audio_data) // CHUNK
    audio_data = audio_data[:timesteps * CHUNK]

    # Reshape into chunks
    audio = np.reshape(audio_data, (timesteps, CHUNK)).astype(np.float32)

    # Normalize (same as training)
    audio = (audio - np.mean(audio)) / np.std(audio)

    # Add batch dimension [batch, timesteps, features]
    audio = np.expand_dims(audio, axis=0)

    return audio, timesteps


def decode_audio(model, audio, timesteps, use_beam_search=True, beam_width=100):
    """
    Decode audio using the model.

    Args:
        model: Loaded Keras model
        audio: Preprocessed audio [1, timesteps, features]
        timesteps: Number of timesteps
        use_beam_search: Whether to use beam search (more accurate but slower)
        beam_width: Beam width for beam search

    Returns:
        Tuple of (decoded_text, log_probability)
    """
    # Forward pass
    logits = model(audio, training=False)

    # CTC decode
    decoded, log_prob = ctc_decode(
        logits,
        sequence_length=[timesteps],
        beam_width=beam_width,
        use_beam_search=use_beam_search
    )

    # Convert to text
    decoded_text = decoded_to_text(decoded[0], MORSE_CHR)

    return decoded_text[0], log_prob[0].numpy()


def main(args):
    """
    Main decoding function.

    Args:
        args: Command line arguments
    """
    if len(args) < 2:
        print("Usage: ./tensorflow_lstm_ctc_decode.py <filename.wav> [--greedy]")
        print("\nOptions:")
        print("  --greedy: Use greedy decoder (faster but less accurate)")
        print("            Default: use beam search with width=100")
        sys.exit(1)

    wav_file = args[1]
    use_beam_search = '--greedy' not in args

    print("="*70)
    print("Morse Code Decoder - TensorFlow 2.x")
    print("="*70)

    # Check if WAV file exists
    if not Path(wav_file).exists():
        print(f"Error: File '{wav_file}' not found")
        sys.exit(1)

    # Load WAV file
    print(f"Loading: {wav_file}")
    try:
        rate, data = scipy.io.wavfile.read(wav_file)
    except Exception as e:
        print(f"Error reading WAV file: {e}")
        sys.exit(1)

    if rate != FRAMERATE:
        print(f"Warning: WAV file sample rate {rate} Hz != expected {FRAMERATE} Hz")
        print(f"Results may be inaccurate. Consider resampling the audio.")

    # Preprocess
    audio, timesteps = preprocess_audio(data, rate)
    duration = len(data) / rate
    print(f"Audio length: {timesteps} timesteps ({duration:.2f} seconds)")

    # Load model
    print("\nLoading model...")
    model = load_model('./model_use')

    if model is None:
        print("\nError: Could not load model")
        print("\nTo decode audio:")
        print("1. Train a model: ./tensorflow_lstm_ctc_train.py")
        print("2. Copy checkpoint from model_train/ to model_use/:")
        print("   - checkpoint file")
        print("   - ckpt-N.index")
        print("   - ckpt-N.data-*")
        sys.exit(1)

    # Decode
    decoder_type = "beam search (width=100)" if use_beam_search else "greedy"
    print(f"Decoding with {decoder_type}...")

    decoded_text, log_prob = decode_audio(
        model, audio, timesteps,
        use_beam_search=use_beam_search,
        beam_width=100
    )

    # Display results
    print("\n" + "="*70)
    print("DECODED TEXT:")
    print("="*70)
    print(decoded_text if decoded_text else "(empty)")
    print("="*70)
    print(f"Log probability: {log_prob[0]:.4f}")
    print(f"Decoder: {decoder_type}")
    print("="*70)


if __name__ == "__main__":
    main(sys.argv)
